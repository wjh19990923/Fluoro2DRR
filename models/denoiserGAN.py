import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.utils as vutils
import torch.optim as optim
import lpips
import piq

from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


# ----------------- Generator (DeepLabV3 for regression) -----------------
class Denoiser(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Use the new weights interface to avoid deprecated warnings
        self.model_deeplabv3 = deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.DEFAULT)

        # Modify the first layer to single-channel input
        original_conv = self.model_deeplabv3.backbone.conv1
        new_conv = nn.Conv2d(
            in_channels, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding, bias=False
        )
        with torch.no_grad():
            new_conv.weight.copy_(
                original_conv.weight.mean(dim=1, keepdim=True))
        self.model_deeplabv3.backbone.conv1 = new_conv

        # Modify the classification head to 1 channel
        self.model_deeplabv3.classifier = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )

        self.l1_loss = nn.L1Loss()
        self.lpips_loss_fn = lpips.LPIPS(net='alex')

    def forward(self, x):
        out = self.model_deeplabv3(x)['out']
        out = F.interpolate(
            out, size=x.shape[2:], mode="bilinear", align_corners=False)
        out = torch.sigmoid(out)  # Output in [0,1]
        return out

    def perceptual_loss(self, pred, target):
        # pred/target ∈ [0,1]
        l1 = self.l1_loss(pred, target)
        ssim_l = 1 - piq.ssim(pred, target, data_range=1.0)
        lp = self.lpips_loss_fn.to(pred.device)(
            pred.repeat(1, 3, 1, 1) * 2 - 1,
            target.repeat(1, 3, 1, 1) * 2 - 1
        ).mean()
        return self.alpha * l1 + self.beta * ssim_l + self.gamma * lp


# ----------------- Discriminator (PatchGAN) -----------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()

        def block(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_channels, base_channels, norm=False),
            block(base_channels, base_channels * 2),
            block(base_channels * 2, base_channels * 4),
            block(base_channels * 4, base_channels * 8),
            nn.Conv2d(base_channels * 8, 1, 4, padding=1)  # (B,1,H/16,W/16)
        )

    def forward(self, x):
        return self.model(x)


# ----------------- Lightning GAN (manual optimization) -----------------
class DenoiserGAN(pl.LightningModule):
    def __init__(self, in_channels=1, out_channels=1, lr_g=1e-4, lr_d=1e-4, adv_weight=1e-3,
                 alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Denoiser(
            in_channels, out_channels, alpha=alpha, beta=beta, gamma=gamma)
        self.discriminator = PatchDiscriminator(in_channels)
        self.adv_loss_fn = nn.BCEWithLogitsLoss()

        # Key: multiple optimizers -> manual optimization
        self.automatic_optimization = False

    def forward(self, x):
        return self.generator(x)

    @torch.no_grad()
    def _log_images(self, noisy, denoised, clean, tag_prefix="val"):
        # Resolve half clamp error: force conversion to float32
        noisy = noisy.detach().float().cpu()
        denoised = denoised.detach().float().cpu()
        clean = clean.detach().float().cpu()

        n = min(4, noisy.size(0))
        idx = torch.randperm(noisy.size(0))[:n]

        grid_noisy = vutils.make_grid(noisy[idx], nrow=2, normalize=True)
        grid_denoised = vutils.make_grid(denoised[idx], nrow=2, normalize=True)
        grid_clean = vutils.make_grid(clean[idx], nrow=2, normalize=True)

        tb = getattr(self.logger, "experiment", None)
        if tb is not None:
            tb.add_image(f"{tag_prefix}/Noisy", grid_noisy, self.current_epoch)
            tb.add_image(f"{tag_prefix}/Denoised",
                         grid_denoised, self.current_epoch)
            tb.add_image(f"{tag_prefix}/Clean", grid_clean, self.current_epoch)

    def training_step(self, batch, batch_idx):
        images, *_ = batch
        noisy = images[:, 0:1]
        clean = images[:, 1:2]

        opt_g, opt_d = self.optimizers()

        # --------- 1) Update discriminator D ---------
        self.toggle_optimizer(opt_d)
        fake = self(noisy).detach()
        pred_real = self.discriminator(clean)
        pred_fake = self.discriminator(fake)
        d_real = self.adv_loss_fn(pred_real, torch.ones_like(pred_real))
        d_fake = self.adv_loss_fn(pred_fake, torch.zeros_like(pred_fake))
        d_loss = 0.5 * (d_real + d_fake)

        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)
        self.log("train/d_loss", d_loss, prog_bar=True,
                 on_step=True, on_epoch=True)

        # --------- 2) Update generator G ---------
        self.toggle_optimizer(opt_g)
        fake = self(noisy)
        pred_fake_for_g = self.discriminator(fake)
        adv_loss = self.adv_loss_fn(
            pred_fake_for_g, torch.ones_like(pred_fake_for_g))
        perc_loss = self.generator.perceptual_loss(fake, clean)
        g_loss = perc_loss + self.hparams.adv_weight * adv_loss

        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        self.log("train/g_loss", g_loss, prog_bar=True,
                 on_step=True, on_epoch=True)
        self.log("train/perc_loss", perc_loss, on_step=True, on_epoch=True)
        self.log("train/adv_loss", adv_loss, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        images, *_ = batch
        noisy = images[:, 0:1]
        clean = images[:, 1:2]

        with torch.no_grad():
            denoised = self(noisy)
            perc = self.generator.perceptual_loss(denoised, clean)

        self.log("val/perc_loss", perc, prog_bar=True,
                 on_step=False, on_epoch=True)
        if batch_idx == 0:
            self._log_images(noisy, denoised, clean, tag_prefix="val")

    def configure_optimizers(self):
        opt_g = optim.Adam(self.generator.parameters(),
                           lr=self.hparams.lr_g, betas=(0.5, 0.999))
        opt_d = optim.Adam(self.discriminator.parameters(),
                           lr=self.hparams.lr_d, betas=(0.5, 0.999))
        return [opt_g, opt_d]
