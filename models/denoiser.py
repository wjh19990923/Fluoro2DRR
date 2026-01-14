import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet50
import torchvision.utils as vutils
from air_network.tools.losses import ncc_loss, gcc_loss, mNCC_loss_parallel, mGCC_loss_parallel
import piq
import lpips  # pip install lpips
from piq import DISTS

# Initialize LPIPS loss function once to avoid re-initialization overhead
lpips_loss_fn = lpips.LPIPS(net='alex')


def ssim_loss(pred, target):
    return 1 - piq.ssim(pred, target, data_range=1.0)


class PretrainedDenoiseDeepLabv3(pl.LightningModule):
    def __init__(self, in_channels=1, out_channels=1, lr=1e-3,
                 alpha=0.5, beta=0.3, gamma=0.2, delta=0., epsilon=0.):
        """
        Use a pretrained DeepLabV3 model for image denoising and train with a mixed loss.
        Parameters:
            in_channels: Number of input channels (set to 1, as we use the noisy image, i.e., the second channel, as input)
            out_channels: Number of output channels (set to 1, corresponding to the clean X-ray image)
            lr: Learning rate
            alpha: Weight for L1 loss
            beta: Weight for SSIM loss
            gamma: Weight for LPIPS loss
            delta: Weight for DISTS loss
        """
        super(PretrainedDenoiseDeepLabv3, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon  # for similarity metric

        # Load the pretrained DeepLabV3 model (default weights trained on COCO)
        model = deeplabv3_resnet50(pretrained=True)

        # Modify backbone.conv1 to change the input from 3 channels to 1 channel
        original_conv = model.backbone.conv1
        new_conv = nn.Conv2d(in_channels,
                             original_conv.out_channels,
                             kernel_size=original_conv.kernel_size,
                             stride=original_conv.stride,
                             padding=original_conv.padding,
                             bias=False)
        with torch.no_grad():
            new_conv.weight.copy_(
                original_conv.weight.mean(dim=1, keepdim=True))
        model.backbone.conv1 = new_conv

        # Modify the classifier: construct a simple head to output a 1-channel image
        model.classifier = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )

        self.model = model
        self.l1_loss = nn.L1Loss()
        # Initialize the DISTS loss function (using default parameters, with data_range set to 1)
        self.dists_loss_fn = DISTS(reduction='none')
        # Step counter for logging images
        self.log_image_step = 0

    def forward(self, x):
        """
        Forward pass:
            x: Input noisy X-ray image, shape (batch, 1, H, W)
            Returns: Output denoised image, shape (batch, 1, H, W)
        """
        out = self.model(x)['out']
        out = F.interpolate(
            out, size=x.shape[2:], mode="bilinear", align_corners=False)
        # Sigmoid activation ensures the output is within [0, 1] (required for SSIM, DISTS, etc.)
        assert x.shape == out.shape
        out = torch.sigmoid(out)
        return out

    def mixed_loss(self, denoised, clean):
        """
        Mixed loss: alpha * L1Loss + beta * SSIM_loss + gamma * LPIPS_loss + delta * DISTS_loss
        """
        l1 = self.l1_loss(denoised, clean)
        ssim_l = ssim_loss(denoised, clean)
        # LPIPS and DISTS losses require 3-channel input, so repeat the single channel
        denoised_3c = denoised.repeat(1, 3, 1, 1)
        clean_3c = clean.repeat(1, 3, 1, 1)
        # Ensure LPIPS and DISTS loss models are on the current device
        lpips_model = lpips_loss_fn.to(denoised.device)
        # dists_model = self.dists_loss_fn.to(denoised.device)

        lpips_l = lpips_model(denoised_3c, clean_3c).mean()
        # dists_l = dists_model(denoised_3c, clean_3c).mean()
        return self.alpha * l1 + self.beta * ssim_l + self.gamma * lpips_l

    def training_step(self, batch, batch_idx):
        # batch returns (images, error_vector, _)
        # images: (batch, 2, H, W), where channel 0 is clean (ground truth), and channel 1 is noisy
        images, error_vector, _ = batch
        # Extract noisy as input and clean as supervision
        target = images[:, 0:1, :, :]
        clean = images[:, 1:2, :, :]
        denoised = self(target)
        # Additional similarity loss (e.g., NCC loss), using mNCC_loss_parallel as an example
        # similarity = ncc_loss(clean, denoised)
        loss = self.mixed_loss(denoised, clean)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, error_vector, _ = batch
        target = images[:, 0:1, :, :]
        clean = images[:, 1:2, :, :]
        denoised = self(target)
        # similarity = ncc_loss(clean, denoised)
        loss = self.mixed_loss(denoised, clean)
        self.log("val_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        if batch_idx == 0:
            self.log_images(target, denoised, clean, self.current_epoch)
        return loss

    def log_images(self, input_image, pred_mask, true_mask, epoch):
        input_image = input_image.cpu().detach()
        pred_mask = pred_mask.cpu().detach()
        true_mask = true_mask.cpu().detach()

        batch_size = input_image.size(0)
        indices = torch.randperm(batch_size)[:4]

        grid_input = vutils.make_grid(
            input_image[indices], nrow=2, normalize=True)
        grid_pred = vutils.make_grid(
            pred_mask[indices], nrow=2, normalize=True)
        grid_true = vutils.make_grid(
            true_mask[indices], nrow=2, normalize=True)

        self.logger.experiment.add_image('Noisy image', grid_input, epoch)
        self.logger.experiment.add_image('Denoised image', grid_pred, epoch)
        self.logger.experiment.add_image('Clean image', grid_true, epoch)
        self.log_image_step += 1

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
