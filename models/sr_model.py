########################
# sr_model.py
########################

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim

import piq          # For SSIM loss
import lpips        # For LPIPS
from torchvision.utils import make_grid

###################################################
# Define SSIM Loss: (1 - SSIM)
###################################################


def ssim_loss(pred, target):
    """
    Wrapper for piq.ssim, using 1 - SSIM as the loss.
    pred, target: [0,1] range
    """
    return 1.0 - piq.ssim(pred, target, data_range=1.0)

###################################################
# SRResNet: Supports 2× Upsampling
# (From 500×500 -> 1000×1000)
###################################################


class ResidualBlock(nn.Module):
    """
    Basic residual block in SRResNet:
      Conv -> ReLU -> Conv + skip
    """

    def __init__(self, n_feats=64):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + identity


class SRResNet(nn.Module):
    """
    A simplified version of SRResNet:
      - Initial convolution
      - Stacked ResidualBlocks
      - PixelShuffle upsampling (default ×2, can be changed to ×4, etc.)
      - Final convolution output

    Default in_channels=1, out_channels=1, n_feats=64, n_blocks=8, upscale=2
    From (500×500) -> (1000×1000).
    """

    def __init__(self, in_channels=1, out_channels=1,
                 n_feats=64, n_blocks=8, upscale=2):
        super().__init__()
        self.upscale = upscale

        # 1) Initial convolution
        self.conv_in = nn.Conv2d(in_channels, n_feats,
                                 kernel_size=9, padding=4)

        # 2) Stacked residual blocks
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResidualBlock(n_feats=n_feats))
        self.res_blocks = nn.Sequential(*blocks)

        # 3) Intermediate convolution
        self.conv_mid = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)

        # 4) Upsampling (PixelShuffle)
        up_layers = []
        if upscale not in [1, 2, 4]:
            raise ValueError("Only scale=1,2,4 are supported.")
        n_shuffle = {1: 0, 2: 1, 4: 2}[upscale]
        for _ in range(n_shuffle):
            up_layers.append(nn.Conv2d(n_feats, n_feats *
                             4, kernel_size=3, padding=1))
            up_layers.append(nn.PixelShuffle(2))
            up_layers.append(nn.ReLU(inplace=True))

        self.upsample = nn.Sequential(*up_layers)

        # 5) Final convolution output
        self.conv_out = nn.Conv2d(
            n_feats, out_channels, kernel_size=9, padding=4)

    def forward(self, x):
        # Initial convolution
        x = F.relu(self.conv_in(x))

        # Residual blocks
        identity = x
        out = self.res_blocks(x)
        out = self.conv_mid(out)
        out = out + identity  # global skip

        # Upsampling
        out = self.upsample(out)

        # Output (no activation)
        out = self.conv_out(out)
        return out


###################################################
# SRLightningModel:
# Uses L1 + SSIM + LPIPS Loss
###################################################
# Global LPIPS (needs to be moved to the corresponding device)
lpips_loss_fn = lpips.LPIPS(net='alex')


class SRLightningModel(pl.LightningModule):
    """
    Uses SRResNet for image reconstruction from low_res (500×500) -> high_res (1000×1000).
    Supports multiple losses: L1 + SSIM + LPIPS

    During training, batch = (low_res, high_res):
      low_res.shape = (B, 1, 500, 500)
      high_res.shape = (B, 1, 1000, 1000)
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 lr=1e-4,
                 alpha=0.5,    # L1 weight
                 beta=0.3,    # SSIM weight
                 gamma=0.2,   # LPIPS weight
                 n_feats=64,
                 n_blocks=8,
                 upscale=2):
        """
        Parameters:
          in_channels, out_channels: Input/output channels (1=grayscale, 3=RGB)
          lr: Learning rate
          alpha, beta, gamma: Weights for L1, SSIM, LPIPS losses
          n_feats, n_blocks, upscale: SRResNet network structure
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # Loss weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Create SRResNet model
        self.model = SRResNet(
            in_channels=in_channels,
            out_channels=out_channels,
            n_feats=n_feats,
            n_blocks=n_blocks,
            upscale=upscale
        )

        self.l1_loss = nn.L1Loss()

    def forward(self, x):
        """
        x: (B,1,500,500)
        Output: (B,1,1000,1000)
        """
        out = self.model(x)
        # Clamp to [0,1], depending on the data
        out = torch.sigmoid(out)
        return out

    def _mixed_loss(self, preds, targets):
        """
        Compute mixed loss:
          total_loss = alpha*L1 + beta*SSIM + gamma*LPIPS
        """
        # 1) L1
        loss_l1 = self.l1_loss(preds, targets)
        # 2) SSIM
        loss_ssim = ssim_loss(preds, targets)
        # 3) LPIPS requires 3 channels
        preds_3c = preds.repeat(1, 3, 1, 1)
        targets_3c = targets.repeat(1, 3, 1, 1)

        # Ensure lpips_loss_fn is on the same device
        lpips_loss_fn.to(preds.device)

        lpips_val = lpips_loss_fn(preds_3c, targets_3c).mean()

        # Combine
        total = (self.alpha * loss_l1 +
                 self.beta * loss_ssim +
                 self.gamma * lpips_val)
        return total, loss_l1, loss_ssim, lpips_val

    def training_step(self, batch, batch_idx):
        """
        batch = (low_res, high_res)
        low_res: (B,1,500,500)
        high_res: (B,1,1000,1000)
        """
        low_res, high_res = batch
        preds = self(low_res)
        total, l1v, ssimv, lpipsv = self._mixed_loss(preds, high_res)

        self.log("train/total_loss", total, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log("train/l1_loss", l1v, on_step=True, on_epoch=True)
        self.log("train/ssim_loss", ssimv, on_step=True, on_epoch=True)
        self.log("train/lpips", lpipsv, on_step=True, on_epoch=True)
        return total

    def validation_step(self, batch, batch_idx):
        low_res, high_res = batch
        preds = self(low_res)
        total, l1v, ssimv, lpipsv = self._mixed_loss(preds, high_res)

        self.log("val/total_loss", total, prog_bar=True)
        self.log("val/l1_loss", l1v)
        self.log("val/ssim_loss", ssimv)
        self.log("val/lpips", lpipsv)

        # Optional: Log visualizations
        if batch_idx == 0:
            self._log_images(low_res, preds, high_res)
        return total

    def _log_images(self, lr_img, sr_img, hr_img):
        """
        Visualize in TensorBoard
        """
        lr_img = lr_img.detach().cpu()
        sr_img = sr_img.detach().cpu()
        hr_img = hr_img.detach().cpu()

        idx = min(lr_img.shape[0], 4)
        grid_lr = make_grid(lr_img[:idx], nrow=2, normalize=True)
        grid_sr = make_grid(sr_img[:idx], nrow=2, normalize=True)
        grid_hr = make_grid(hr_img[:idx], nrow=2, normalize=True)

        self.logger.experiment.add_image(
            "low_res", grid_lr, self.current_epoch)
        self.logger.experiment.add_image(
            "sr_result", grid_sr, self.current_epoch)
        self.logger.experiment.add_image(
            "high_res", grid_hr, self.current_epoch)

    def configure_optimizers(self):
        # Can use Adam or other optimizers
        return optim.Adam(self.parameters(), lr=self.lr)


########################
# If you want to test this file
########################
if __name__ == "__main__":
    # Simulate a batch
    dummy_low_res = torch.randn(2, 1, 500, 500)
    dummy_high_res = torch.randn(2, 1, 1000, 1000)

    model = SRLightningModel()
    sr_out = model(dummy_low_res)
    print("SR output size:", sr_out.shape)  # Expected (2,1,1000,1000)
