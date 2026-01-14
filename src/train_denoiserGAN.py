import numpy as np
import argparse
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from air_network.datasets.siplaDataset import SiplaDataset18
from air_network.models.denoiserGAN import DenoiserGAN

parser = argparse.ArgumentParser()
parser.add_argument('--anatomy_type', type=str, default='bone')
parser.add_argument('--resolution', type=int, default=256)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--mask_only', type=str, default='both')
args = parser.parse_args()

anatomy_type = args.anatomy_type
resolution = args.resolution
epochs = args.epoch
mask_only = args.mask_only

mask_femur = mask_only != 'tibia'
mask_tibia = mask_only != 'femur'

def main():
    transform = transforms.Compose([transforms.Resize((resolution, resolution))])

    seed = 923
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_name_femur = 'SUBN_02_Femur_RE_Volume'
    model_name_tibia = 'SUBN_02_Tibia_RE_Volume'
    expectedSizePxl = 1000 if anatomy_type == 'bone' else 512

    dataset = SiplaDataset18(
        outSizePxl=resolution,
        femur_name=model_name_femur,
        tibia_name=model_name_tibia,
        expectedSizePxl=expectedSizePxl,
        anatomy_type=anatomy_type,
        add_noise=False, with_synth=False,
        transform=transform, data_length=None,
        mask_femur=mask_femur, mask_tibia=mask_tibia, mask_binary=False
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # num worker set to 1 to avoid potential issues on Windows
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1)

    model = DenoiserGAN(
        in_channels=1, out_channels=1,
        lr_g=1e-4, lr_d=1e-4,
        adv_weight=1e-3,  
        alpha=0.5, beta=0.3, gamma=0.2
    )

    logger = TensorBoardLogger("tb_logs_Aug", name="denoiserGAN_SUBN_02")

    checkpoint_callback = ModelCheckpoint(
        monitor='val/perc_loss',
        dirpath='checkpoints_denoiserGAN_SUBN_02',
        filename='denoiserGAN_SUBN_02-{epoch:02d}-{val_perc:.4f}',
        save_top_k=1, mode='min', auto_insert_metric_name=False
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, lr_monitor],
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()
