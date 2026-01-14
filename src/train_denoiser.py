import numpy as np
import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, random_split
from air_network.datasets.siplaDataset import SiplaDataset18
# from air_network.models.lossnet_simple_ncc import TwinResNeXtCostVolume
# from air_network.models.lossnet_simple_stn import TwinResNeXtCostVolumeSTN, TwinResNeXtFlowNetC_NoSTN
from air_network.models.denoiser import Denoiser
from torchvision import transforms
from air_network.tools.callbacks import ResolutionMonitor

angle_representation = '6d'

parser = argparse.ArgumentParser(description="Train ShadowNet Model")
parser.add_argument('--anatomy_type', type=str, default='bone', help='Number of epochs for training')
parser.add_argument('--resolution', type=int, default=256, help='Resolution of the rendered images')
parser.add_argument('--epoch', type=int, default=1000, help='Number of epochs for training')
parser.add_argument('--mask_only', type=str, default='both', help='only femur or only tibia, or both')
args = parser.parse_args()

anatomy_type=args.anatomy_type
resolution = args.resolution
epochs = args.epoch
mask_only=args.mask_only


mask_femur=True
mask_tibia=True
if mask_only=='femur':
    mask_tibia=False
if mask_only=='tibia':
    mask_femur=False


def main():
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),  
        # transforms.Normalize(mean=[0.5], std=[0.5])  
    ])
    # set random seed
    # set random seed
    seed = 923
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Create the dataset
    model_name_femur='SUBN_02_Femur_RE_Volume'
    model_name_tibia='SUBN_02_Tibia_RE_Volume'
    expectedSizePxl=512
    if anatomy_type=='bone':
        expectedSizePxl=1000
    dataset = SiplaDataset18(outSizePxl=resolution,femur_name=model_name_femur,tibia_name=model_name_tibia,
                             expectedSizePxl=expectedSizePxl, # 512 for implant,1000 for bone
                             anatomy_type=anatomy_type,
                             add_noise=False, with_synth=False, transform=transform,data_length=None,
                             mask_femur=mask_femur,
                             mask_tibia=mask_tibia, mask_binary=False)

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4,persistent_workers=True)
    # can turn shuffle to True to see random log image segmentation check
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4,persistent_workers=True)
    # Initialize model
    # Reduce growth rate and block layers
    # model_attention = ModelUDenseNetAttention(growth_rate=16, block_layers=(2, 3, 4, 5))
    # model_densenet = TwinResNeXtCostVolume(femur_path=femur_path,tibia_path=tibia_path,)
    # model_denoiser = PretrainedDenoiseDeepLabv3()  # for similarity metric)
    model_denoiser = Denoiser()


    # Set up logger
    logger = TensorBoardLogger("tb_logs_Sep", name=f"denoiser_SUBN_02")

    # Set up checkpoint callback
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_loss',
    #     dirpath=f'checkpoints_denoiser_SUBN_02',
    #     filename=f'denoiser_SUBN_02_{resolution}_{angle_representation}-' + '{epoch:02d}-{val_loss:.2f}',
    #     save_top_k=1,
    #     mode='min'
    # )

    # Set up learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # Initialize the ResolutionMonitor with the current resolution
    resolution_monitor = ResolutionMonitor(resolution=resolution)
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,  # Use 1 GPU
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=2,
        callbacks=[resolution_monitor, lr_monitor],
        enable_progress_bar=True,

    )
    # Set appropriate matmul precision for CUDA device with Tensor Cores
    # torch.set_float32_matmul_precision('high')
    # Train the model
    trainer.fit(model_denoiser, train_loader, val_loader)

if __name__ == '__main__':
    main()