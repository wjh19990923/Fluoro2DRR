import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger

# Import the SRLightningModel
from air_network.models.sr_model import SRLightningModel
from air_network.models.unet_model import PretrainedUNetModel
from air_network.datasets.SRdataset import SRDataset


# Your custom dataset, for example, SRDataset
# The following is just an example, modify it according to your actual situation
def main():
    # If you have a real Dataset, you can import it here
    # dataset = MyRealDataset(...)
    # Split dataset into train and validation sets
    resolution = 512
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        # transforms.ToTensor(),  # This step converts a PIL Image (mode "L") to shape (1, H, W)
    ])

    dataset = SRDataset(data_length=1000)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    # Initialize the model
    # You can pass in learning_rate, ssim_weight, lpips_weight, etc.
    model = SRLightningModel()

    # Set up logger
    logger = TensorBoardLogger("tb_logs", name=f"SR")

    # Trainer
    trainer = Trainer(
        max_epochs=10,
        accelerator='gpu',
        devices=1,  # Use 1 GPU
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
