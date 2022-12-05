import torch
from data import DiffSet, ImageDataset
import pytorch_lightning as pl
from model import DiffusionModel
from torch.utils.data import DataLoader
import imageio
import glob
import torchvision.transforms as transforms


# Training hyperparameters
diffusion_steps = 100
dataset_choice = "Fashion"
max_epoch = 5
batch_size = 1

# Loading parameters
load_model = False
load_version_num = 1

dataroot = "./datasets/real"
size = 256


# Code for optionally loading model
pass_version = None
last_checkpoint = None

if load_model:
    pass_version = load_version_num
    last_checkpoint = glob.glob(
        f"./lightning_logs/{dataset_choice}/version_{load_version_num}/checkpoints/*.ckpt"
    )[-1]

# Create datasets and data loaders
# train_dataset = DiffSet(True, dataset_choice)
# val_dataset = DiffSet(False, dataset_choice)
transforms_ = [
    transforms.RandomCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
train_dataset = ImageDataset(dataroot, transforms_=transforms_)
val_dataset = ImageDataset(dataroot, transforms_=transforms_)


train_loader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                        num_workers=4, shuffle=True)

# Create model and trainer
if load_model:
    model = DiffusionModel.load_from_checkpoint(
        last_checkpoint, in_size=train_dataset.size*train_dataset.size, t_range=diffusion_steps, img_depth=train_dataset.depth)
else:
    print(train_dataset.size)
    model = DiffusionModel(
        train_dataset.size, diffusion_steps, train_dataset.depth)

# Load Trainer model
tb_logger = pl.loggers.TensorBoardLogger(
    "lightning_logs/",
    name=dataset_choice,
    version=pass_version,
)

trainer = pl.Trainer(
    max_epochs=max_epoch,
    log_every_n_steps=10,
    gpus=1,
    auto_select_gpus=True,
    resume_from_checkpoint=last_checkpoint,
    logger=tb_logger
)


# Train model
trainer.fit(model, train_loader, val_loader)
