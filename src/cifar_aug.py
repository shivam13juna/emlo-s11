# Use all possible augmentation using albumentation for trainining cifar with TIMM resnet18 model

import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.models import resnet18
import pytorch_lightning as pl
# Use hydra for configuration

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from hydra.experimental import compose, initialize
import hydra

@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
        
    # # parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    # parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    # parser.add_argument(
    #     "--resume", "-r", action="store_true", help="resume from checkpoint"
    # )
    # parser.add_argument("--seed", default=0, type=int, help="random seed")
    # parser.add_argument("--batch-size", default=128, type=int, help="batch size")
    # parser.add_argument("--epochs", default=10, type=int, help="epochs")
    # config = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    cudnn.benchmark = True

    # Data
    print("==> Preparing data..")
    # Use all possible augmentation using albumentation

    class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
        def __init__(self, root="data/", train=True, download=True, transform=None):
            super().__init__(root=root, train=train, download=download, transform=transform)

        def __getitem__(self, index):
            image, label = self.data[index], self.targets[index]

            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]

            return image, label


    transform_train = A.Compose(
        [
            A.RandomCrop(32, 32),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
            ToTensorV2(),
        ]
    )


    transform_test = A.Compose(
        [
            A.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
            ToTensorV2(),
        ]
    )


    trainset = Cifar10SearchDataset(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=2
    )

    testset = Cifar10SearchDataset(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config.batch_size, shuffle=False, num_workers=2
    )

    # Build and Train model using pytorch lightning


    class LitModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = resnet18(pretrained=True, num_classes=10)
            self.criterion = nn.CrossEntropyLoss()

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean()
            return {"val_loss": loss, "val_acc": acc}

        def validation_epoch_end(self, outputs):
            avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
            avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
            return {"val_loss": avg_loss, "val_acc": avg_acc}

        def test_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean()
            return {"test_loss": loss, "test_acc": acc}

        def test_epoch_end(self, outputs):
            avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
            avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
            return {"test_loss": avg_loss, "test_acc": avg_acc}

        def configure_optimizers(self):
            optimizer = optim.SGD(
                self.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4
            )
            return optimizer

        def train_dataloader(self):
            return trainloader

        def val_dataloader(self):
            return testloader

        def test_dataloader(self):
            return testloader


    model = LitModel()
    trainer = pl.Trainer(gpus=1, max_epochs=config.epochs)
    trainer.fit(model)


if __name__ == "__main__":
    main()


# Test
