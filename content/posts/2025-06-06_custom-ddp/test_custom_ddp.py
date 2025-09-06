from torch.optim import SGD
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Grayscale, Compose
from torchvision import models
import torch
from custom_ddp import train_dist
import torch.nn.functional as F
import random
import numpy as np
from argparse import ArgumentParser
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from typing import Iterator
from torch import Generator


def train(num_epochs: int, batch_size: int, num_workers: int) -> None:
    # Model initialization and training loop.
    model = models.resnet18(num_classes=10).to(0)
    learning_rate = 0.001
    optimizer = SGD(model.parameters(), lr=learning_rate)

    # Load FashionMNIST dataset.
    train_set = FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=Compose([Grayscale(num_output_channels=3), ToTensor()]),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )

    val_set = FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=Compose([Grayscale(num_output_channels=3), ToTensor()]),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    for epoch in range(num_epochs):
        epoch_train_losses = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(0), targets.to(0)

            # Forward pass.
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            # Backward pass and optimization.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_losses += [loss.item()]

        epoch_val_losses = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(0), targets.to(0)

                # Forward pass.
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                epoch_val_losses += [loss.item()]

        epoch_train_loss = torch.tensor(
            sum(epoch_train_losses) / len(epoch_train_losses), device=0
        )
        epoch_val_loss = torch.tensor(
            sum(epoch_val_losses) / len(epoch_val_losses), device=0
        )

        # Print loss for the current iteration.
        print(
            f"Iteration {epoch + 1}/{num_epochs}\t"
            f"Train Loss: {epoch_train_loss:.4f}\t"
            f"Val Loss: {epoch_val_loss:.4f}"
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    print("Running distributed training with 2 GPUs...")
    mp.spawn(
        train_dist,
        args=(2, args.num_epochs, args.batch_size, args.num_workers),
        nprocs=2,
    )

    print("\nRunning distributed training with 4 GPUs...")
    mp.spawn(
        train_dist,
        args=(4, args.num_epochs, args.batch_size, args.num_workers),
        nprocs=4,
    )

    print("\nRunning single-GPU training...")
    train(args.num_epochs, args.batch_size, args.num_workers)
