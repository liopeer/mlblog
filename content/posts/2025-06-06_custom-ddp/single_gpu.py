# custom_ddp.py
from argparse import ArgumentParser
from typing import Iterator
import time

import torch
from torch import Generator
from torch.optim import SGD
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Grayscale, Compose
from torchvision import models


class CustomSampler:
    def __init__(self, dataset: Dataset, shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        if not self.shuffle:
            yield from iter(range(len(self.dataset)))
        else:
            gen = Generator()
            gen.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=gen)
            yield from iter(indices.tolist())

    def __len__(self):
        return len(self.dataset)


def train(num_epochs: int, batch_size: int, num_workers: int) -> None:
    # Setup the process group.
    # Model initialization and training loop.
    model = models.resnet18(num_classes=10).to("cuda")
    learning_rate = 0.001
    optimizer = SGD(model.parameters(), lr=learning_rate)

    # Load FashionMNIST dataset.
    train_set = FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=Compose([Grayscale(num_output_channels=3), ToTensor()]),
    )
    train_sampler = CustomSampler(train_set, shuffle=True)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_set = FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=Compose([Grayscale(num_output_channels=3), ToTensor()]),
    )
    val_sampler = CustomSampler(val_set, shuffle=False)
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    for epoch in range(num_epochs):
        epoch_train_losses = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to("cuda"), targets.to("cuda")

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
                inputs, targets = inputs.to("cuda"), targets.to("cuda")

                # Forward pass.
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                epoch_val_losses += [loss.item()]

        epoch_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        epoch_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)

        # Print loss for the current iteration.
        print(
            f"Iteration {epoch + 1}/{num_epochs}\t"
            f"Train Loss: {epoch_train_loss:.4f}\t"
            f"Val Loss: {epoch_val_loss:.4f}"
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    torch.manual_seed(0)

    start_time = time.time()
    train(args.num_epochs, args.batch_size, args.num_workers)
    end_time = time.time()

    print("Completed training in {:.2f} seconds.".format(end_time - start_time))
