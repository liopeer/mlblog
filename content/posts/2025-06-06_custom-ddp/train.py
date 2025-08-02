# train.py
from argparse import ArgumentParser

from torchvision.datasets import FashionMNIST
from torchvision import models
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch.multiprocessing as mp
import torch

import custom_ddp
from custom_ddp import CustomDDP

def get_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_dataset = FashionMNIST(
        root="./data", 
        train=True, 
        download=True, 
        transform=Compose([ToTensor(), lambda x: x.repeat(3, 1, 1)])
    )
    test_dataset = FashionMNIST(
        root="./data", 
        train=False, 
        download=True, 
        transform=Compose([ToTensor(), lambda x: x.repeat(3, 1, 1)])
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader

def train_dist(rank: int, world_size: int, num_iter: int = 10):
    with custom_ddp.setup_dist(rank, world_size):
        print(f"Rank {rank} is training...")
        model = CustomDDP(
            module=models.resnet18(num_classes=10),
            world_size=world_size
        )
        train_load, test_loader = get_dataloaders(batch_size=32)
        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=0.001)

        for epoch in range(num_iter):
            model.train()
            for batch_idx, (data, target) in enumerate(train_load):
                data, target = data, target
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

            model.eval()
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(rank), target.to(rank)
                    output = model(data)
                    test_loss = criterion(output, target)
                    print(f"Rank {rank}, Epoch {epoch}, Test Loss: {test_loss.item()}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--num-iter", type=int, default=10)
    args = parser.parse_args()

    mp.spawn(
        train_dist, 
        args=(args.world_size, args.num_iter), 
        nprocs=args.world_size
    )

    print("Finished training!")