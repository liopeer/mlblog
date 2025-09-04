import os
import contextlib
from argparse import ArgumentParser
from typing import Iterator
import time

import torch
from torch import Tensor, Generator
from torch.nn import Module, BatchNorm1d, BatchNorm2d, BatchNorm3d, Parameter
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch.multiprocessing as mp
from torch.optim import SGD
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Grayscale, Compose
from torchvision import models


CUDA_AVAILABLE = torch.cuda.is_available()
MASTER_ADDR = "localhost"
MASTER_PORT = "12355"
DIST_BACKEND = "nccl" if CUDA_AVAILABLE else "gloo"


@contextlib.contextmanager
def setup_dist(rank: int, world_size: int):
    try:
        os.environ["MASTER_ADDR"] = MASTER_ADDR
        os.environ["MASTER_PORT"] = MASTER_PORT
        dist.init_process_group(DIST_BACKEND, rank=rank, world_size=world_size)
        yield
    finally:
        dist.destroy_process_group()


class CustomDDP(Module):
    def __init__(self, module: Module):
        super().__init__()
        self.module = module
        for param in self.module.parameters():
            param.register_post_accumulate_grad_hook(grad_avg_hook)
            dist.broadcast(param.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)


def grad_avg_hook(param: Tensor) -> None:
    if CUDA_AVAILABLE:
        dist.all_reduce(param.grad, op=ReduceOp.AVG)
    else:
        dist.all_reduce(param.grad, op=ReduceOp.SUM)
        param.grad /= dist.get_world_size()


class CustomDistSampler:
    def __init__(self, dataset: Dataset, shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_samples = len(self.dataset) // dist.get_world_size()
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if not self.shuffle:
            indices = range(self.num_samples)[rank::world_size]
            yield from iter(indices)
        else:
            gen = Generator()
            gen.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=gen)[rank::world_size]
            yield from iter(indices.tolist())

    def __len__(self):
        return self.num_samples
    

class CustomSyncBatchNorm(Module):
    def __init__(self, module: BatchNorm1d | BatchNorm2d | BatchNorm3d):
        super().__init__()
        self.module = module

    def forward(self, x: Tensor) -> Tensor:
        if self.module.training:
            reduce_dims = [0] + list(range(2, x.dim()))
            local_sum = x.sum(dim=reduce_dims)
            local_sq_sum = (x * x).sum(dim=reduce_dims)

            dist.all_reduce(local_sum, op=ReduceOp.SUM)
            dist.all_reduce(local_sq_sum, op=ReduceOp.SUM)

            count = x.numel() / x.size(1)

            mean = local_sum / count
            var = local_sq_sum / count - mean * mean
            
            self.module.running_var = (1 - self.module.momentum) * \
                self.module.running_var + self.module.momentum * var

            self.module.running_mean.mul_(1 - self.module.momentum).add_(mean * self.module.momentum)
            self.module.running_var.mul_(1 - self.module.momentum).add_(var * self.module.momentum)
            self.module.num_batches_tracked += 1
        else:
            mean = self.module.running_mean
            var = self.module.running_var

        x = (x - mean[None, :, *(None,) * (x.dim() - 2)]) / torch.sqrt(var[None, :, *(None,) * (x.dim() - 2)] + self.module.eps)
        if self.module.affine:
            x = x * self.module.weight[None, :, *(None,) * (x.dim() - 2)] + self.module.bias[None, :, *(None,) * (x.dim() - 2)]
        return x

    @classmethod
    def convert_sync_batchnorm(cls, module: Module) -> Module:
        if isinstance(module, (BatchNorm1d, BatchNorm2d, BatchNorm3d)):
            return cls(module)
        for name, child in module.named_children():
            module.add_module(name, cls.convert_sync_batchnorm(child))
        return module


def train_dist(
    rank: int, world_size: int, num_epochs: int, batch_size: int, num_workers: int
) -> None:
    # Setup the process group.
    with setup_dist(rank, world_size):
        # Model initialization and training loop.
        model = models.resnet18(num_classes=10)
        if CUDA_AVAILABLE:
            model = model.to(rank)
        model = CustomDDP(model)
        learning_rate = 0.001
        optimizer = SGD(model.parameters(), lr=learning_rate)

        # Load FashionMNIST dataset.
        train_set = FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=Compose([Grayscale(num_output_channels=3), ToTensor()]),
        )
        train_sampler = CustomDistSampler(train_set, shuffle=True)
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size // world_size,
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
        val_sampler = CustomDistSampler(val_set, shuffle=False)
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size // world_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

        for epoch in range(num_epochs):
            epoch_train_losses = []
            for inputs, targets in train_loader:
                if CUDA_AVAILABLE:
                    inputs, targets = inputs.to(rank), targets.to(rank)

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
                    inputs, targets = inputs.to(rank), targets.to(rank)

                    # Forward pass.
                    outputs = model(inputs)
                    loss = F.cross_entropy(outputs, targets)
                    epoch_val_losses += [loss.item()]

            epoch_train_loss = torch.tensor(
                sum(epoch_train_losses) / len(epoch_train_losses), device=rank
            )
            epoch_val_loss = torch.tensor(
                sum(epoch_val_losses) / len(epoch_val_losses), device=rank
            )
            dist.all_reduce(epoch_train_loss, op=ReduceOp.AVG)
            dist.all_reduce(epoch_val_loss, op=ReduceOp.AVG)

            # Print loss for the current iteration.
            if rank == 0:
                print(
                    f"Iteration {epoch + 1}/{num_epochs}\t"
                    f"Train Loss: {epoch_train_loss:.4f}\t"
                    f"Val Loss: {epoch_val_loss:.4f}"
                )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    torch.manual_seed(0)

    start_time = time.time()
    mp.spawn(
        train_dist,
        args=(args.world_size, args.num_epochs, args.batch_size, args.num_workers),
        nprocs=args.world_size,
    )
    end_time = time.time()

    print("Completed training in {:.2f} seconds.".format(end_time - start_time))
