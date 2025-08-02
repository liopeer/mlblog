+++
title = 'Distributed Training in PyTorch - II'
date = 2025-06-06T20:48:33+02:00
draft = true
author = 'Lionel Peer'
+++
In the [part I of this series]({{< relref "/posts/2025-05-04_dist_comm" >}}), we saw how we could
- launch different processes with `mp.spawn` or through multiple terminal windows
- group processes in process groups for communication and synchronization
- implement a distributed trraining loop by manually sychronizing gradients or by hooking them into the backward pass

In this part, we will now follow up on this and implement our own simplified version of `DistributedDataParallel` (DDP) and other helpers for distributed training. This will allow us to better structure our code and abstract away some of the boilerplate code that we had to use inside the training loop.

## Our own DDP
You can find the documentation of the original class here: [`DistributedDataParallel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html). We will implement a significantly simplified version of it, but the main ideas will be the same.

We start by creating the class and hooking the gradient synchronization into the backward pass. Additionally, we also copy over our context manager from the previous part that sets up and destroys the process group.

```python
# custom_ddp.py
import torch
from torch.nn import Module
import torch.distributed as dist

MASTER_ADDR = "localhost"
MASTER_PORT = "12355"
DIST_BACKEND = "nccl" if torch.cuda.is_available() else "gloo"

@contextlib.contextmanager
def setup_dist(rank: int, world_size: int):
    try:
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        os.environ['MASTER_PORT'] = MASTER_PORT
        dist.init_process_group(DIST_BACKEND, rank=rank, world_size=world_size)
        yield
    finally:
        dist.destroy_process_group()

class CustomDDP(Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        hook = functools.partial(grad_avg_hook, world_size)
        for param in self.module.parameters():
            param.register_post_accumulate_grad_hook(hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def state_dict(self, *args, **kwargs):
        """Override state_dict to ensure that we only return the state of the module."""
        return self.module.state_dict(*args, **kwargs)

def grad_avg_hook(world_size: int, param: Tensor) -> None:
    dist.all_reduce(param.grad, op=ReduceOp.SUM)
    param.grad /= world_size
```

In the previous post we used a dummy loop that did not do any training. Let's now try to train a proper neural network with our custom DDP. Specifically, we will train a ResNet18 on the FashionMNIST dataset.

```python
# train.py
from argparse import ArgumentParser

from torchvision.datasets import FashionMNIST
from torchvision import models
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch.multiprocessing as mp

def get_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_dataset = FashionMNIST(
        root="./data", 
        train=True, 
        download=True, 
        transform=ToTensor()
    )
    test_dataset = FashionMNIST(
        root="./data", 
        train=False, 
        download=True, 
        transform=ToTensor()
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
    with setup_dist(rank, world_size):
        model = CustomDDP(
            models.resnet18(num_classes=10).to(rank)
        )
        train_load, test_loader = get_dataloaders(batch_size=32)
        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=0.001)

        for epoch in range(num_iter):
            model.train()
            for batch_idx, (data, target) in enumerate(train_load):
                data, target = data.to(rank), target.to(rank)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

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
```