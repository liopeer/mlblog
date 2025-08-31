+++
title = 'FSDP vs. DDP'
date = 2025-07-01T20:48:33+02:00
draft = true
author = 'Lionel Peer'
+++

## PyTorch Fully-Sharded Data Parallelism (FSDP)
Whenever your model, together with its gradients and optimizer states, does not fit into the memory of a single GPU or you would have to use prohibitively small batch sizes, we need to split. Over the years, several methods of splitting the model have been developed, however, the most common and most efficient approach is Fully-Sharded Data Parallelism (FSDP). FSDP splits all parameters (usually along the first dimension) and distributes them across the available GPUs. Since moving data between GPUs is much less expensive than moving data between the CPU and GPU, this approach is relatively efficient.

### Comparing DDP and FSDP
```python
import os
import contextlib

import torch
from torch.optim import Adam
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import Linear
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from mlp import MLPModel

MASTER_ADDR = "localhost"
MASTER_PORT = "12355"

@contextlib.contextmanager
def setup_dist(rank: int, world_size: int):
    try:
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        os.environ['MASTER_PORT'] = MASTER_PORT
        dist.init_process_group(
            backend="nccl", 
            rank=rank, 
            world_size=world_size
        )
        print(f"Rank {rank} joined the process group.")
        yield
    finally:
        dist.destroy_process_group()

def main(rank: int, world_size: int, fsdp: bool):
    param_size = 16384
    num_layers = 5
    with setup_dist(rank, world_size):
        device = torch.device(f"cuda:{rank}")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.set_device(rank)

        model = MLPModel(
            input_size=param_size, 
            output_size=param_size, 
            hidden_size=param_size, 
            num_layers=num_layers,
        )
        if fsdp:
            model = FSDP(
                model,
                auto_wrap_policy=ModuleWrapPolicy({Linear}),
                device_id=device,
            )
        else:
            model = DDP(
                model.to(device),
            )

        optimizer = Adam(model.parameters(), lr=1e-5)

        y = torch.randn(256, param_size).to(rank)
        x = torch.randn(256, param_size).to(rank)

        for i in range(5):
            optimizer.zero_grad()
            y_hat = model(x)
            loss = F.mse_loss(y_hat, y)
            loss.backward()
            optimizer.step()

            print(f"Epoch {i+1}, Loss: {loss.item():.4f}")
        
        print(f"Max memory allocated on rank {rank}: {torch.cuda.max_memory_allocated(device=rank) / (1024 ** 2):.2f} MB")

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--fsdp', action='store_true', help='Use FSDP instead of DDP')
    args = parser.parse_args()

    world_size = 4
    mp.spawn(main, args=(world_size, args.fsdp), nprocs=world_size)
```