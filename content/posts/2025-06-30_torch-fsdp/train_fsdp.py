import os
import contextlib
import logging

import torch
from torch.optim import Adam
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.nn import Module

from linear_model import LinearModel

MASTER_ADDR = "localhost"
MASTER_PORT = "12355"

@contextlib.contextmanager
def setup_dist(rank: int, world_size: int):
    try:
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        os.environ['MASTER_PORT'] = MASTER_PORT
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo", 
            rank=rank, 
            world_size=world_size
        )
        print(f"Rank {rank} joined the process group.")
        yield
    finally:
        dist.destroy_process_group()

def main(rank: int, world_size: int):
    param_size = 8192 * 2
    with setup_dist(rank, world_size):
        torch.cuda.reset_peak_memory_stats()
        model = FSDP(
            LinearModel(input_size=param_size, output_size=param_size).to(rank)
        )
        optimizer = Adam(model.parameters(), lr=1e-1)

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
    torch.cuda.reset_peak_memory_stats()
    world_size = 2
    mp.spawn(main, args=(world_size,), nprocs=world_size)
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")