import os
import contextlib

import torch
from torch.optim import Adam
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import fully_shard
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

def main(rank: int, world_size: int, fsdp2: bool):
    param_size = 8192
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
        if not fsdp2:
            model = FSDP(
                model,
                auto_wrap_policy=ModuleWrapPolicy({Linear}),
                device_id=device,
            )
        else:
            for _, module in model.named_modules():
                if isinstance(module, Linear):
                    fully_shard(module)
            fully_shard(model)

        optimizer = Adam(model.parameters(), lr=1e-3)

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