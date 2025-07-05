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

        optimizer = Adam(model.parameters(), lr=1e-3)

        y = torch.randn(256, param_size).to(rank)
        x = torch.randn(256, param_size).to(rank)

        for i in range(10):
            optimizer.zero_grad()
            y_hat = model(x)
            loss = F.mse_loss(y_hat, y)
            loss.backward()
            optimizer.step()

            print(f"Epoch {i+1}, Loss: {loss.item():.4f}")
        
        print(f"Max memory allocated on rank {rank}: {torch.cuda.max_memory_allocated(device=rank) / (1024 ** 2):.2f} MB")

if __name__ == "__main__":
    from argparse import ArgumentParser
    import time

    parser = ArgumentParser()
    parser.add_argument("--num-devices", type=int, default=2, help="Number of devices to use")
    args = parser.parse_args()

    print("Running DDP ...")
    time_start = time.time()
    mp.spawn(main, args=(args.num_devices, False), nprocs=args.num_devices)
    print(f"DDP execution time: {time.time() - time_start:.2f} seconds")

    print("Running FSDP ...")
    time_start = time.time()
    mp.spawn(main, args=(args.num_devices, True), nprocs=args.num_devices)
    print(f"FSDP execution time: {time.time() - time_start:.2f} seconds")