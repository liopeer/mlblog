import os
from argparse import ArgumentParser
import functools
import contextlib

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Linear, Sequential
from torch import Tensor
from torch.distributed import ReduceOp
from torch.optim import SGD

MASTER_ADDR = "localhost"
MASTER_PORT = "12355"

@contextlib.contextmanager
def setup_dist(rank: int, world_size: int):
    try:
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        os.environ['MASTER_PORT'] = MASTER_PORT
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        yield
    finally:
        dist.destroy_process_group()

def grad_avg_hook(world_size: int, param: Tensor) -> None:
    dist.all_reduce(param.grad, op=ReduceOp.SUM)
    param.grad /= world_size

def train_dist(rank: int, world_size: int, num_iter: int) -> None:
    # Setup the process group.
    with setup_dist(rank, world_size):
        # Model initialization and training loop.
        model = Sequential(Linear(10, 10), Linear(10, 10))
        learning_rate = 0.001
        optimizer = SGD(model.parameters(), lr=learning_rate)
        hook = functools.partial(grad_avg_hook, world_size)
        # Ensure that all processes start with the same model weights.
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

        for param in model.parameters():
            param.register_post_accumulate_grad_hook(hook)

        for i in range(num_iter):
            if rank == 0:
                print(f"Iteration {i+1}/{num_iter}")
            optimizer.zero_grad()
            loss = model(torch.rand(10)).sum()
            loss.backward()
            optimizer.step()

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