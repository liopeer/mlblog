import os
from argparse import ArgumentParser

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Linear, Sequential
from torch import autograd
from torch.distributed import ReduceOp

MASTER_ADDR = "localhost"
MASTER_PORT = "12355"


def setup_dist(rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def train_dist(rank: int, world_size: int, num_iter: int) -> None:
    # Setup the process group.
    setup_dist(rank, world_size)

    # Model initialization and training loop.
    model = Sequential(Linear(10, 10), Linear(10, 10))
    learning_rate = 0.001

    for _ in range(num_iter):
        loss = model(torch.randn(10)).sum()
        grads = autograd.grad(loss, model.parameters())
        for param, grad in zip(model.parameters(), grads):
            dist.all_reduce(grad, op=ReduceOp.SUM)
            param.data -= learning_rate * grad / world_size

    dist.destroy_process_group()

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