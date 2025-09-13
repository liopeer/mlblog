import os
from argparse import ArgumentParser

import torch.distributed as dist
import torch.multiprocessing as mp

MASTER_ADDR = "localhost"
MASTER_PORT = "12355"


def setup_dist(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Process {rank} has joined the process group!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()

    mp.spawn(setup_dist, args=(args.world_size,), nprocs=args.world_size)

    dist.destroy_process_group()
