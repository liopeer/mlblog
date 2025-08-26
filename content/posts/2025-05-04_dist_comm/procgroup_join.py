import os
from argparse import ArgumentParser

import torch.distributed as dist

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int)
    args = parser.parse_args()

    dist.init_process_group(
        backend="gloo", 
        rank=args.rank, 
        world_size=args.world_size,
    )

    print(f"Rank {args.rank} has joined the process group!")

    dist.destroy_process_group()