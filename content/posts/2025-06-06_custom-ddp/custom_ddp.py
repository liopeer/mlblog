# custom_ddp.py
import functools
import os
import contextlib

import torch
from torch import Tensor
from torch.nn import Module
import torch.distributed as dist
from torch.distributed import ReduceOp

MASTER_ADDR = "localhost"
MASTER_PORT = "12355"
DIST_BACKEND = "gloo"

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
    def __init__(self, module, world_size: int):
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