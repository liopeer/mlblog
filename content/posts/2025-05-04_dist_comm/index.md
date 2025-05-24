+++
title = 'Distributed Training in PyTorch – I'
date = 2025-05-04T15:09:46+02:00
draft = false
+++
This blog post is the first in a series in which I would like to showcase several methods on how you can train your machine learning model with PyTorch on several GPUs/CPUs across multiple nodes in a cluster. In this first part we will try to look a bit under the hood of how you can launch distributed jobs and how you can ensure proper communication between them. In follow-ups we will create our own PyTorch `DistributedDataParallel` (DDP) class and we will also look at popular frameworks such as PyTorch Lightning and resource schedulers like SLURM that can help you getting your distributed training running. We will strictly focus on data parallelism, meaning a parallelism where the whole model fits into the memory of a single GPU and we exchange gradients (and potentially batch norms) across the GPUs, while keeping the whole optimization local on each GPU.

{{< notice note >}}
You don't need access to a compute cluster or multiple GPUs to follow along. We can simulate everything using our CPU and a single computer.
{{< /notice >}}

## Terminology of Distributed Training
PyTorch offers communication between distributed processes through [`torch.distributed`](https://pytorch.org/docs/stable/distributed.html), which is a wrapper around other communication libraries, such as [Gloo](https://github.com/pytorch/gloo), [NCCL](https://github.com/NVIDIA/nccl) or [MPI](https://github.com/open-mpi/ompi). PyTorch recommends using Gloo when doing distributed training on CPUs and NCCL when using Nvidia GPUs, which is why we will be relying on Gloo over the course of this blog post.

Let's harmonize some terms here, before we continue on how one can set up distributed data parallelism:
1. *Number of Nodes*: A node is the same as a *computer* or a *machine*, it therefore refers to how many computers we use in parallel for our training.
2. *World Size*: This is the number of processes that we run in parallel. This is independent of how many nodes we run training on, i.e. if we have 2 nodes and world size 8 we are most likely running 4 processes on each node (it could also be 3+5 or ny other combination, but let's assume we run equal numbers of processes on each node).
3. *Rank* or *Global Rank*: A specific process, which is identified by an integer between 0 and the world size.
4. *Node Rank*: An integer between 0 and the total number of nodes, which uniquely identifies a node.
5. *Local Rank*: The process number on a specific node, i.e. the modulus of the global rank with respect to the number of processes per node (local_rank = global_rank % processes_per_node, again assuming that all nodes have equal numbers of processes).

## Let Processes "find" each other
Before processes can communicate, we first need to make sure they know of each other's existence and how they can address each other. The first contact for this is done through the [`torch.distributed.init_process_group` function](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group). By default, this function assumes that the environment variables `MASTER_ADDR` (the IP of the master node) and `MASTER_PORT` (a free port on the master node) are set. That port on the master node, will then be used by all processes to join the process group. Besides setting environment variables, one could also use a TCP key/value store or a shared file to communicate these two parameters, however for this I refer the interested reader to the [PyTorch documentation](https://pytorch.org/docs/stable/distributed.html#initialization).

The `init_process_group` will wait until all processes have joined (or until the `timeout` parameter has been reached), therefore it needs to know how many processes in total are expected and each process should communicate its rank (so that potential duplicates can raise errors).

We can integrate all of this and create a super minimal script and launch it in (in 2 separate terminals) with the commands:
```python
# procgroup_join.py
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
```
```bash
# first terminal
MASTER_ADDR="localhost" MASTER_PORT="12355" python procgroup_join.py --rank=0 --world-size=2
```

```bash
# second terminal
MASTER_ADDR="localhost" MASTER_PORT="12355" python procgroup_join.py --rank=1 --world-size=2
```
As you'll see, nothing will happen in the first terminal until you launch the command in the second terminal, since the rank 0 process has to wait at `dist.init_process_group` until all the expected processes have joined. Manually launching the processes from separate terminals is of course a bit cumbersome, that's why we will automate this in the next section.

## Automated Process Spawning
We can use the Python built-in [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html) or torch's own [`torch.multiprocessing`](https://pytorch.org/docs/stable/multiprocessing.html#spawning-subprocesses) modules to launch subprocesses from our script instead of having to use multiple terminal windows. `torch.multiprocessing` is simply a wrapper around `multiprocessing` with some PyTorch specific optimization and some additional functions like the `spawn` function that we will use here:

```python
# dist_mpspawn.py
import os
from argparse import ArgumentParser

import torch.distributed as dist
import torch.multiprocessing as mp

MASTER_ADDR = "localhost"
MASTER_PORT = "12355"

def setup_dist(rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Process {rank} has joined the process group!")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()

    mp.spawn(setup_dist, args=(args.world_size, ), nprocs=args.world_size)

    dist.destroy_process_group()
```

You can launch this script by giving it a number of sub-processes to create through the `--world-size` command line argument. For example to launch with 8 processes:

```bash
python dist_mpspawn.py --world-size=8
```
As you can see, we also don't need to pass the environment variables before the command since we set them already in the script.

Much more convenient than opening 8 terminals, right? Unfortunately, `mp.spawn` does of course not scale to muliple nodes, so with the current script, you'd still have to manually launch the script on each node. Workload schedulers like [SLURM](https://slurm.schedmd.com/documentation.html) can help with this, but we will cover that in a later part of this series. For now, let's just assume that we run the script on a single node with multiple GPUs.

{{< notice info >}}
The `mp.spawn` function takes a function as the first argument (here `setup_dist`), which needs to have the rank as the first argument and all other arguments of that function (here `world_size`) can be passed through the `args` parameter as a tuple.
{{< /notice >}}

## Averaging Gradients Across Processes
Now that we know how to initialize a process group, let's actually train a model on several processes and average the gradients over the processes. The averaging operation can be done using `torch.distributed.all_reduce`, which takes a tensor as input, averages each element of the tensor across the processes and returns the averaged tensor on every process.

```python
# dist_train_manual.py
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
```

In above script we manually implemented the gradient descent step, which makes the code very complicated and very different from standard PyTorch code. This code would also not be easily compatible with other PyTorch optimizers, such as SGD with momentum or Adam. This can be avoided, by hooking the `all_reduce` operation into the model's backward pass through `Module`'s `register_full_backward_hook` method.

```python
# dist_train_hooks.py
import os
from typing import Any
from argparse import ArgumentParser
import functools

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Linear, Sequential, Module
from torch import autograd
from torch.distributed import ReduceOp
from torch.optim import SGD

MASTER_ADDR = "localhost"
MASTER_PORT = "12355"


def setup_dist(rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def grad_avg_hook(
    world_size: int, module: Module, grad_input: Any, grad_output: Any
) -> None:
    # This hook will average the gradients across all processes
    for param in module.parameters():
        dist.all_reduce(param.grad.data, op=ReduceOp.SUM)
        param.grad.data /= world_size

def train_dist(rank: int, world_size: int, num_iter: int) -> None:
    # Setup the process group.
    setup_dist(rank, world_size)

    # Model initialization and training loop.
    model = Sequential(Linear(10, 10), Linear(10, 10))
    hook = functools.partial(grad_avg_hook, world_size)
    model.register_full_backward_hook(hook)
    learning_rate = 0.001
    optimizer = SGD(model.parameters(), lr=learning_rate)

    for _ in range(num_iter):
        loss = model(torch.randn(10)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
```

## Wrap Up
In this first part of the series we have seen how we can set up a distributed process group in PyTorch and how we can average gradients across processes. We have also seen how to automate the spawning of processes using `torch.multiprocessing.spawn`. In the next part we will extend the idea of averaging gradients to batch normalization layers, how to distribute dataset sampling across processes and how to create our own `DistributedDataParallel` class.