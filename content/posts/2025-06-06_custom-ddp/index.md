+++
title = 'Distributed Training in PyTorch - II'
date = 2025-06-06T20:48:33+02:00
draft = true
author = 'Lionel Peer'
+++
In the [part I of this series]({{< relref "/posts/2025-05-04_dist_comm" >}}), we saw how we could
- launch different processes with `mp.spawn` or through multiple terminal windows
- group processes in process groups for communication and synchronization
- implement a distributed training loop by manually synchronizing gradients or by hooking them into the backward pass

In this part, we will now follow up on this and implement our own simplified version of `DistributedDataParallel` (DDP) and other helpers for distributed training – particularly distributed sampling of data and synchronization of batch norms. This will allow us to better structure our code and abstract away some of the boilerplate code that we had to use inside the training loop. We will then verify the correctness of our implementation by training a ResNet18[^1] on the FashionMNIST[^2] dataset and comparing the results to the original `DistributedDataParallel` implementation.

## Our own DDP
You can find the documentation of the original class here: [`DistributedDataParallel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html). We will implement a significantly simplified version of it, but the main ideas will be the same.

We can start by copying over our context manager from the previous part that sets up and destroys the process group.

```python
import os
import contextlib

import torch.distributed as dist

MASTER_ADDR = "localhost"
MASTER_PORT = "12355"
DIST_BACKEND = "nccl"

@contextlib.contextmanager
def setup_dist(rank: int, world_size: int):
    try:
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        os.environ['MASTER_PORT'] = MASTER_PORT
        dist.init_process_group(DIST_BACKEND, rank=rank, world_size=world_size)
        yield
    finally:
        dist.destroy_process_group()
```

Since we plan on actually training a model this time, we will use the `"nccl"` backend, which is optimized for GPUs. If you don't have multiple GPUs available, you can also use the `"gloo"` backend like we did in the previous part.

Next, we can start implementing our `DistributedDataParallel` class. This class will simply wrap an existing `nn.Module` and take care of synchronizing the gradients and ensuring that all models (on all ranks) start from the same weights. Other than that, we want this wrapper to be really lightweight and behave as closesly as possible to the original module, which is why we will simply propagate the `forward` method and the `state_dict` method of the wrapped module. Since we are using the `"nccl"` backend, we can use the `ReduceOp.AVG` operation. This makes the code a bit simpler, since we do not need to explicitly pass the world size to the gradient averaging hook. The hook will simply average the gradients across all processes. If you want to use the `"gloo"` backend, you can use `functools.partial` and pass the world size to the `CustomDDP` constructor – the previous post already showed how to do this.

```python
from torch.nn import Module
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ReduceOp


class CustomDDP(Module):
    def __init__(self, module: Module):
        super().__init__()
        self.module = module
        for param in self.module.parameters():
            param.register_post_accumulate_grad_hook(grad_avg_hook)
            dist.broadcast(param.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """Make the state_dict compatible with the unwrapped module."""
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)


def grad_avg_hook(param: Tensor) -> None:
    dist.all_reduce(param.grad, op=ReduceOp.AVG)
```

## Distributing the Samples across the Ranks

The next thing we have to take care of is the distributed sampling of the dataset. If we just use our regular dataloader on all ranks, we will end up with each rank processing all the data, which is not what we want – we want each rank to only process a subset of the data, since this is the thing that will actually let us go through a single epoch faster.

PyTorch's `Sampler` interface is documented [here](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) and requires each sampler to have at least an `__iter__` method that returns an iterator over the indices of the samples, optionally also implementing `__len__`. So, while the usual sampler for a single GPU would simply return an iterator over `[0, 1, 2, ..., n-1]`, our distributed sampler will return an iterator over disjoint subsets of the dataset for each rank.

When we don't want to shuffle the dataset, i.e. during the validation loop, we can simply generate these indices as `indices = range(self.num_samples)[rank::world_size]`. However when we want to shuffle the dataset, we need to make sure that we shuffle it exactly the same way across all ranks. There are several ways of solving this:
1. Similar to how we handled broadcasting of the model weights: Randomly shuffle the indices on rank 0 and broadcast it to the other ranks. The issue with this is that we need to allocate a relatively large array to hold these indices. For large datasets this will require a lot of memory and I would therefore discourage from doing that.
2. We seed our pseudo-random number generator with a number that is generated from the epoch number. This ensures that all ranks have exactly the same shuffled indices, but the shuffling will still be different in every epoch.

For above mentioned limitations of the first approach, I will show here how to implement the second one:

```python
class CustomDistSampler:
    def __init__(self, dataset: Dataset, shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_samples = len(self.dataset) // dist.get_world_size()
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if not self.shuffle:
            indices = range(self.num_samples)[rank::world_size]
            yield from iter(indices)
        else:
            gen = Generator()
            gen.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=gen)[rank::world_size]
            yield from iter(indices.tolist())

    def __len__(self):
        return self.num_samples
```



[^1]: He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
[^2]: Xiao, Han, Kashif Rasul, and Roland Vollgraf. "Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms." arXiv preprint arXiv:1708.07747 (2017).