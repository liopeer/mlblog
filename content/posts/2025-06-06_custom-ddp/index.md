+++
title = 'Distributed Training in PyTorch - II'
date = 2025-09-05T20:48:33+02:00
draft = false
author = 'Lionel Peer'
+++
In the [part I of this series]({{< relref "/posts/2025-05-04_dist_comm" >}}), we saw how we could
- launch different processes with `mp.spawn` or through multiple terminal windows
- group processes in process groups for communication and synchronization
- implement a distributed training loop by manually synchronizing gradients or by hooking them into the backward pass

In this part, we will now follow up on this and implement our own simplified version of `DistributedDataParallel` (DDP) and other helpers for distributed training – particularly distributed sampling of data. Initially I meant to also include a custom `SyncBatchNorm` implementation, however this requires knowledge on how to extend PyTorch's Autograd engine, which I would first like to cover separately. The helpers and our DDP implementation will allow us to better structure our code and abstract away some of the boilerplate code that we had to use inside the training loop. We will then verify the correctness of our implementation by training a ResNet18[^1] on the FashionMNIST[^2] dataset and comparing the results to training it on a single GPU.

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

Since we plan on actually training a model this time, we will use the `"nccl"` backend, which is optimized for GPUs. If you don't have multiple GPUs available, you can also use the `"gloo"` backend like we did in the previous part. However, this will require you to modify some of the calls to `all_reduce` since the `"gloo"` backend does not support the `ReduceOp.AVG` operation.

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
1. Similar to how we handled broadcasting of the model weights: Randomly shuffle the indices on rank 0 and broadcast it to the other ranks. The issue with this is that in order to use the broadcasting option we would need to allocate it on the GPU first, taking away precious GPU memory which we actually want to use for the model optimization. It is generally not encouraged to utilize the GPU during the data loading phase, especially for something that needs no computation, only consumes memory.
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

Relatively straightforward, right? We just need to make sure that we increment the sampler's `epoch` attribute at the end of each epoch in order to get different shufflings in each epoch.

## Training with our Custom DDP
We will demonstrate the correctness of our implementation by training a ResNet18 on the FashionMNIST dataset. Compared to a normal PyTorch training loop there is not much that changes:
- We wrap our model with `CustomDDP` after moving it to the correct GPU.
- As mentioned earlier, we keep PyTorch's `SyncBatchNorm` implementation and don't implement our own version of it in this post. Any PyTorch model simply be wrapped by `SyncBatchNorm.convert_sync_batchnorm` in order to convert all `BatchNorm` layers to `SyncBatchNorm` layers. This will ensure that the batch statistics are calculated on the whole (global) batch instead of just the local sub-batch on each rank. 
- Instead of specifying `shuffle=True` or `shuffle=False` in the dataloader, we will instead pass our custom sampler through the `sampler` argument. 
- Before printing the losses we also make sure to average them across all ranks, so that we can directly compare the losses printed by rank 0 to the losses of the single-GPU training.
- Finally, we need to make sure that we set the `epoch` attribute of the samplers at the end of each epoch, so that we get different shufflings in each epoch.

{{% py_script script="custom_ddp.py" %}}

## Verifying the Correctness
Creating 100% coherent training runs between a single GPU and multiple GPUs would take a bit more engineering effort, since we would need to make sure that all sources of randomness are controlled. However, we can check how the training curves behave when we train on a single GPU and with multiple GPUs. For this, I ran the training loop from above on 2 and 4 GPUs and I created a "standard" single-GPU training loop that does not use any components required for distributed training. The outputs are shown below and show that the behaviour between the runs is consistent. And if you don't believe me, you can also download the script below and run it yourself. 😊

```bash
Running distributed training with 2 GPUs...
Iteration 1/10  Train Loss: 0.5944      Val Loss: 0.4728
Iteration 2/10  Train Loss: 0.3944      Val Loss: 0.4090
Iteration 3/10  Train Loss: 0.3384      Val Loss: 0.3770
Iteration 4/10  Train Loss: 0.3009      Val Loss: 0.3555
Iteration 5/10  Train Loss: 0.2744      Val Loss: 0.3463
Iteration 6/10  Train Loss: 0.2497      Val Loss: 0.3527
Iteration 7/10  Train Loss: 0.2325      Val Loss: 0.3287
Iteration 8/10  Train Loss: 0.2161      Val Loss: 0.3372
Iteration 9/10  Train Loss: 0.2022      Val Loss: 0.3346
Iteration 10/10 Train Loss: 0.1876      Val Loss: 0.3350

Running distributed training with 2 GPUs...
Iteration 1/10  Train Loss: 0.5937      Val Loss: 0.4382
Iteration 2/10  Train Loss: 0.3910      Val Loss: 0.3805
Iteration 3/10  Train Loss: 0.3356      Val Loss: 0.3603
Iteration 4/10  Train Loss: 0.3019      Val Loss: 0.3261
Iteration 5/10  Train Loss: 0.2735      Val Loss: 0.3098
Iteration 6/10  Train Loss: 0.2505      Val Loss: 0.3067
Iteration 7/10  Train Loss: 0.2317      Val Loss: 0.3118
Iteration 8/10  Train Loss: 0.2144      Val Loss: 0.2965
Iteration 9/10  Train Loss: 0.2006      Val Loss: 0.2921
Iteration 10/10 Train Loss: 0.1860      Val Loss: 0.2902

Running single-GPU training...
Iteration 1/10  Train Loss: 0.5893      Val Loss: 0.4605
Iteration 2/10  Train Loss: 0.3940      Val Loss: 0.3993
Iteration 3/10  Train Loss: 0.3376      Val Loss: 0.3635
Iteration 4/10  Train Loss: 0.3005      Val Loss: 0.3473
Iteration 5/10  Train Loss: 0.2735      Val Loss: 0.3341
Iteration 6/10  Train Loss: 0.2520      Val Loss: 0.3263
Iteration 7/10  Train Loss: 0.2330      Val Loss: 0.3237
Iteration 8/10  Train Loss: 0.2176      Val Loss: 0.3175
Iteration 9/10  Train Loss: 0.2005      Val Loss: 0.3121
Iteration 10/10 Train Loss: 0.1881      Val Loss: 0.3189
```

{{% py_script_downloadonly script="test_custom_ddp.py" %}}

[^1]: He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
[^2]: Xiao, Han, Kashif Rasul, and Roland Vollgraf. "Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms." arXiv preprint arXiv:1708.07747 (2017).