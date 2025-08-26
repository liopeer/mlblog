+++
title = 'Distributed Training in PyTorch – I'
date = 2025-05-04T15:09:46+02:00
draft = false
author = 'Lionel Peer'
+++
This blog post is the first in a series in which I would like to showcase several methods on how you can train your machine learning model with PyTorch on several GPUs/CPUs across multiple nodes in a cluster. In this first part we will try to look a bit under the hood of how you can launch distributed jobs and how you can ensure proper communication between them. In follow-ups we will create our own PyTorch `DistributedDataParallel` (DDP) class and we will also look at popular frameworks such as PyTorch Lightning and resource schedulers like SLURM that can help you getting your distributed training running. We will strictly focus on data parallelism, meaning a parallelism where the whole model fits into the memory of a single GPU and we exchange gradients (and potentially batch norms) across the GPUs, while keeping the whole optimization local on each GPU.

{{< notice note >}}
You don't need access to a compute cluster or multiple GPUs to follow along. We can simulate everything using our CPU and a single computer.
{{< /notice >}}

## Terminology of Distributed Training
PyTorch offers communication between distributed processes through [`torch.distributed`](https://pytorch.org/docs/stable/distributed.html), which is a wrapper around other communication libraries, such as [Gloo](https://github.com/pytorch/gloo), [NCCL](https://github.com/NVIDIA/nccl) or [MPI](https://github.com/open-mpi/ompi). PyTorch recommends using Gloo when doing distributed training on CPUs and NCCL when using Nvidia GPUs, which is why we will be relying on Gloo over the course of this blog post.

Let's harmonize some terms here, before we continue on how one can set up distributed data parallelism:
1. *Number of Nodes*: A node is the same as a *computer* or a *machine*, it therefore refers to how many computers we use in parallel for our training.
2. *World Size*: This is the number of processes that we run in parallel. This is independent of how many nodes we run training on, i.e. if we have 2 nodes and world size 8 we are most likely running 4 processes on each node (it could also be 3+5 or any other combination, but let's assume we run equal numbers of processes on each node).
3. *Rank* or *Global Rank*: A specific process, which is identified by an integer between 0 and the world size.
4. *Node Rank*: An integer between 0 and the total number of nodes, which uniquely identifies a node.
5. *Local Rank*: The process number on a specific node, i.e. the modulus of the global rank with respect to the number of processes per node (local_rank = global_rank % processes_per_node, again assuming that all nodes have equal numbers of processes).

## Let Processes "find" each other
Before processes can communicate, we first need to make sure they know of each other's existence and how they can address each other. The first contact for this is done through the [`torch.distributed.init_process_group` function](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group). By default, this function assumes that the environment variables `MASTER_ADDR` (the IP of the master node) and `MASTER_PORT` (a free port on the master node) are set. That port on the master node, will then be used by all processes to join the process group. Besides setting environment variables, one could also use a TCP key/value store or a shared file to communicate these two parameters, however for this I refer the interested reader to the [PyTorch documentation](https://pytorch.org/docs/stable/distributed.html#initialization).

The `init_process_group` will wait until all processes have joined (or until the `timeout` parameter has been reached), therefore it needs to know how many processes in total are expected and each process should communicate its rank (so that potential duplicates can raise errors).

We can integrate all of this and create a super minimal script and launch it (in 2 separate terminals) with the commands:

{{% py_script script="procgroup_join.py" %}}

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

{{% py_script script="dist_mpspawn.py" %}}

{{< notice info >}}
As you can see from the script above, we have to pass a function to `mp.spawn`. Importantly, that function has to take the rank as the first argument, which is automatically passed by `mp.spawn`. The other arguments can be passed through the `args` parameter as a tuple. Additionally, we also give `mp.spawn` the number of processes to create through the `nprocs` parameter, which is the same as the world size.
{{< /notice >}}

You can launch this script by giving it a number of sub-processes to create through the `--world-size` command line argument. For example to launch with 8 processes:

```bash
python dist_mpspawn.py --world-size=8
```
As you can see, we also don't need to pass the environment variables before the command since we set them already in the script. Much more convenient than opening 8 terminals, right? Unfortunately, `mp.spawn` does not scale to muliple nodes, so with the current script, you'd still have to manually launch the script on each node. Workload schedulers like [SLURM](https://slurm.schedmd.com/documentation.html) can help with this, but we will cover that in a later part of this series. For now, let's just assume that we run the script on a single node with multiple GPUs.

## Averaging Gradients Across Processes
Now that we know how to initialize a process group, let's actually train a model on several processes and average the gradients over the processes. The averaging operation can be done using `torch.distributed.all_reduce`, which takes a tensor as input, averages each element of the tensor across the processes and returns the averaged tensor on every process.

{{% py_script script="dist_train_manual.py" %}}

{{< notice info >}}
If you use the `"nccl"` backend (which you should if you use multiple Nvidia GPUs), you can also use the `ReduceOp.AVG` operation instead of `ReduceOp.SUM` and avoid the manual division by the world size. Unfortunately, the averaging operation is not supported by the `"gloo"` backend, so we have to do it manually in the script above.
{{< /notice >}}

In above script we manually implemented the gradient descent step, which makes the code very complicated and very different from standard PyTorch code. This code would also not be easily compatible with other PyTorch optimizers, such as SGD with momentum or Adam. This can be avoided, by hooking the `all_reduce` operation into the model's backward pass. This can be done through the `register_post_accumulate_grad_hook` that each of the parameters has. Additionally, to beautify a bit more, we can use a context manager to set up the process group and destroy it after training. This way we can also ensure that the process group is properly cleaned up even if an error occurs during training.

{{% py_script script="dist_train_hooks.py" %}}

This is already pretty sophisticated code, and by quickly integrating an actual dataset and a real model (not a linear model), you could actually train a model with this code in distributed fashion!

## Replicating the Model Across Ranks
So far we only worked with dummy data and randomly initialized the model on all processes independently. This was okay, since we were so far not trying to train an actual model. However, once we do, it is important that we start with the same model weights across all processes – since we will only exchange the gradients. In contrast to the previously mentioned `all_reduce` function, this can be done with the `broadcast` function, which will allow a single rank (usually rank 0) to share its model weights with all other ranks.

{{% py_script script="dist_model_replication.py" %}}

## Wrap Up
In this first part of the series we have seen how we can set up a distributed process group in PyTorch and how we can average gradients across processes, particularly how we can hook the `all_reduce` operation into the model's backward pass. We have also seen how to automate the spawning of processes using `torch.multiprocessing.spawn`. In the next part we will extend the idea of averaging gradients to batch normalization layers, how to distribute dataset sampling across processes and how to create our own `DistributedDataParallel` class that allows us to further abstract away the distributed training code.