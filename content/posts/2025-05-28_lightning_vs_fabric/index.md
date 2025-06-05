+++
title = 'Distributed Training in PyTorch - III'
date = 2025-05-28T11:31:18+02:00
draft = true
author = 'Lionel Peer'
+++

In this third part of the series on distributed training in PyTorch, we will look at the frameworks from Lightning AI, PyTorch Lightning and Fabric, how they abstract away the complexities of PyTorch's DDP (DistributedDataParallel) – which we implemented from scratch in the previous parts – and how they can be easily extended for parallelism beyond PyTorch's DDP.

## The `ClusterEnvironment` Interface
In the previous parts, we looked at so called *non-managed* clusters, which means that we were ourselves responsible for for launching the processes. On a single node we were able to use the `torch.multiprocessing` module, but for multi-node training we had to manually launch the scripts on each node, which does not scale well if you want to train on a large number of nodes.

On the other hand, a *managed* cluster is one where a central orchestration service can not only allocate resources for a certain training job, but can also launch the processes on each of the nodes (i.e. each *node rank*) and each *local rank* (the processes responsible for a single GPU on a node). This is where the `ClusterEnvironment` interface comes into play and its most important methods are: 

 - The `.detect()` method, which is used to detect whether the current job is running on a managed cluster of the given type.
 - The `.creates_processes_externally` property, which indicates whether the cluster environment is responsible for launching the processes or if Lightning has to do it manually.
 - The `.teardown()`, which is called at the end of the training job to clean up any resources that were allocated by the cluster environment.

Besides that, the `ClusterEnvironment` also provides properties and methods to get the number of processes (*world size*), the local rank, the node rank, and also the address and port of the master node, which is used for communication between the processes.

The default `ClusterEnvironment` implementation in Lightning is the `LightningEnvironment`, which is used for non-managed single node training. This environment will make sure that Lightning takes care of launching the processes. An example of a managed cluster environment is the `SLURMEnvironment`. [SLURM](https://slurm.schedmd.com/) is a widely used workload manager for clusters, which can allocate resources and launch processes on the nodes. Therefore if this kind of environment is detected, Lightning will not launch the processes itself, but will rely on SLURM to do so.

## The `Strategy` Interface
The core of the distributed training with both of Lightning's frameworks is the `Strategy` interface.