class CustomSyncBatchNorm2d(Module):
    def __init__(self, module: BatchNorm2d):
        super().__init__()
        self.module = module

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)

    def _normalize_affine(
        self,
        input: Tensor,  # (N, C, H, W)
        running_mean: Tensor,  # (C,)
        running_var: Tensor,  # (C,)
        weight: Tensor | None,  # (C,)
        bias: Tensor | None,  # (C,)
        eps: float,
    ) -> Tensor:
        y = (input - running_mean[None, :, None, None]) / torch.sqrt(
            running_var[None, :, None, None] + eps
        )
        if weight is not None:
            y = y * weight[None, :, None, None]
        if bias is not None:
            y = y + bias[None, :, None, None]
        return y

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with synchronized batch normalization.

        Args:
            x: Input tensor of shape (N, C, H, W).
        Returns:
            Tensor of the same shape as input.
        """
        if not self.training:
            return self._normalize_affine(
                input=x,
                running_mean=self.module.running_mean,
                running_var=self.module.running_var,
                weight=self.module.weight,
                bias=self.module.bias,
                eps=self.module.eps,
            )
        with torch.no_grad():
            # Compute the global mean and variance.
            glob_num_feats = x.numel() // x.size(1) * dist.get_world_size()
            batch_sum = torch.sum(x, dim=(0, 2, 3))  # (C,)
            batch_ssum = torch.sum(x * x, dim=(0, 2, 3))  # (C,)
            dist.all_reduce(batch_sum, op=ReduceOp.SUM)
            dist.all_reduce(batch_ssum, op=ReduceOp.SUM)
            # Compute the global mean and variance.
            batch_mean = batch_sum / glob_num_feats
            batch_var = batch_ssum / glob_num_feats - batch_mean * batch_mean

            # Update running statistics.
            self.module.running_mean = (
                1 - self.module.momentum
            ) * self.module.running_mean + self.module.momentum * batch_mean
            self.module.running_var = (
                1 - self.module.momentum
            ) * self.module.running_var + self.module.momentum * batch_var

        return self._normalize_affine(
            input=x,
            running_mean=batch_mean,
            running_var=batch_var,
            weight=self.module.weight,
            bias=self.module.bias,
            eps=self.module.eps,
        )

    @classmethod
    def convert_sync_batchnorm(cls, module: Module) -> Module:
        if isinstance(module, BatchNorm1d) or isinstance(module, BatchNorm3d):
            raise NotImplementedError("Only BatchNorm2d is implemented.")
        if isinstance(module, BatchNorm2d):
            return cls(module)
        for name, child in module.named_children():
            module.add_module(name, cls.convert_sync_batchnorm(child))
        return module