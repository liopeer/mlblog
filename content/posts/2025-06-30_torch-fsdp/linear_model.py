from torch.nn import Parameter, Module
import torch
from torch import Tensor
from lightning import LightningModule
import torch.nn.functional as F
from torch.optim import Adam


class LinearModel(Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = Parameter(torch.randn(input_size, output_size))

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("ij,jk->ik", x, self.linear)


class LinearLightningModel(LightningModule):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.model = LinearModel(input_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-1)