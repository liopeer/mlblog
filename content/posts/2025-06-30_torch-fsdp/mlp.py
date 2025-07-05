from torch.nn import Module, Linear, Sequential, ReLU
import torch
from torch import Tensor


class MLPModel(Module):
    def __init__(
            self, 
            input_size: int, 
            output_size: int, 
            hidden_size: int = 8192, 
            num_layers: int = 5
        ) -> None:
        super().__init__()
        layers = [Linear(input_size, hidden_size)]
        for _ in range(num_layers - 1):
            layers.append(ReLU())
            layers.append(Linear(hidden_size, hidden_size))
        layers.append(ReLU())
        layers.append(Linear(hidden_size, output_size))
        self.mlp = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)