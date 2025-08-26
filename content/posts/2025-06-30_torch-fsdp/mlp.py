from torch.nn import Module, Linear, ReLU, ModuleList
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
        self.layers = [Linear(input_size, hidden_size)]
        for _ in range(num_layers - 1):
            self.layers.append(ReLU())
            self.layers.append(Linear(hidden_size, hidden_size))
        self.layers.append(ReLU())
        self.layers.append(Linear(hidden_size, output_size))
        self.layers = ModuleList(self.layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def forward_with_shapes(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            # instance check not working in FSDP1
            if not hasattr(layer, 'weight') or not hasattr(layer, 'bias'):
                x = layer(x)
                continue
            print(f"Layer pre-forward shape:\t {layer.weight.shape}\t {layer.bias.shape}")
            x = layer(x)
            print(f"Layer post-forward shape:\t {layer.weight.shape}\t {layer.bias.shape}")
        return x