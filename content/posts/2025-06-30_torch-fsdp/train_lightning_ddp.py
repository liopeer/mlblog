import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L

from linear_model import LinearLightningModel

if __name__ == "__main__":
    param_size = 8192 * 2

    torch.cuda.reset_peak_memory_stats()

    # Same parameters as DDP script
    x = torch.randn(256, param_size)
    y = torch.randn(256, param_size)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=256)

    model = LinearLightningModel(input_size=param_size, output_size=param_size)
    trainer = L.Trainer(max_epochs=5, devices=4, strategy="ddp")
    trainer.fit(model, dataloader)

    # Use trainer's global rank instead of model.rank
    if trainer.is_global_zero:
        print(
            f"Max memory allocated: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB"
        )
