# dist_train_manual.py
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.plugins.environments import LightningEnvironment
import torch


def to_run():
    strategy.setup_environment()

    print(f"Rank {strategy.global_rank} has joined the process group.")

    strategy.teardown()


if __name__ == "__main__":
    strategy = DDPStrategy(
        accelerator=CPUAccelerator(),
        parallel_devices=[torch.device("cpu")] * 4,
        cluster_environment=LightningEnvironment(),
    )
    strategy._configure_launcher()
    strategy.launcher.launch(to_run)
