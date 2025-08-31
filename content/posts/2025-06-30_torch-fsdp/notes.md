# Notes on FSDP

## Benchmarking Speed and Memory Usage of DDP vs FSDP

Trains a 5 layer MLP with very large 8192 hidden units on 4 GPUs using DDP, FSDP1, and FSDP2.
```bash
python ddp_vs_fsdp.py --num-devices 4
```

| Method | Runtime (s) | Max Memory (MB) |
|--------|-------------|-----------------|
| DDP    | 6.42        | 9265            |
| FSDP1  | 7.62        | 2496            |
| FSDP2  | 7.50        | 2504            |

Conclusions:
 - As expected, DDP is faster than FSDP.
 - FSDP1 and FSDP2 seem to perform pretty much the same.
 - Memory consumption difference between DDP and FSDP is significant.

## 