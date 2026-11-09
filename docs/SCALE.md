# Planning large models: `apsi plan` and `use scale`

Training a 21B, 500B, 1T or 2T-parameter model is decided by arithmetic before any code runs: how much memory each
GPU needs, how the model must be split, how many GPUs, and how many days. `apsi plan` does that arithmetic.

```text
$ apsi plan 1T --active 32B --tokens 15T --precision fp8
Training plan: 1T params (32B active), 15T tokens: 520 x H100 (65 nodes), 8x1x65 TPxPPxDP, none recompute, 68 GB/GPU, 89.3 days at 40% MFU

  Model       1T parameters, 32B active per token (MoE)
  Data        15T tokens (~55,879 GB as 32-bit token ids), 3,576,279 steps of 4.19M tokens
  Hardware    520 x H100 (80 GB) in 65 node(s)
  Layout      TP 8 x PP 1 x DP 65 (tensor x pipeline x data parallel), ZeRO-3/FSDP sharding
  Memory/GPU  68 GB (model states in total: 14,901 GB)
  Duration    89.3 days
  Serving     931 GB weights in fp8 + KV cache -> 13 x H100
```

| Option | Meaning |
|---|---|
| `PARAMS` | total parameters: `21B`, `500B`, `1T`, `2T`, … |
| `--active N` | active parameters per token for mixture-of-experts models (compute uses these, memory uses all) |
| `--tokens N` | training tokens (default 20 per active parameter, the Chinchilla-optimal amount) |
| `--gpu NAME` | `H100` (default), `H200`, `B200`, `GB200`, `A100`, `A100-40`, `L40S`, `MI300X`, `MI325X`, `RTX5090`, `RTX4090`, `RTX3090`, `GTX1660` |
| `--gpus N` / `--days D` | the GPUs you have, or a target duration (default 90 days) |
| `--precision bf16\|fp8`, `--optimizer adamw\|adam8bit\|sgd`, `--seq`, `--layers`, `--width`, `--mfu` | the rest of the setup |

The same from a program: `use scale`, then `plan("2T", gpu="B200", precision="fp8")`, `inference("70B",
precision="fp8")`, `gpu_table()`.

## How it is computed

- **Compute:** 6·N·D + 12·L·d·s·D FLOPs (N active parameters, D tokens, L layers, d width, s sequence length),
  divided by GPUs × peak × MFU (40 % by default, typical of well-tuned large runs).
- **Model states:** 16 bytes per parameter with mixed-precision AdamW (BF16 weights and gradients, FP32 master
  weights and two moments), sharded over every GPU (ZeRO-3 / FSDP); 10 bytes with 8-bit Adam.
- **Activations** (Korthikanti et al. 2022): per layer and micro-batch s·b·d·(34 + 5·a·s/d) bytes without
  recomputation, 34·s·b·d with selective and 2·s·b·d with full recomputation, divided by tensor parallelism.
- **Layout:** the planner tries every tensor-parallel degree (within one NVLink domain), pipeline depth and
  recomputation level, keeps those that fit in 90 % of GPU memory and whose data parallelism is not wider than the
  batch, and picks the one that meets the target with the fewest GPUs, counting recomputation and pipeline idle time.
- **Honest limits:** clusters are capped at 200,000 GPUs (`max_gpus=`); when a target cannot be met the plan says so
  and gives the fastest layout. Real runs differ with the framework and the network.

## What the numbers say

| Model | Plan (H100 unless noted, 90-day target) |
|---|---|
| 21B dense, 420B tokens | 20 GPUs, about 88 days |
| 70B dense, 1.4T tokens | 208 GPUs (8 × 26), about 89 days |
| 1T MoE, 32B active, 15T tokens, FP8 | 520 GPUs, about 89 days |
| 2T MoE, 64B active, 1.28T tokens, B200 FP8 | 226 B200, about 15 days |
| 2T dense, 40T tokens, B200 FP8 | 31,248 B200, about 103 days |
| 2T dense, 40T tokens | needs a ~200,000-GPU cluster for ~500 days: this is why models of this size are mixture-of-experts |

Training itself runs on PyTorch (`fit(…, backend="torch")` on one GPU, or `torch_module(model, params)` with
PyTorch FSDP / DDP across many GPUs, see [`BACKENDS.md`](BACKENDS.md)); your own numeric kernels run with `@native`
and `@gpu` ([`NATIVE.md`](NATIVE.md)).
