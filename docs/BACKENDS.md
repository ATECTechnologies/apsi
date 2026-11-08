# Compute backends: NumPy and PyTorch (CPU, CUDA, ROCm, MPS, XPU)

Every model built with [`use layers`](LAYERS.md) runs on two backends without changing a line of the model:

| Backend | Devices | Number formats | When to use it |
|---|---|---|---|
| **NumPy** (default) | every CPU core through multi-threaded BLAS | float64 (exact reference results) | always available; physics-grade accuracy, small and medium models, reproducible tests |
| **PyTorch** | CPU, NVIDIA CUDA, AMD ROCm, Apple MPS, Intel XPU | fp64, fp32, tf32, bf16 and fp16 mixed precision | large models and data sets, GPUs, cuDNN, fused (flash) attention, `torch.compile` |

The PyTorch backend is not a separate model format: it executes the *same* layers with torch kernels and torch
autograd, and gives the trained parameters back as an ordinary Apsi parameter tree. A model trained on a GPU can be
evaluated on NumPy, saved with `save_params`, served by `use web`, or handed to PyTorch as a real `torch.nn.Module`.

- [1. Setup](#1-setup)
- [2. Training on PyTorch](#2-training-on-pytorch)
- [3. Devices](#3-devices)
- [4. Precision and speed](#4-precision-and-speed)
- [5. Real torch.nn.Module objects](#5-real-torchnnmodule-objects)
- [6. Checking the backend on your machine](#6-checking-the-backend-on-your-machine)
- [7. What runs where](#7-what-runs-where)
- [8. Troubleshooting](#8-troubleshooting)

---

## 1. Setup

1. Install **one** PyTorch build into the Python that runs Apsi. A CUDA build also runs on the CPU, so with an
   NVIDIA card the CUDA build is all you need. The simplest way picks it for you:

   ```bash
   apsi pkg install torch-auto      # reads the NVIDIA driver version (nvidia-smi) and installs the matching build
   ```

   (`apsi pkg install ai` does the same together with ONNX Runtime and Transformers, and the setup wizard's
   *PyTorch* option uses it too.) Or choose the build yourself:

   | Your hardware | Command | Build |
   |---|---|---|
   | NVIDIA GTX 16 / RTX 20, 30, 40 series, driver 560 or newer | `apsi pkg install torch-cuda` | PyTorch ≥ 2.6 + CUDA 12.6 |
   | NVIDIA RTX 50 series (or driver 570+) | `apsi pkg install torch-cu128` | PyTorch ≥ 2.7 + CUDA 12.8 |
   | NVIDIA with an older driver (550+) | `apsi pkg install torch-cu124` | PyTorch 2.3–2.6 + CUDA 12.4 |
   | NVIDIA GTX 10 series or driver 520+ | `apsi pkg install torch-cu118` | PyTorch + CUDA 11.8 |
   | no NVIDIA GPU (CPU, or Apple silicon on macOS) | `apsi pkg install torch` | CPU build (MPS is included on macOS) |
   | AMD GPU (Linux, ROCm) | `apsi pkg install torch --index-url https://download.pytorch.org/whl/rocm6.2` | ROCm build |

   `nvidia-smi` shows your driver version. `apsi pkg install ai-gpu` installs the CUDA 12.6 build together with
   ONNX Runtime GPU, CuPy and Transformers.

   - **Already installed?** Any PyTorch 2.3 or newer already present in Apsi's Python is used as it is — nothing
     is downloaded again (`apsi pkg list` shows the installed build, e.g. `2.6.0+cu126`). Installing another build
     replaces the old one, so there is never more than one.
   - pip keeps downloaded wheels in its cache, so a build you downloaded before for the **same Python version** is
     reused instead of downloaded again.
   - The Apsi installer has its own Python (in `%LOCALAPPDATA%\Programs\Apsi\runtime` on Windows); a PyTorch that
     you installed into another Python is not visible to it. To use that Python and its PyTorch instead, run Apsi
     from the source folder with it: `python -m apsi run --allow-python app.apsi`.
2. Run programs that use the torch backend with the **`python` permission**, because PyTorch is Python code that
   runs outside Apsi's sandbox:

   ```bash
   apsi run --allow-python train.apsi
   apsi --allow-python train.apsi          # the same, short form
   ```

   In a project, put it in `apsi.toml` once: `[run] allow = ["python"]`.
3. Check what Apsi sees: `apsi --allow-python -e "use layers; devices()"` (a list of records `name`, `kind`, `detail`).

Without the permission `backend="torch"` stops with a clear message, and `backend="auto"` quietly stays on NumPy.

## 2. Training on PyTorch

Add `backend="torch"` to `fit` (and optionally to `predict`):

```apsi
use layers
use nn

let lm = transformer_lm(vocab, d=256, heads=8, layers=6, context=256, dropout=0.1)
let r = fit(lm, lm.init(seed=0), inputs, targets, loss="cross_entropy",
            opt=adamw(lr=3e-4, weight_decay=0.1), epochs=10, batch_size=64, clip=1.0,
            schedule=fn(s) => cosine_lr(s, 20000, base=3e-4, warmup=500),
            backend="torch", device="cuda", precision="bf16", compile=true)

print("trained on", r.device, "· last loss", r.history[-1])
save_params("lm.npz", r.params)                 # r.params is a normal Apsi parameter tree
let logits = predict(lm, r.params, prompt, backend="torch", device="cuda")
```

Every `fit` option works the same on both backends — `loss`, `opt`, `epochs`, `batch_size`, `seed`, `clip`,
`schedule`, `graph`, `mask`, `val`, `verbose` — plus three torch-only options:

| Option | Values | Default |
|---|---|---|
| `device` | `"auto"` (CUDA/ROCm → MPS → XPU → CPU), `"cpu"`, `"cuda"`, `"cuda:1"`, `"mps"`, `"xpu"` | `"auto"` |
| `precision` | `"fp64"`, `"fp32"`, `"tf32"`, `"bf16"`, `"fp16"` | `"fp32"` |
| `compile` | `true` compiles the forward pass with `torch.compile` (falls back to eager mode, with a note, if the compiler is not available) | `false` |

`backend="auto"` uses PyTorch when it is installed and permitted, NumPy otherwise — handy for programs that should
run everywhere.

How the optimizer is mapped: `sgd(lr, momentum)` → `torch.optim.SGD`, `adam(lr, b1, b2, eps)` → `torch.optim.Adam`,
`adamw(…, weight_decay)` → `torch.optim.AdamW` with the same hyper-parameters. On the torch backend `opt` may also
be a name: `"rmsprop"`, `"adagrad"`, `"nadam"`, `"radam"` (PyTorch defaults; learning rate via `schedule`).

Losses: `"mse"`, `"mae"`, `"huber"`, `"cross_entropy"` (logits of any shape `[…, classes]` with integer labels),
`"bce"` (numerically stable, from logits) and `"vae"`. A loss written in Apsi works on the NumPy backend; on the
torch backend use a named loss.

## 3. Devices

`devices()` lists what this machine offers, one record per device:

```apsi
for d in devices():
    print(d.name, d.kind, d.detail)
```

```text
numpy   cpu    NumPy 2.1.3, float64, 16 threads
cpu     cpu    PyTorch 2.5.1, 16 threads
cuda:0  cuda   NVIDIA GeForce RTX 4090, 23.6 GiB, bf16 yes
```

AMD GPUs with the ROCm build of PyTorch appear as `cuda:N` with kind `rocm` (that is how PyTorch names them);
Apple-silicon GPUs as `mps`; Intel GPUs as `xpu:N`.

## 4. Precision and speed

| `precision` | Weights stored as | Arithmetic | Notes |
|---|---|---|---|
| `fp64` | float64 | float64 | bit-for-bit comparable with the NumPy backend; slow on consumer GPUs; not on MPS |
| `fp32` | float32 | float32 | the default |
| `tf32` | float32 | TF32 tensor cores for matrix products (NVIDIA Ampere and newer) | ~3–8× faster matmuls, ~1e-3 relative error |
| `bf16` | float32 | bfloat16 autocast (mixed precision) | the usual choice for training on modern GPUs; no loss scaling needed |
| `fp16` | float32 | float16 autocast with dynamic loss scaling (`GradScaler`) | for GPUs without bf16 |

What makes the torch backend fast:

- the training data is moved to the device once; mini-batches are sliced there (no copy per step)
- the loss is synchronised with the CPU only once per epoch
- attention uses `torch.nn.functional.scaled_dot_product_attention` — Flash/memory-efficient kernels on GPUs
- `lstm_layer` runs as a single cuDNN call on CUDA/XPU (same weights, mapped onto `torch.nn.LSTM`); on the CPU the
  input projection for all time steps is computed in one matrix product
- graph layers use `index_add` / `scatter_reduce`, which run in parallel on the GPU
- `compile=true` fuses element-wise operations with `torch.compile`

For multi-GPU training (DDP/FSDP), ONNX export or TorchScript, take the model to PyTorch as a module (next section)
and use PyTorch's own tools.

## 5. Real `torch.nn.Module` objects

```apsi
use layers
let torch = pyimport("torch")

let net = resnet_mlp(16, 4, width=128, blocks=4)
let m = torch_module(net, net.init(seed=0), device="cuda")     # a torch.nn.Module
print(m)                                                       # ApsiModule(…, the flat parameter list)
let y = m(torch.randn(8, 16).cuda())                           # forward pass in PyTorch
# … train it with any PyTorch code: torch.optim, DDP, Lightning, …
let params = params_from_torch(net, m)                         # back to an Apsi parameter tree
```

The module holds every weight as a `torch.nn.Parameter` (`m.parameters()`, `m.state_dict()`, `m.to(…)`,
`m.train()` / `m.eval()` all behave as usual) and its `forward(x, graph=…, mask=…)` runs the Apsi model on torch
tensors. The parameter order follows the Apsi parameter tree, so `params_from_torch` always maps the weights back to
the right places.

## 6. Checking the backend on your machine

`check_torch()` runs every layer type — dense, convolution, pooling, batch/layer/RMS norm, LSTM, GRU, attention,
transformers, residual/parallel/DAG composition, siamese weight sharing, GCN, GraphSAGE, GAT, GIN, VAE — on NumPy and
on PyTorch in float64 and compares the outputs **and the gradients**. It uses the GPU when there is one
(`device="auto"`); pass `device="cpu"` to check the CPU:

```bash
apsi --allow-python -e "use layers; check_torch()"
```

In PowerShell, put code that contains double quotes inside single quotes:
`apsi --allow-python -e 'use layers; check_torch(device="cpu")'`.

```text
mlp                      output  2.22e-16   gradient  4.44e-16
lstm                     output  5.55e-17   gradient  3.33e-16
gat                      output  1.11e-16   gradient  8.88e-16
…
OK: largest difference 1.78e-15 on cpu
```

The project's test suite runs the same check: `python -m unittest tests.test_torch` (skipped when PyTorch is not
installed).

## 7. What runs where

| Feature | NumPy | PyTorch |
|---|---|---|
| every layer, composition and zoo model of `use layers` | ✓ | ✓ |
| `lambda_layer(fn)` and `activation(fn)` with an Apsi function | ✓ | named activations only; a lambda layer needs a torch version (`lambda_layer(fn, torch_fn=…)` from Python) |
| a loss written in Apsi | ✓ | use a named loss |
| units of measurement on tensors | ✓ | values are converted to plain numbers (SI) |
| exact float64 reference results | ✓ | with `precision="fp64"` |
| GPUs, mixed precision, `torch.compile` | — | ✓ |
| `grad`, `value_and_grad`, `hessian` on your own functions | ✓ | (inside `fit` only) |

## 8. Troubleshooting

| Message | Fix |
|---|---|
| `fit(backend="torch") needs permission 'python'` | run with `--allow-python` (or `[run] allow = ["python"]` in `apsi.toml`) |
| `backend="torch" needs PyTorch` | `pip install torch` into the Python that runs Apsi (`apsi doctor` shows its version) |
| `device="cuda": PyTorch sees no CUDA/ROCm GPU` | install the GPU build of PyTorch and a current driver, or use `device="auto"` |
| `precision="fp64" is not supported on Apple MPS` | use `fp32` on MPS |
| `the torch backend has no implementation of …` | a custom layer class without `torch_apply`; use `backend="numpy"` for it |
| `torch.compile failed (…); continuing eagerly` | not an error — the compiler (e.g. a C++ toolchain for Inductor) is missing; training continues without it |
| results differ slightly between backends | expected with fp32/bf16 (rounding); `precision="fp64"` matches NumPy to ~1e-15 |
