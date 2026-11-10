# Performance: CPU, GPU, RAM, SSD and number formats

## Number formats (`use precision`)

| Format | Bits | Largest value | Used for |
|---|---|---|---|
| `f64` | 64 | 1.8e308 | physics that needs the highest accuracy (the default) |
| `f32` | 32 | 3.4e38 | general AI, large simulations |
| `f16` | 16 | 65504 | inference, mixed precision |
| `bf16` | 16 | 3.4e38 | AI training (same range as f32) |
| `fp8_e4m3` (`fp8`) | 8 | 448 | weights and activations for inference/training (as on Hopper/Blackwell GPUs) |
| `fp8_e5m2` (`bf8`) | 8 | 57344 | gradients |
| `fp6_e3m2`, `fp6_e2m3` | 6 | 28 / 7.5 | OCP MX (4 values packed in 3 bytes) |
| `fp4_e2m1` (`fp4`) | 4 | 6 | highly compressed weights (2 values per byte) |
| `int8` | 8 | 127·scale | classic quantization |
| `int4` | 4 | 7·scale | compressed LLM weights |
| `uint8` | 8 | — | image/signal data (with a zero point) |

- **Bit-exact with the standards**: every conversion is tested against `ml_dtypes` (the reference library of
  JAX/TensorFlow) on 20,000 random values with **no differences** — round-to-nearest-even, subnormals and saturation included
- `quantize(x, fmt, axis=)` stores real-size codes (FP4 uses 1/16 of the memory of f64); `dequantize(q)` converts back
- `round_to(x, fmt)` rounds while staying a normal tensor, with a straight-through gradient, for **quantization-aware training**
- `qmatmul(a, b)`: matrix multiply with quantized operands, accumulated in FP32 (the same arithmetic as tensor cores)
- `@precision(bf16)`, `@precision(fp8_e4m3)` …: round the tensors going into and out of a function
- `Tensor[N, bf16]` in a type annotation: rounds the value when the function is called
- `quant_error(x, fmt)`, `format_info(fmt)`, `nbytes(x)`

**About speed:** f64/f32/f16 use the CPU's instructions directly. In the current runtime BF16/FP8/FP6/FP4/INT4
are stored as real-size codes (saving RAM, SSD and bandwidth in proportion to the bits) and decoded to FP32 for
computing. Results match the hardware exactly, but the speed of FP8/FP4 tensor cores needs the native GPU
backend of a later phase (see [`ROADMAP.md`](ROADMAP.md)). Neural networks already train on GPUs in fp32, tf32,
bf16 and fp16 through the PyTorch backend (below).

## Native code (`@native`, `parallel for`, `@gpu`)

The fastest way to run loops and numeric functions: machine code through LLVM for the exact CPU, all cores with
`parallel for`, NVIDIA GPUs with `@gpu`. On par with C, C++ and Rust, and 100–3,500× faster than bytecode on loops.
Benchmarks and the full reference: [`NATIVE.md`](NATIVE.md).

## CPU

- `@native` functions run as machine code (AVX2/AVX-512 vectorised); `parallel for` uses every core — see above
- tensor math runs on multi-threaded BLAS (the OpenBLAS/MKL that comes with numpy) on every core automatically
- other loops and functions are compiled to bytecode, 5–20× faster than the tree-walker
- `use parallel`
  - `spawn(f, args…)`, then `.wait()`, `wait_all`, `wait_any`
  - `channel(cap)` (send/recv/close, iterate with for), `lock()`, `atomic()`
  - `parallel_for(items, f)`, `parallel_reduce(items, f, combine, init)`, `pmap(f, xs)`
  - `set_threads(n)` limits the BLAS and pool threads (uses threadpoolctl when installed)
  - `cpu_count()`, `bench(f)`
- reductions over short axes (such as xyz coordinates) use an unrolled kernel, about 1.5× faster than plain numpy in N-body work

## GPU (`use gpu`)

- your own loops on the GPU: `@gpu` with a `parallel for` — no CuPy or CUDA toolkit needed, several GPUs at once
  ([`NATIVE.md`](NATIVE.md#gpu))

- `device()` returns `"cuda:0"` when an NVIDIA GPU with compute capability ≥ 7.5 (GTX 1660 or newer) and CuPy are present, otherwise `"cpu"`
- `to_device(x)`, `to_gpu(x)`, `to_cpu(x)`, `gpu_info()` (name, memory, number of SMs)
- GPU tensors work with every operator and with autodiff (through numpy's array-function protocol)
- install: `apsi pkg install gpu` (or `pip install cupy-cuda12x`), then check with `apsi doctor`

## PyTorch backend (`use layers`)

Models built with `use layers` can be trained and run by PyTorch instead of NumPy — on the CPU or on NVIDIA CUDA,
AMD ROCm, Apple MPS and Intel XPU GPUs:

- `fit(…, backend="torch", device="auto", precision="bf16", compile=true)`; `predict(…, backend="torch")`
- precisions `fp64` (matches NumPy to ~1e-15), `fp32`, `tf32` (tensor-core matmuls), `bf16` and `fp16` mixed
  precision with automatic loss scaling
- data stays on the device; fused attention (`scaled_dot_product_attention`), cuDNN LSTMs, parallel
  `index_add`/`scatter_reduce` for graph layers, one host synchronisation per epoch
- `torch_module(model, params)` for DDP/FSDP multi-GPU training, ONNX export and other PyTorch tools
- `check_torch()` verifies every layer against NumPy on your machine

Needs PyTorch (`apsi pkg install torch-cuda` for NVIDIA GPUs, `apsi pkg install torch` for the CPU) and `--allow-python`. Full guide: [`BACKENDS.md`](BACKENDS.md).

## RAM and SSD (`use data`)

- **memory-mapped tensors**
  - `open_tensor("x.npy", shape, f32)` creates a file on SSD that behaves like a normal tensor — it may be larger than RAM
  - `load_tensor(path, mmap=true)` opens a file instantly without loading it into RAM
- **streaming pipelines:** `from_tensors(X, y, batch=512, shuffle=true) |> map(f, workers=8) |> prefetch(4)`
  - batches are sliced vectorized and indices are read in order, so SSD/mmap reads stay fast
  - `map` runs in parallel on a worker pool and keeps the order
  - `prefetch` prepares data ahead of time on a separate thread, overlapping with computation
  - measured with `examples/25_data_pipeline.apsi`: **about 2 million rows per second** (4,000 batches × 512 rows) on 2 vCPUs
- `cache(dir)` stores a pipeline's output on SSD as shards, so the next pass reads from disk without recomputing; `cache()` without a folder keeps it in RAM
- `@disk_cache("dir")` remembers results of expensive functions on disk, across runs
- **every cache file is signed with HMAC-SHA256** using a per-user secret key (`~/.apsi/cache.key`); modified files are never loaded
- other sources: `from_csv` (read in chunks), `from_npy`, `from_files("data/**/*.png")`, `from_generator`, `from_range`, `from_list`
- other steps: `filter`, `shuffle(buffer)`, `batch`, `unbatch`, `take`, `skip`, `repeat`, `throughput(ds)`

## Start-up: compile cache and executables

- a program file is parsed once per process (the Guard scan, the checker, imports and the run share the result)
  and kept in the signed on-disk compile cache, so an unchanged file is not parsed again on the next run
- `apsi compile PATHS` compiles and checks programs ahead of time; `apsi compile --runtime` compiles Apsi itself
- `apsi build app.apsi` makes an executable that carries the parsed program and Apsi as bytecode; it starts
  without parsing or compiling anything (measured: 0.48 s for a small program, most of it loading numpy)
- details: the *Executables and the compile cache* section of [`LANGUAGE.md`](LANGUAGE.md)

## Measuring

- `apsi bench app.apsi -n 10` — best / median / mean / worst over several runs
- `apsi run app.apsi --time` — wall time of one run
- `:time CODE` and `:bench CODE` inside the REPL
- `@timed` on a function prints how long every call takes

## Matrix products written as loops

Write the matrix product the way the textbook does — Apsi runs it with the tuned BLAS kernel (the one behind
NumPy's `@`), in the compiled engine and in `@native` machine code alike:

```apsi
fn matmul_into(a, b, c):
    for i in 0..a.shape[0]:
        for k in 0..a.shape[1]:
            let aik = a[i, k]
            for j in 0..b.shape[1]:
                c[i, j] += aik * b[k, j]
```

| n (float64, 2 cores) | same loops in C, `-O3 -march=native` | Apsi loops | |
|---:|---:|---:|---|
| 256 | 2.90 ms | 0.63 ms | 4.6× faster |
| 512 | 28.96 ms | 4.30 ms | 6.7× faster |
| 1024 | 251.1 ms | 33.0 ms | 7.6× faster |
| 2048 | 2,409 ms | 234.2 ms | **10.3× faster** (73 GFLOP/s) |

Details and how to reproduce: [`benchmarks/matmul`](../benchmarks/matmul/README.md).

**What is recognised.** Three nested counting loops (`0..X` or `range(X)`, the outer one may be `parallel for`)
whose innermost statement is

* `c[i, j] += a[i, k] * b[k, j]` in any loop order, with the factors in either order, or with `a[i, k]` hoisted
  into `let t = a[i, k]` between the `i`/`k` loops and the `j` loop, or
* the dot-product form: `var s = 0.0`, then `for k: s += a[i, k] * b[k, j]`, then `c[i, j] = s`.

The loop bounds may be any expressions without side effects that do not depend on the loop variables
(`n`, `a.shape[0]`, `len(b)`, `m - 1` …); sub-blocks (`0..m` smaller than the arrays) are supported.

**When the loops run instead.** Only when the result is exactly what the fast path computes: `a`, `b` and `c` must be
2-D arrays of float64 (or all float32) and `c` must not share memory with `a` or `b`. Integer arrays, arrays with
units carried at run time, bounds beyond the arrays (the loops then raise the same index error they always did),
`c` overlapping an input — all run the loops as written, so behaviour never changes. The only difference you can
see is the one every optimised BLAS has: products are summed in a different order, so results can differ from the
loops in the last bits (relative differences around 1e-16 for float64).

`APSI_IDIOMS=0` turns the recognition off, for example to time the loops themselves.

## Outer loops four at a time (exact SIMD for reductions)

```apsi
for i in 0..n:
    var fx = 0.0
    for j in 0..n:
        fx += (x[j] - x[i]) * m[j] / d(i, j)
    ax[i] = fx
```

The inner loop adds into `fx` one term after another. Vectorising it along `j` would change the order of the
additions — and so the result — which strict IEEE arithmetic forbids. The outer iterations are independent,
though, so `@native` code runs **four values of `i` at once**: four accumulators, each receiving exactly the same
additions in exactly the same order as before. The four lanes are identical instruction streams on different data,
which LLVM packs into SIMD registers. Results are bit-for-bit identical to the plain loop (the test suite checks
this); only the speed changes. C compilers call this outer-loop vectorisation.

It applies when the outer loop counts by one, its body is straight-line code around one inner counting loop that
accumulates into variables declared in the outer body, tensors written in the nest are written only at the outer
index (`ax[i]`) — or only after the inner loop, like `sums[cluster] += x[i]`, which keeps its original order — and
only built-in math functions are called. Examples: n-body forces, k-means assignment, matrix-vector products,
distance matrices. `APSI_JAM=0` turns it off.
