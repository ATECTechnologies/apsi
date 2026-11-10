# Number formats: f64, f32, f16, bf16, FP8 and int8

Apsi tensors can hold six floating-point formats and the usual integer ones. All of them work in ordinary code, in
`@native` machine code and in `@gpu` kernels, with one rule that keeps results trustworthy: **small formats are a
way to store numbers, not to compute with them.** Values are read into 32-bit registers, all arithmetic happens in
f32 (or f64), and results are rounded once, correctly, when they are stored.

| Format | Name in Apsi | Bits (sign / exponent / mantissa) | Largest value | Decimal digits | Typical use |
|---|---|---|---:|---:|---|
| IEEE double | `f64` | 1 / 11 / 52 | 1.8 × 10³⁰⁸ | 15–16 | science, finance, anything that must be exact to the last digit |
| IEEE single | `f32` | 1 / 8 / 23 | 3.4 × 10³⁸ | 7 | graphics, signal processing, most of machine learning |
| IEEE half | `f16` | 1 / 5 / 10 | 65,504 | 3 | inference, images, half the memory of f32 |
| bfloat16 | `bf16` | 1 / 8 / 7 | 3.4 × 10³⁸ | 2–3 | training: f32's range in 16 bits |
| FP8 E4M3 | `fp8_e4m3` (`fp8`) | 1 / 4 / 3 | 448 | 1–2 | weights and activations of large models |
| FP8 E5M2 | `fp8_e5m2` (`bf8`) | 1 / 5 / 2 | 57,344 | 1 | gradients (more range, less precision) |
| integers | `int8`, `i32`, `i64`, `uint8`, `bool` | | | exact | quantised weights, labels, images, masks |

## Making tensors

```apsi
let w = zeros(4096, 4096, dtype=bf16)          # 32 MB instead of 128 MB in f64
let h = randn(1024, 1024).astype("float16")
let q = zeros(1000, dtype=fp8_e4m3)
var acc: f32 = 0.0                              # a scalar kept in f32 (also f16, bf16, f64)
```

`bf16` and the FP8 formats come from the `ml_dtypes` package (the format library JAX and TensorFlow use), which is
installed with Apsi. They behave like any NumPy dtype: slicing, `astype`, saving and loading all work.

## In `@native` and `@gpu` code

```apsi
@native
fn scale(x, y, a: Float):              # x, y may be f64, f32, f16, bf16, FP8 or int8 tensors
    for i in 0..len(x):
        y[i] = x[i] * a
```

| | Reading `x[i]` | Arithmetic | Writing `y[i] = v` |
|---|---|---|---|
| f64 | as is | f64 | as is |
| f32 | as is | f32 | as is |
| f16, bf16, FP8 | widened to f32 (exact) | f32 | rounded to nearest, ties to even |
| int8 … i64 | widened to Int | Int (checked for overflow) | checked: a value that does not fit is an error |

Why this design:

* **Accuracy.** Adding 100,000 copies of 0.1 in f16 arithmetic stops at 256, because 256 + 0.1 rounds back to
  256. With the data in f16 and the sum in f32 the total is 9,998.4; with a `Float` (f64) accumulator it is
  9,997.55859375 — exactly 100,000 × the f16 value of 0.1. Products of two f16 or bf16 numbers are exact in f32.
* **Speed.** Most loops are limited by memory bandwidth, not arithmetic. Reading 2 bytes instead of 8 per element
  moves 4× less data; FP8 moves 8× less.
* **The same answer everywhere.** The conversions are written with integer operations, so every CPU (with or without
  F16C) and every GPU produces the same bits. They are checked against NumPy and `ml_dtypes` for every one of the
  65,536 16-bit patterns and all 256 FP8 patterns.

Rounding details: conversions to a smaller format round to the nearest value, ties to even. NaN stays NaN. A value
too large for f16, bf16 or FP8 E5M2 becomes infinity; FP8 E4M3 has no infinity, so it becomes NaN (the OCP FP8
standard and `ml_dtypes` do the same). Use `clip(x, -448, 448)` first if you want saturation instead.

## Matrix products and Tensor Cores

A matrix product written as loops (see [Performance](PERFORMANCE.md#matrix-products-written-as-loops)) uses the
fastest kernel available for its format:

| Format | CPU | NVIDIA GPU with cuBLAS | NVIDIA GPU without cuBLAS |
|---|---|---|---|
| f64 | BLAS DGEMM | cuBLAS DGEMM | Apsi's kernel |
| f32 | BLAS SGEMM | cuBLAS SGEMM (TF32 only with `APSI_TF32=1`) | Apsi's kernel |
| f16, bf16 | Apsi's loops (f32 math) | **Tensor Cores**, f32 accumulation | Apsi's kernel (f32 math) |
| FP8, int8 | Apsi's loops | Apsi's kernel | Apsi's kernel |

cuBLAS comes with the CUDA Toolkit, or `pip install nvidia-cublas-cu12` (no toolkit needed). `apsi native info`
says whether Apsi found it.

## Smaller still: FP6, FP4, int4 and others

Formats below 8 bits (`fp6_e3m2`, `fp6_e2m3`, `fp4_e2m1`, `int4`, `nf4` …) are available for quantisation:
`quantize(x, fp4_e2m1)` packs a tensor, `dequantize` unpacks it, `@precision(fp4_e2m1)` runs a function as if its
values had that format, and `load_safetensors(path, dtype=int4)` loads weights into it. See [AI](AI.md) and
`use precision`.

## Choosing a format

* **Results you publish or bill for:** f64. It costs little on a CPU.
* **Large arrays where 7 digits are plenty:** f32 — half the memory of f64, and far faster on consumer GPUs (an
  RTX 3060 does f64 at 1/64 of its f32 speed).
* **Model weights and activations:** bf16 for training, f16 or bf16 for inference, FP8 E4M3 when memory is tight.
* **Accumulators:** always f32 or f64 (`var acc: f32 = 0.0`), whatever the data format.
