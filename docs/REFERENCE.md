# Apsi 1.10.0 — everything on one page

This page is the whole language in one place: every piece of syntax with a short example, then **every built-in
function, every standard module and every member, and every command** of Apsi 1.10.0, generated from the runtime
itself so nothing is missing. Read it top to bottom once, then use it with your browser's search (Ctrl+F).
The long-form explanations live in the [language reference](/docs/1.10.0/language).

## How a program looks

```apsi
use physics.constants                   # modules: use NAME / use NAME as n
import "helpers.apsi" as h              # another file of yours

# comments start with #
let g = 9.81[m/s^2]                     # let = constant, units in [ ] or after a number: 2kg, 3.5s
var t = 0s                              # var = variable
fn fall(h: Length) -> Time = sqrt(2 * h / g)     # one-line function with unit-checked types
print(f"10 m takes {fall(10m):.3f}")    # f-strings with format specs
```

Blocks are indented (4 spaces), like Python. Lines may continue inside ( ), [ ], { }.

## Values

| Write | Type | Notes |
|---|---|---|
| `42`, `1_000_000`, `-7` | `Int` | any size |
| `3.14`, `6.02e23`, `inf`, `nan` | `Float` | 64-bit IEEE |
| `2kg`, `9.81[m/s^2]`, `1.5[kWh]` | quantity | number + SI unit; wrong units are an error before running |
| `3.0 ± 0.1`, `x +/- dx` | uncertain | errors propagate through every formula |
| `exact("1/3")` | exact | rational numbers without rounding |
| `true`, `false`, `none` | `Bool`, none | |
| `"text"`, `f"x = {x:.2f}"` | `Str` | `+`, `*`, slicing, `.upper() .split() .replace() .strip() .find() …` |
| `[1, 2]`, `(1, 2)`, `{"k": 1}` | `List`, `Tuple`, `Dict` | Python methods: `.append .pop .get .items .keys …` |
| `0..10`, `0..=10`, `0s..5s` | `Range` | end excluded / included; ranges with units |
| `tensor([[1, 2], [3, 4]])`, `zeros(3, 3)`, `randn(64)` | `Tensor` | NumPy-style arrays, may carry a unit; `@` is matrix product |
| `record(W=w, b=b)` | `Record` | named fields, no declaration |
| `fn(x) => x * 2` | `Fn` | functions are values |

## Variables and operators

```apsi
let a = 1                  # cannot be reassigned
var b = 2                  # can: b = 3, b += 1 (also -= *= /= //= %= **= @=)
let x, y = (1, 2)          # unpacking
a + b  a - b  a * b  a / b  a // b  a % b  a ** b  a @ b       # arithmetic, matrix product
a == b  a != b  a < b  a <= b  a ~= b                          # ~= approximately equal (also checks units)
a and b  a or b  not a     x in xs                             # logic, membership
a if cond else b           data |> normalize |> train          # conditional, pipe
xs[0]  xs[-1]  xs[1:3]  xs[::2]  A[:, 0]  A[A > 0]            # indexing and slicing
```

## Control flow

```apsi
if x > 0:
    print("positive")
elif x == 0:
    print("zero")
else:
    print("negative")

while n > 1:
    n //= 2
for i in 0..10:                         # also lists, strings, dict keys, tensors, files, channels
    if i == 3:
        continue
    if i == 7:
        break
for i, item in enumerate(items):
    pass
assert total > 0, "total must be positive"
let squares = [x * x for x in 0..10 if x % 2 == 0]
let pairs = [(i, j) for i in 0..3 for j in 0..3 if i != j]
```

## Errors

`error("message")` stops the program with a `RuntimeError`; `assert` stops with an `AssertionError`.
Apsi 1.10.0 has no `try`/`except` — that arrived in [2.0.0](/docs/2.0.0/reference#errors).

## Functions

```apsi
fn area(w, h = 1.0) = w * h             # one expression
fn stats(xs: List) -> Tuple:            # a block
    let m = mean(xs)
    return m, std(xs)
area(2.0, h=3.0)                        # keyword arguments
let double = fn(x) => x * 2             # lambda
fn attention[B, T, D](q: Tensor[B, T, D], k: Tensor[B, T, D]) -> Tensor[B, T, T]:
    return q @ transpose(k, (0, 2, 1)) / sqrt(D)      # shapes are checked at every call
```

Decorators: `@memo` (cache results), `@precision(f32)` / `@precision(exact)`.

## Structs (your own types)

```apsi
struct Particle:
    pos: Tensor[3, Length]
    vel: Tensor[3, Velocity]
    mass: Mass = 1kg
    fn kinetic(self) -> Energy = 0.5 * self.mass * dot(self.vel, self.vel)

let p = Particle(tensor([0.0, 0, 0]) * u.m, tensor([1.0, 0, 0]) * u.m / u.s)
print(p.kinetic())                      # 0.5 J
```

## Physical units

`2kg`, `3.5[m/s]`, `9.81[m/s^2]`, `1[kWh]`, `90deg` — every SI unit and prefix. Adding a length to a time, or
passing a mass where a `Length` is declared, is reported **before the program runs**. `d as [km]` converts
for display; `magnitude(x)` gives the plain number in SI units; `unit(x)` and `dim(x)` describe it. Type names such as
`Length`, `Mass`, `Time`, `Velocity`, `Force`, `Energy`, `Power`, `Pressure` are used in annotations.

## Differential equations (`evolve`)

```apsi
var x = 1m
var v = 0[m/s]
evolve x, v over t in 0[s]..10[s] step 1[ms] using rk4 record every 10 into traj:
    x' = v
    v' = -4[N/m] / 1kg * x
```

Methods: `euler`, `midpoint`, `rk4`, `rk45` (adaptive, `tol 1e-10`), `symplectic`, `verlet`.

## Gradients

`grad(f)(x)`, `value_and_grad(f)(x)`, `hessian(f)`, `jacobian(f)`, `derivative(f, x)` — on numbers, tensors,
quantities (the gradient has the right units) and parameter trees (records, structs, lists, dicts).

## Tests

```apsi
test "falling takes longer from higher":
    assert fall(20m) > fall(10m)
```

`apsi test` runs every `test` block of a file, folder or project.

## Permissions

Programs run sandboxed. Files, network, environment variables and Python need explicit flags:
`--allow-read[=DIRS]`, `--allow-write[=DIRS]`, `--allow-net[=HOSTS]`, `--allow-env`, `--allow-python[=safe|all]`,
or `--allow-all`. `apsi audit app.apsi` prints exactly the flags a program needs.


## Built-in functions and values (201)

Available everywhere without `use`. Grouped by first letter; the signature shows parameters (`=…` means optional).

### A

| Name | What it is |
|---|---|
| `abs(x)` | absolute value (element-wise for tensors; keeps units) |
| `acos(x)` | inverse cosine, radians |
| `all(iterable, /)` | Return True if bool(x) is True for all values x in the iterable. |
| `any(iterable, /)` | Return True if bool(x) is True for any x in the iterable. |
| `approx(a, b, rel=1e-06, atol=1e-12)` | true when a and b are equal within a relative and absolute tolerance (also `a ~= b`) |
| `arange(start, stop=None, step=1)` | evenly spaced values in [start, stop) with a step, as a tensor |
| `argmax(x, axis=None)` | index of the largest value (along an axis) |
| `argmin(x, axis=None)` | index of the smallest value (along an axis) |
| `args()` | Command-line arguments after the program name: `apsi run app.apsi a b` -> ["a", "b"]. |
| `asin(x)` | inverse sine, radians |
| `atan(x)` | inverse tangent, radians |
| `atan2(y, x)` | angle of the point (x, y), radians |

### B

| Name | What it is |
|---|---|
| `bf16` | value |
| `bf8` | value |
| `bin(number, /)` | Return the binary representation of an integer. |
| `bool(/, *args, **kwargs)` | true/false of a value |

### C

| Name | What it is |
|---|---|
| `callable(obj, /)` | Return whether the object is callable (i.e., some kind of function). |
| `cbrt(x)` | cube root |
| `ceil(x)` | round up to an integer |
| `chars(s)` | the characters of a string as a list |
| `chr(i, /)` | Return a Unicode string of one character with ordinal i; 0 <= i <= 0x10ffff. |
| `clip(x, lo, hi)` | limit values to [lo, hi] |
| `cload(path)` | load a C library (needs --allow-ffi) |
| `clock()` | time since the program started, in seconds (a Quantity) |
| `concat(items, axis=0)` | join tensors along an axis |
| `cos(x)` | cosine (radians or an angle quantity) |
| `cosh(x)` | hyperbolic cosine |
| `cross(a, b)` | cross product of 3-vectors |
| `cstr` | value |
| `cumsum(x, axis=None)` | running sum |

### D

| Name | What it is |
|---|---|
| `dequantize(q)` | turn a quantized tensor back into floats |
| `derivative(f, x)` | numerical derivative of a function at a point |
| `dict(/, *args, **kwargs)` | make a dictionary |
| `dim(x)` | the physical dimension of a value (length, time …) |
| `divmod(x, y, /)` | Return the tuple (x//y, x%y). Invariant: div*y + mod == x. |
| `dot(a, b)` | dot product |

### E

| Name | What it is |
|---|---|
| `e` | constant |
| `e4m3` | value |
| `e5m2` | value |
| `einsum(spec, *operands)` | Einstein summation, e.g. einsum("bij,bjk->bik", A, B). Units multiply; differentiable. |
| `enumerate(xs, start=0)` | pairs (index, item) |
| `error(msg)` | stop the program with an error message |
| `exact(x)` | switch to exact rational arithmetic in a block or for a value |
| `exists(path)` | true when a file exists (needs read permission) |
| `exit(code=0)` | Stop the program with an exit status (0 = success). |
| `exp(x)` | e to the power x |
| `eye(n)` | identity matrix |

### F

| Name | What it is |
|---|---|
| `f16` | value |
| `f32` | value |
| `f64` | value |
| `filter(f, xs)` | items of a list for which a function returns true |
| `flatten(x)` | tensor as one dimension |
| `float(x)` | convert to a floating-point number |
| `floor(x)` | round down to an integer |
| `format(value, format_spec='', /)` | Return type(value).__format__(value, format_spec) |
| `format_info(fmt)` | details of a number format (f16, bf16, f32 …) |
| `fp16` | value |
| `fp32` | value |
| `fp4` | value |
| `fp4_e2m1` | value |
| `fp6` | value |
| `fp64` | value |
| `fp6_e2m3` | value |
| `fp6_e3m2` | value |
| `fp8` | value |
| `fp8_e4m3` | value |
| `fp8_e5m2` | value |
| `from_py(x)` | Python/JSON value -> Apsi value ({value, unit} dicts become Quantities). |
| `from_torch(t)` | turn a PyTorch tensor into an Apsi tensor |
| `fsum(x)` | Correctly rounded sum (no accumulated rounding error), for lists and tensors of any size. |
| `full(shape, value)` | tensor filled with one value |

### G

| Name | What it is |
|---|---|
| `gelu(x)` | function |
| `getattr(obj, name, *default)` | getattr(obj, "field"[, default]); names starting with _ are never reachable. |
| `getenv(name, default=None)` | Read an environment variable (needs --allow-env); the place for secrets like API keys. |
| `grad(f, wrt=0)` | gradient of a function (automatic differentiation) |

### H

| Name | What it is |
|---|---|
| `hasattr(obj, name)` | true when a value has a field or method |
| `hessian(f)` | hessian(f)(x) -> matrix of second derivatives of a scalar f. |
| `hex(number, /)` | Return the hexadecimal representation of an integer. |
| `http_get(url, timeout=30)` | download a URL (needs --allow-net) |
| `http_post_json(url, data, timeout=30)` | POST JSON to a URL (needs --allow-net) |

### I

| Name | What it is |
|---|---|
| `i32` | value |
| `i64` | value |
| `inf` | constant |
| `input(prompt='')` | Read one line from standard input (without the newline); returns none at end of input. |
| `int(x)` | convert to an integer |
| `int4` | value |
| `int8` | value |
| `iter(…)` | iter(iterable) -> iterator iter(callable, sentinel) -> iterator |

### J

| Name | What it is |
|---|---|
| `jacobian(f)` | jacobian(f)(x) -> d f_i / d x_j (f: Tensor[M] -> Tensor[M], x: Tensor[N]) as Tensor[M, N]. |
| `join(xs, sep='')` | join strings with a separator |
| `json_parse(text)` | parse JSON text into lists, dicts, numbers and strings |
| `json_str(value)` | turn a value into JSON text (indent, sort_keys …) |

### K

| Name | What it is |
|---|---|
| `keys(d)` | keys of a dict as a list |

### L

| Name | What it is |
|---|---|
| `laplace(f)` | laplace(f)(x) -> Σ ∂²f/∂x_i² (trace of the Hessian), e.g. for PDE residuals. |
| `len(x)` | number of items |
| `library_path(path)` | The actual file cload() would load (e.g. vecmath -> vecmath.dll on Windows). |
| `linspace(start, stop, n=50)` | n evenly spaced values from start to stop |
| `list(x=())` | make a list from any sequence |
| `load_csv(path, skip_header=0)` | read a CSV file into columns/rows |
| `load_json(path)` | read a JSON file |
| `log(x)` | natural logarithm |
| `log10(x)` | base-10 logarithm |
| `log2(x)` | base-2 logarithm |
| `log_softmax(x, axis=-1)` | log of softmax, numerically stable |

### M

| Name | What it is |
|---|---|
| `magnitude(x)` | the number without its unit (in SI base units) |
| `map(f, xs)` | apply a function to every item |
| `matmul(a, b)` | matrix product (also `a @ b`) |
| `max(*args, axis=None, keepdims=False)` | largest value (of a list/tensor, or of the arguments) |
| `mean(x, axis=None, keepdims=False)` | average |
| `measure(value, sigma)` | value ± sigma -> an independent measurement with standard uncertainty sigma. |
| `memo(fn)` | cache a function's results (@memo) |
| `meshgrid(x, y)` | coordinate grids from 1-D axes |
| `min(*args, axis=None, keepdims=False)` | smallest value |

### N

| Name | What it is |
|---|---|
| `nan` | constant |
| `nbytes(x)` | memory a tensor uses, in bytes |
| `ndim(x)` | number of tensor dimensions |
| `next(…)` | next(iterator[, default]) |
| `nominal(x)` | central value of an uncertain quantity |
| `norm(x, axis=None)` | Euclidean (or other) norm |

### O

| Name | What it is |
|---|---|
| `oct(number, /)` | Return the octal representation of an integer. |
| `ones(*shape, dtype=None)` | tensor of ones |
| `ones_like(x)` | ones with the shape of another tensor |
| `ord(character, /)` | Return the ordinal value of a character. |
| `outer(a, b)` | outer product |

### P

| Name | What it is |
|---|---|
| `param_count(tree)` | number of trainable numbers in a model |
| `permissions()` | the permissions the program runs with |
| `permutation(n, seed=None)` | random permutation of 0..n |
| `pi` | constant |
| `platform()` | "windows", "linux" or "macos". |
| `pmap(f, xs, workers=None)` | Parallel map on a thread pool (numpy/BLAS work releases the GIL). |
| `pow(base, exp, mod=None)` | Equivalent to base**exp with 2 arguments or base**exp % mod with 3 arguments |
| `precision(dtype)` | @precision(f32) / @precision(exact): numeric precision of a function. |
| `print(*args, sep=' ', end='\n')` | write values to the output, separated by spaces |
| `prod(x, axis=None)` | product of all values |
| `ptr_f32` | value |
| `ptr_f64` | value |
| `ptr_i32` | value |
| `ptr_i64` | value |
| `pyimport(name)` | use a Python module (pure ones like collections need no permission) |

### Q

| Name | What it is |
|---|---|
| `qmatmul(a, b)` | Matrix multiply with quantised operands: decode to FP32, multiply, accumulate in FP32 (the same numerics as low-precision tensor-core GEMMs). |
| `quant_error(x, fmt, axis=None)` | Relative RMS error introduced by storing x in `fmt`. |
| `quantize(x, fmt, scale='auto', axis=None)` | Quantise a tensor. scale="auto" maps max\|x\| to the format's largest value (per tensor, or per channel along `axis`); scale=1.0 keeps raw values (no scaling). |

### R

| Name | What it is |
|---|---|
| `rand(*shape, seed=None)` | uniform random numbers in [0, 1) |
| `randint(lo, hi, *shape, seed=None)` | random integers |
| `randn(*shape, seed=None)` | normally distributed random numbers |
| `range(a, b=None, step=1)` | integers start..stop with a step (like Python) |
| `read_text(path)` | read a text file |
| `record(**kw)` | a lightweight object with named fields |
| `relative_uncertainty(x)` | uncertainty divided by the value |
| `relu(x)` | max(x, 0) |
| `repr(x)` | text form of a value as code would write it |
| `reshape(x, *shp)` | same data, new shape |
| `reversed(xs)` | items in reverse order |
| `roll(x, shift, axis=None)` | shift values cyclically |
| `round(x, digits=0)` | round to n digits |
| `round_to(x, fmt, scale='auto', axis=None)` | Round x to what the format can represent and return it as a normal tensor ("fake quantisation"). Gradients pass straight through, so this works inside training. |

### S

| Name | What it is |
|---|---|
| `save_csv(path, data, header='')` | write rows to a CSV file |
| `save_json(path, value)` | write a value as a JSON file |
| `security_policy()` | the active security profile |
| `seed(n)` | make random numbers reproducible |
| `set(/, *args, **kwargs)` | a set: unique items, fast `in` |
| `shape(x)` | shape of a tensor |
| `sigma(x)` | standard uncertainty of a value |
| `sigmoid(x)` | 1 / (1 + e^-x) |
| `sin(x)` | sine (radians or an angle quantity) |
| `sinh(x)` | hyperbolic sine |
| `softmax(x, axis=-1)` | normalised exponentials (probabilities) |
| `softplus(x)` | log(1 + e^x) |
| `sort(x)` | sorted copy of a tensor |
| `sorted(xs, key=None, reverse=False)` | sorted copy of a list (key=, reverse=) |
| `sqrt(x)` | square root (units become their square root) |
| `stack(items, axis=0)` | stack tensors along a new axis |
| `std(x, axis=None)` | standard deviation |
| `stop_gradient(x)` | use a value without differentiating through it |
| `str(x)` | text form of a value |
| `sum(x, axis=None, keepdims=False)` | sum of all values (or along an axis) |

### T

| Name | What it is |
|---|---|
| `tan(x)` | tangent |
| `tanh(x)` | hyperbolic tangent |
| `tau` | constant |
| `tensor(x, dtype=None)` | make a tensor from numbers or nested lists (dtype=…) |
| `testing()` | true while `apsi test` runs |
| `timed(fn)` | @timed: print how long each call takes. |
| `to_float(x)` | plain float from a quantity or 1-element tensor |
| `to_py(x)` | Apsi value -> plain Python/JSON-friendly value (units become {value, unit}). |
| `to_torch(x, device=None)` | turn an Apsi tensor into a PyTorch tensor (needs --allow-python) |
| `transpose(x, axes=None)` | swap tensor axes |
| `tree_map(f, *trees)` | apply a function to every leaf of nested lists/dicts/records |
| `tuple(/, *args, **kwargs)` | make a tuple |
| `type(v)` | type name of a value ("Int", "Tensor" …) |

### U

| Name | What it is |
|---|---|
| `u` | `u.km`, `u.eV` ... -> one of that unit as a Quantity. |
| `uint8` | value |
| `unique(xs)` | distinct values |
| `unit(x)` | the unit of a quantity |

### V

| Name | What it is |
|---|---|
| `value_and_grad(f, wrt=0)` | value and gradient of a function (automatic differentiation) |
| `var(x, axis=None)` | variance |
| `void` | value |

### W

| Name | What it is |
|---|---|
| `where(cond, a, b)` | choose element-wise: where(cond, a, b) |
| `write_text(path, text)` | write a text file (needs write permission) |

### Z

| Name | What it is |
|---|---|
| `zeros(*shape, dtype=None)` | tensor of zeros |
| `zeros_like(x)` | zeros with the shape of another tensor |
| `zip(*xs)` | pairs items of several lists |

## Standard modules (22)

Load a module with `use NAME` (all its names) or `use NAME as n` (then `n.member`).

### `use crypto` — 11 names

| Name | What it is |
|---|---|
| `b64decode(text)` | function |
| `b64encode(text)` | function |
| `hash(data, algo='sha256')` | function |
| `hmac_sign(key, message, algo='sha256')` | function |
| `hmac_verify(key, message, signature, algo='sha256')` | function |
| `password_hash(password, n=16384, r=8, p=1)` | Salted password hash to store in a database. |
| `password_verify(password, stored)` | function |
| `random_hex(nbytes=32)` | function |
| `random_token(nbytes=32)` | function |
| `secure_equal(a, b)` | Constant-time comparison (prevents timing attacks). |
| `secure_randint(lo, hi)` | Uniform integer in [lo, hi) from the OS CSPRNG. |

### `use data` — 28 names

| Name | What it is |
|---|---|
| `batch(p, size, drop_last=False)` | function |
| `cache(p, directory=None, shard=4096)` | Cache a pipeline. directory=none -> RAM; otherwise memory-mapped shards on disk. |
| `disk_cache(directory='cache')` | @disk_cache("cache"): memoise a function's results on disk across runs. |
| `filter(p, f)` | function |
| `flush(x)` | function |
| `from_csv(path, skip_header=0, chunk=10000, delimiter=',')` | function |
| `from_files(pattern)` | function |
| `from_generator(f)` | f() is called at the start of every epoch and must return an iterable. |
| `from_jsonl(path)` | function |
| `from_list(xs)` | function |
| `from_npy(path)` | function |
| `from_parquet(path, columns=None, rows=65536)` | Stream a Parquet file in record batches: each item is a dict of column -> tensor. |
| `from_range(n)` | function |
| `from_tensors(*arrays, batch=None, shuffle=False, seed=None, drop_last=False)` | Rows of one or more equal-length tensors: from_tensors(X, y) yields (x_i, y_i). |
| `interleave(pipelines, cycle=True)` | Alternate items from several pipelines (round robin) until all are exhausted. |
| `load_tensor(path, mmap=False)` | function |
| `map(p, f, workers=None, ordered=True)` | Apply f to every item; workers > 1 runs f in parallel (numpy work releases the GIL). |
| `open_tensor(path, shape, dtype='f32', mode='w+')` | A memory-mapped tensor on disk (SSD): larger than RAM, written/read lazily. |
| `prefetch(p, n=2)` | Produce up to n items ahead on a background thread (overlaps loading and compute). |
| `repeat(p, times=None)` | function |
| `save_tensor(path, x)` | function |
| `shard(p, count, index)` | Every `count`-th item starting at `index` — split one dataset across workers or machines. |
| `shuffle(p, buffer=1024, seed=None)` | Streaming shuffle with a bounded buffer (a full shuffle when buffer >= dataset size). |
| `skip(p, n)` | function |
| `take(p, n)` | function |
| `throughput(p, limit=None)` | Measure items per second of a pipeline (optionally only the first `limit` items). |
| `unbatch(p)` | function |
| `window(p, size, shift=1, drop_last=True)` | Sliding windows over a stream: each item is a tensor of `size` consecutive items (time series). |

### `use ffi` — 6 names

| Name | What it is |
|---|---|
| `cload(path)` | function |
| `ptr_f32` | value |
| `ptr_f64` | value |
| `ptr_i32` | value |
| `ptr_i64` | value |
| `void` | value |

### `use gpu` — 8 names

| Name | What it is |
|---|---|
| `cpu_threads()` | function |
| `device()` | "cuda:0" when a GPU meeting Apsi's target tier is usable through CuPy, otherwise "cpu". |
| `gpu_available()` | function |
| `gpu_info()` | function |
| `hardware()` | function |
| `to_cpu(x)` | function |
| `to_device(x, dev=None)` | function |
| `to_gpu(x)` | function |

### `use io` — 9 names

| Name | What it is |
|---|---|
| `exists(path)` | function |
| `json_parse(text)` | function |
| `json_str(value)` | function |
| `load_csv(path, skip_header=0)` | function |
| `load_json(path)` | function |
| `read_text(path)` | function |
| `save_csv(path, data, header='')` | function |
| `save_json(path, value)` | function |
| `write_text(path, text)` | function |

### `use layers` — 67 names

| Name | What it is |
|---|---|
| `activation(fn)` | function |
| `add()` | function |
| `autoencoder(d_in, latent, hidden=(64,), act='relu')` | Returns a DAG with outputs [reconstruction, code]. |
| `avg_pool(k=2)` | type |
| `batch_norm_layer(d, eps=1e-05)` | Normalises over the batch (and spatial) axes with the current batch's statistics. |
| `bilinear(n1, n2, n_out)` | type |
| `check_torch(*a, **k)` | Compare every layer type on NumPy and PyTorch (outputs and gradients, float64). |
| `cnn(in_channels, classes, image=28, channels=(16, 32), dense_width=64)` | Small convolutional classifier for images shaped [N, C, H, W]. |
| `concatenate()` | function |
| `conv2d_layer(c_in, c_out, k=3, stride=1, padding='same')` | type |
| `count_params(params)` | function |
| `dag(nodes, inputs=('x',), output=None)` | function |
| `dense(n_in, n_out, act=None)` | function |
| `devices()` | The compute devices available: NumPy, and PyTorch CPU / CUDA / ROCm / MPS / XPU when installed. |
| `dropout_layer(p=0.1)` | type |
| `embedding(vocab, d)` | type |
| `fit(model, params, X, y=None, backend='numpy', **kw)` | Train `model`. Returns record(params, history, val_history, device). |
| `flatten_layer(/, *args, **kwargs)` | type |
| `gat(n_in, n_out, heads=4, concat=True, slope=0.2)` | Graph attention (Veličković et al.) with several heads; heads are concatenated (or averaged). |
| `gat_net(n_in, hidden, classes, heads=4, dropout=0.3)` | function |
| `gather(x, ids)` | Rows of x at ids (differentiable). |
| `gcn(n_in, n_out, bias=True)` | Graph convolution (Kipf & Welling): h_i = Σ_j x_j W / sqrt(deg_i deg_j). |
| `gcn_net(n_in, hidden, classes, layers=2, dropout=0.5)` | function |
| `gin(n_in, n_out, hidden=None)` | Graph isomorphism network layer: MLP((1 + eps) x_i + Σ_j x_j). |
| `gin_net(n_in, hidden, classes, layers=3, pool='sum')` | Graph-level classifier: GIN layers, pooling per graph, linear head. |
| `global_avg_pool(/, *args, **kwargs)` | type |
| `graph(edges=None, n=None, src=None, dst=None, self_loops=True, undirected=False, batch=None)` | graph(edges=[[0, 1], [1, 2]], n=3) or graph(src=tensor, dst=tensor, n=…). `batch` assigns nodes to separate graphs for graph-level tasks (used by graph_pool). |
| `graph_pool(how='mean')` | Node features -> one vector per graph (mean, sum or max over each graph's nodes). |
| `gru_layer(n_in, n_hidden, return_sequences=True)` | type |
| `identity(/, *args, **kwargs)` | type |
| `lambda_layer(fn, name='lambda', torch_fn=None)` | type |
| `last_step(/, *args, **kwargs)` | type |
| `layer_norm_layer(d, eps=1e-05)` | type |
| `linear(n_in, n_out, bias=True, init='he')` | function |
| `lstm_classifier(n_in, hidden, classes, layers=1)` | function |
| `lstm_layer(n_in, n_hidden, return_sequences=True)` | type |
| `max_pool(k=2)` | type |
| `mean_pool(axis=1)` | Mean over one axis, e.g. over time: [B, T, D] -> [B, D]. |
| `mlp(sizes, act='relu', dropout=0.0, norm=False, out_act=None)` | Multilayer perceptron, e.g. mlp([2, 64, 64, 3]). |
| `multi_head_attention(d, heads=4, causal=False)` | type |
| `multiply()` | function |
| `parallel(branches, combine='concat')` | function |
| `params_from_torch(*a, **k)` | Parameters of a module made by torch_module (after training it in Python) as an Apsi parameter tree. |
| `positional_encoding(max_len, d, learned=False)` | type |
| `predict(model, params, X, backend='numpy', **kw)` | function |
| `reshape_layer(*shape)` | type |
| `residual(inner, pre_norm=None)` | function |
| `resnet_mlp(d_in, d_out, width=64, blocks=3, act='gelu')` | Deep MLP with residual blocks and layer norm — trains stably at any depth. |
| `rms_norm_layer(d, eps=1e-06)` | type |
| `sage(n_in, n_out, aggregate='mean')` | GraphSAGE: h_i = x_i W_self + mean_{j→i} x_j W_neigh. |
| `scale(d, shift=True)` | A learnable per-feature scale (and shift). |
| `segment_count(ids, n)` | function |
| `segment_max(data, ids, n)` | Per-segment maximum (no gradient through the choice of maximum). |
| `segment_mean(data, ids, n)` | function |
| `segment_softmax(scores, ids, n)` | Softmax of `scores` within each segment (e.g. over the incoming edges of every node). |
| `segment_sum(data, ids, n)` | out[k] = sum of data[i] where ids[i] == k (k < n). Differentiable in `data`. |
| `select(i)` | Pick one element of a list/tuple input (e.g. the output of parallel). |
| `sequential(layers)` | function |
| `siamese(encoder, combine='concat', head=None)` | Shared encoder applied to [a, b]; the outputs are combined (concat/add/mul/list) and optionally fed to head. |
| `summary(model, params=None)` | A table of the model's layers and parameter counts (printed, and returned as a string). |
| `torch_module(*a, **k)` | The model with its parameters as a torch.nn.Module: module(x) / module(x, graph=g). |
| `transformer_block(d, heads=4, ff=None, causal=False, dropout=0.0, activation='gelu')` | Pre-norm transformer block: x + attention(norm(x)), then x + MLP(norm(x)). |
| `transformer_classifier(vocab, classes, d=64, heads=4, layers=2, context=128)` | Encoder for sequences of token ids [B, T] -> class logits [B, classes] (mean pooling). |
| `transformer_encoder(d, heads=4, layers=2, ff=None, causal=False, dropout=0.0)` | function |
| `transformer_lm(vocab, d=64, heads=4, layers=2, context=128, dropout=0.0)` | Decoder-only (GPT-style) language model: token ids [B, T] -> logits [B, T, vocab]. |
| `vae(d_in, latent, hidden=(64,), act='relu')` | Variational autoencoder. Returns [reconstruction, mu, logvar]; train with vae_loss. |
| `vae_loss(outputs, x, beta=1.0)` | function |

### `use linalg` — 8 names

| Name | What it is |
|---|---|
| `det(a)` | function |
| `eig(a)` | function |
| `eigh(a)` | function |
| `inv(a)` | function |
| `norm(x, axis=None)` | function |
| `solve(a, b)` | function |
| `svd(a)` | function |
| `trace(a)` | function |

### `use math` — 25 names

| Name | What it is |
|---|---|
| `abs(x)` | function |
| `acos(x)` | function |
| `asin(x)` | function |
| `atan(x)` | function |
| `atan2(y, x)` | function |
| `cbrt(x)` | function |
| `ceil(x)` | function |
| `cos(x)` | function |
| `cosh(x)` | function |
| `e` | constant |
| `exp(x)` | function |
| `floor(x)` | function |
| `inf` | constant |
| `log(x)` | function |
| `log10(x)` | function |
| `log2(x)` | function |
| `nan` | constant |
| `pi` | constant |
| `round(x, digits=0)` | function |
| `sin(x)` | function |
| `sinh(x)` | function |
| `sqrt(x)` | function |
| `tan(x)` | function |
| `tanh(x)` | function |
| `tau` | constant |

### `use models` — 10 names

| Name | What it is |
|---|---|
| `dequantize_model(params)` | function |
| `hf_pipeline(task, model=None, device=None, **kw)` | A Hugging Face transformers pipeline: classify = hf_pipeline("sentiment-analysis"). |
| `load_safetensors(path, dtype=None, names=None, nested=True)` | function |
| `model_size(params)` | Bytes used by all tensors in a parameter tree (counts compact formats at their real size). |
| `onnx_model(path, providers=None)` | Run an ONNX model with onnxruntime: m = onnx_model("net.onnx"); m({"input": x}). |
| `qlinear(x, W, b=None)` | x @ W (+ b) where W may be a QTensor — decodes, multiplies and accumulates in FP32. |
| `quantize_model(params, dtype=None, min_size=1024)` | Store every weight matrix (>= min_size elements) in a compact format; others stay as-is. |
| `safetensors_info(path)` | function |
| `save_safetensors(path, params, metadata=None)` | function |
| `torch_load(path, map_location='cpu')` | function |

### `use net` — 2 names

| Name | What it is |
|---|---|
| `http_get(url, timeout=30)` | function |
| `http_post_json(url, data, timeout=30)` | function |

### `use nn` — 51 names

| Name | What it is |
|---|---|
| `accuracy(logits, labels)` | function |
| `adagrad(lr=0.01, eps=1e-10)` | function |
| `adam(lr=0.001, b1=0.9, b2=0.999, eps=1e-08)` | function |
| `adamw(lr=0.001, b1=0.9, b2=0.999, eps=1e-08, weight_decay=0.01)` | function |
| `attention(q, k, v, mask=None)` | Scaled dot-product attention over the last two axes. |
| `batch_norm(x, gamma=None, beta=None, eps=1e-05, axis=0)` | Batch normalisation over `axis` (training-mode statistics). |
| `batches(X, y, size, seed=None)` | function |
| `causal_mask(T)` | function |
| `clip_grad_norm(grads, max_norm=1.0)` | function |
| `conv2d(x, w, b=None, stride=1, padding=0)` | function |
| `conv_init(c_in, c_out, k=3, seed=None)` | function |
| `cosine_lr(step, total, base=0.001, warmup=0, minimum=0.0)` | function |
| `cross_entropy(logits, labels)` | function |
| `dense(n_in, n_out, seed=None, scale=None)` | A dense layer's parameters: record(W, b) with He initialisation. |
| `dropout(x, p=0.1, train=True, seed=None)` | function |
| `embedding_init(vocab, d, seed=None)` | function |
| `gelu(x)` | function |
| `global_norm(grads)` | function |
| `gru(p, xs)` | function |
| `gru_cell(p, x, h)` | function |
| `gru_init(n_in, n_hidden, seed=None)` | function |
| `layer_norm(x, gamma=None, beta=None, eps=1e-05)` | function |
| `linear_warmup(step, base=0.001, warmup=100)` | function |
| `load_params(path, like)` | function |
| `log_softmax(x, axis=-1)` | function |
| `lstm(p, xs)` | Run an LSTM over a sequence xs: [B, T, n_in] -> outputs [B, T, n_hidden]. |
| `lstm_cell(p, x, h, c)` | One LSTM step: returns (h, c). x: [B, n_in], h, c: [B, n_hidden]. |
| `lstm_init(n_in, n_hidden, seed=None)` | LSTM parameters: one fused weight for the 4 gates (input, forget, cell, output). |
| `make_blobs(n=300, centers=3, dim=2, spread=0.5, seed=0)` | function |
| `make_moons(n=200, noise=0.1, seed=0)` | function |
| `make_regression(n=100, features=3, noise=0.1, seed=0)` | function |
| `max_pool2d(x, k=2)` | function |
| `mlp_init(sizes, seed=0)` | function |
| `mse(pred, target)` | function |
| `norm_init(d)` | function |
| `numeric_grad(f, x, eps=1e-06)` | Finite-difference gradient (for checking autodiff). |
| `one_hot(labels, k)` | function |
| `relu(x)` | function |
| `rms_norm(x, gamma=None, eps=1e-06)` | function |
| `rmsprop(lr=0.001, decay=0.9, eps=1e-08)` | function |
| `save_params(path, params)` | function |
| `set_lr(opt, lr)` | function |
| `sgd(lr=0.01, momentum=0.0)` | function |
| `sigmoid(x)` | function |
| `softmax(x, axis=-1)` | function |
| `spiral_data(n_per_class=100, classes=3, noise=0.2, seed=0)` | Classic spiral toy dataset -> (X: Tensor[N, 2], y: Tensor[N] of class ids). |
| `standardize(X, axis=0)` | Return (X_scaled, mean, std) so new data can be scaled the same way. |
| `step_lr(step, base=0.001, every=100, factor=0.5)` | function |
| `tanh(x)` | function |
| `train(loss_fn, params, data, opt, epochs=10, batch_size=None, seed=0, verbose=True, clip=None, schedule=None, …)` | Generic training loop. |
| `train_test_split(X, y, test=0.2, seed=0)` | function |

### `use parallel` — 12 names

| Name | What it is |
|---|---|
| `atomic(value=0)` | function |
| `bench(f, repeat=5, warmup=1)` | Time f() -> record(best, mean, runs) in seconds (as Time quantities). |
| `channel(capacity=0)` | function |
| `cpu_count()` | function |
| `lock()` | function |
| `parallel_for(items, f, workers=None)` | Call f(item) for every item on a thread pool; returns when all are done. |
| `parallel_reduce(items, f, combine, init, workers=None)` | Map f over items in parallel, then fold the results with combine(acc, x). |
| `set_threads(n)` | Limit compute threads (BLAS and Apsi's worker pool). Returns the active setting. |
| `sleep(t)` | function |
| `spawn(f, *args, **kwargs)` | Run f(*args) concurrently; returns a Task (use task.wait() for the result). |
| `wait_all(tasks, timeout=None)` | function |
| `wait_any(tasks, timeout=None)` | Return (index, result) of the first task to finish. |

### `use physics` — 24 names

| Name | What it is |
|---|---|
| `a_0` | constant |
| `alpha_fs` | constant |
| `AU` | constant |
| `c` | constant |
| `epsilon_0` | constant |
| `G` | constant |
| `g0` | constant |
| `gradient(u, dx, axis=0)` | Central-difference derivative of a field along one axis (periodic). |
| `h` | constant |
| `hbar` | constant |
| `k_B` | constant |
| `laplacian(u, dx, boundary='periodic')` | Discrete Laplacian (∇²u) on a regular grid with spacing dx. |
| `m_e` | constant |
| `M_earth` | constant |
| `m_n` | constant |
| `m_p` | constant |
| `M_sun` | constant |
| `meshgrid(x, y)` | function |
| `mu_0` | constant |
| `N_A` | constant |
| `q_e` | constant |
| `R_earth` | constant |
| `R_gas` | constant |
| `sigma_SB` | constant |

### `use physics.constants` — 21 names

| Name | What it is |
|---|---|
| `a_0` | constant |
| `alpha_fs` | constant |
| `AU` | constant |
| `c` | constant |
| `epsilon_0` | constant |
| `G` | constant |
| `g0` | constant |
| `h` | constant |
| `hbar` | constant |
| `k_B` | constant |
| `m_e` | constant |
| `M_earth` | constant |
| `m_n` | constant |
| `m_p` | constant |
| `M_sun` | constant |
| `mu_0` | constant |
| `N_A` | constant |
| `q_e` | constant |
| `R_earth` | constant |
| `R_gas` | constant |
| `sigma_SB` | constant |

### `use physics.fields` — 3 names

| Name | What it is |
|---|---|
| `gradient(u, dx, axis=0)` | Central-difference derivative of a field along one axis (periodic). |
| `laplacian(u, dx, boundary='periodic')` | Discrete Laplacian (∇²u) on a regular grid with spacing dx. |
| `meshgrid(x, y)` | function |

### `use physics.units` — 0 names

| Name | What it is |
|---|---|


### `use precision` — 28 names

| Name | What it is |
|---|---|
| `bf16` | value |
| `bf8` | value |
| `dequantize(q)` | function |
| `e4m3` | value |
| `e5m2` | value |
| `f16` | value |
| `f32` | value |
| `f64` | value |
| `format_info(fmt)` | function |
| `fp16` | value |
| `fp32` | value |
| `fp4` | value |
| `fp4_e2m1` | value |
| `fp6` | value |
| `fp64` | value |
| `fp6_e2m3` | value |
| `fp6_e3m2` | value |
| `fp8` | value |
| `fp8_e4m3` | value |
| `fp8_e5m2` | value |
| `int4` | value |
| `int8` | value |
| `nbytes(x)` | function |
| `qmatmul(a, b)` | Matrix multiply with quantised operands: decode to FP32, multiply, accumulate in FP32 (the same numerics as low-precision tensor-core GEMMs). |
| `quant_error(x, fmt, axis=None)` | Relative RMS error introduced by storing x in `fmt`. |
| `quantize(x, fmt, scale='auto', axis=None)` | Quantise a tensor. scale="auto" maps max\|x\| to the format's largest value (per tensor, or per channel along `axis`); scale=1.0 keeps raw values (no scaling). |
| `round_to(x, fmt, scale='auto', axis=None)` | Round x to what the format can represent and return it as a normal tensor ("fake quantisation"). Gradients pass straight through, so this works inside training. |
| `uint8` | value |

### `use python` — 5 names

| Name | What it is |
|---|---|
| `from_py(x)` | Python/JSON value -> Apsi value ({value, unit} dicts become Quantities). |
| `from_torch(t)` | function |
| `pyimport(name)` | function |
| `to_py(x)` | Apsi value -> plain Python/JSON-friendly value (units become {value, unit}). |
| `to_torch(x, device=None)` | function |

### `use random` — 5 names

| Name | What it is |
|---|---|
| `permutation(n, seed=None)` | function |
| `rand(*shape, seed=None)` | function |
| `randint(lo, hi, *shape, seed=None)` | function |
| `randn(*shape, seed=None)` | function |
| `seed(n)` | function |

### `use sci` — 20 names

| Name | What it is |
|---|---|
| `bisect(f, a, b, tol=1e-12, max_iter=200)` | function |
| `conj(x)` | function |
| `cumtrapz(y, x)` | function |
| `curve_fit(model, x, y, p0, sigma=None, max_iter=200, tol=1e-12)` | Levenberg–Marquardt least squares with an exact autodiff Jacobian. |
| `einsum(spec, *operands)` | Einstein summation, e.g. einsum("bij,bjk->bik", A, B). Units multiply; differentiable. |
| `fft(x, axis=-1)` | function |
| `fftfreq(n, d=1.0)` | function |
| `find_root(f, x0, tol=1e-12, max_iter=100)` | Newton's method with the exact derivative from autodiff (unit-aware). |
| `ifft(x, axis=-1)` | function |
| `imag(x)` | function |
| `integrate(f, a, b, points=64)` | ∫_a^b f(x) dx by Gauss–Legendre quadrature. f receives a tensor of points. Exact for polynomials up to degree 2·points−1; units are [f]·[x]. |
| `interp(x, xp, fp)` | Piecewise-linear interpolation (units of xp and x must match). |
| `irfft(x, n=None, axis=-1)` | function |
| `minimize(f, x0, method='lbfgs', steps=500, tol=1e-10, lr=0.01, history=10, verbose=False)` | Minimise a scalar function of a parameter tree using exact gradients. |
| `power_spectrum(x, d=1.0)` | One-sided power spectrum \|rfft\|² with matching frequencies -> (freqs, power). |
| `real(x)` | function |
| `rfft(x, axis=-1)` | function |
| `rfftfreq(n, d=1.0)` | function |
| `solve_ivp(f, t_span, y0, method='bdf', rtol=1e-06, atol=1e-09, t_eval=None, first_step=None, max_steps=200000)` | Solve y' = f(t, y) from t_span[0] to t_span[1]. |
| `trapz(y, x=None, dx=None)` | Trapezoidal integral of samples y over x (or uniform spacing dx). Units: [y]·[x]. |

### `use sys` — 8 names

| Name | What it is |
|---|---|
| `cpu_threads()` | function |
| `device()` | "cuda:0" when a GPU meeting Apsi's target tier is usable through CuPy, otherwise "cpu". |
| `gpu_available()` | function |
| `gpu_info()` | function |
| `hardware()` | function |
| `to_cpu(x)` | function |
| `to_device(x, dev=None)` | function |
| `to_gpu(x)` | function |

### `use web` — 9 names

| Name | What it is |
|---|---|
| `abort(status, message='error')` | Abort the current request with an HTTP error: abort(404, "no such user"). |
| `bearer_token(req)` | function |
| `jwt_sign(claims, secret, expires_in=3600)` | Create an HS256 JSON Web Token. expires_in: seconds (number or Time quantity). |
| `jwt_verify(token, secret, leeway=30)` | Return the claims of a valid token, or none if the signature/expiry/algorithm is wrong. |
| `redirect(location, status=302)` | function |
| `require_api_key(keys, header='x-api-key', exempt=())` | A before-hook accepting any of the given API keys (constant-time comparison). |
| `require_jwt(secret, exempt=())` | A before-hook: rejects requests without a valid Bearer JWT; sets req.user to its claims. |
| `response(body='', status=200, headers=None, type=None)` | Build a response explicitly: response("created", status=201, headers={"X-A": "1"}). |
| `server(name='apsi', max_body=1048576, workers=None, max_connections=10000, log=True, handler_timeout=60.0, head…)` | function |

## Command line (38 commands)

Every command takes `--help` (`apsi help run` works too).

| Command | What it does |
|---|---|
| `apsi run` | run a .apsi program (default: the project's main file) |
| `apsi check` | statically check files or folders without running them |
| `apsi test` | run test blocks in files or folders |
| `apsi repl` | interactive prompt |
| `apsi fmt` | format .apsi files in place (safe: syntax tree must stay identical) |
| `apsi audit` | list the capabilities a program uses and the minimal flags it needs |
| `apsi scan` | scan programs, folders and companion files for malware patterns |
| `apsi verify` | check the installed runtime against its release manifest |
| `apsi config` | show or change settings (security profile) |
| `apsi lsp` | run the language server (stdio) for editors |
| `apsi doctor` | check this machine against Apsi's hardware targets |
| `apsi serve` | expose a program's functions as a JSON HTTP API |
| `apsi ast` | print the syntax tree of a file |
| `apsi tokens` | print the tokens of a file |
| `apsi pkg` | install, list, remove and lock Python libraries (GPU, ONNX, PyTorch, …) |
| `apsi install` | install Apsi libraries (.apsil) from the registry or a file |
| `apsi uninstall` | remove installed Apsi libraries |
| `apsi pack` | check a library and build dist/NAME-VERSION.apsil |
| `apsi publish` | upload an .apsil library to the registry |
| `apsi search` | search the library registry |
| `apsi lib` | libraries: list, info, registry, login, logout |
| `apsi yank` | hide a version of your library from new installs (--undo to show it again) |
| `apsi unpublish` | delete a version of your library from the registry |
| `apsi release` | upload, list or delete Apsi release files and benchmark results on the website |
| `apsi editors` | install Apsi support into VS Code, Cursor, Windsurf, Sublime, Vim/Neovim, … |
| `apsi convert` | translate Python files or folders into Apsi (numpy, math, torch, … mapped) |
| `apsi version` | print the version |
| `apsi new` | create a new project folder (apsi new NAME --template app\|lib\|physics\|ai\|server) |
| `apsi init` | turn the current (or given) folder into an Apsi project |
| `apsi task` | list or run the [tasks] of apsi.toml |
| `apsi eval` | evaluate code from the command line and print the result |
| `apsi watch` | re-run a program, tests or the checker whenever files change |
| `apsi bench` | time a program over several runs (output suppressed) |
| `apsi doc` | documentation for built-ins and modules (apsi doc nn, apsi doc sqrt) |
| `apsi completion` | print a shell completion script (bash, zsh, fish, powershell) |
| `apsi help` | show help for a command (apsi help run) |
| `apsi build` | make an executable from a program (app.exe on Windows) |
| `apsi compile` | parse, check and compile programs now so later runs start from the cache |
