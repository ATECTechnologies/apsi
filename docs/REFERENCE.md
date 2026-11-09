# Apsi 2.4.4 — everything on one page

This page is the whole language in one place: every piece of syntax with a short example, then **every built-in
function, every standard module and every member, and every command** of Apsi 2.4.4, generated from the runtime
itself so nothing is missing. Read it top to bottom once, then use it with your browser's search (Ctrl+F).
The long-form explanations live in the [language reference](/docs/2.4.4/language).

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
| `[1, 2]`, `(1, 2)`, `{"k": 1}`, `set([1, 2])` | `List`, `Tuple`, `Dict`, `Set` | Python methods: `.append .pop .get .items .keys …` |
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
parallel for i in 0..n:                 # iterations on all CPU cores (native)
    out[i] = work(i)
assert total > 0, "total must be positive"
let squares = [x * x for x in 0..10 if x % 2 == 0]
let pairs = [(i, j) for i in 0..3 for j in 0..3 if i != j]
```

## Errors

```apsi
try:
    let cfg = load_json("config.json")
except FileNotFoundError, PermissionError:
    print("using defaults")
except ValueError as e:                 # e.kind, e.message, e.line
    print("bad config:", e.message)
else:
    print("loaded")
finally:
    print("done")
raise ValueError("speed must be positive")
raise "plain message"                   # a RuntimeError
```

Kinds use Python's names (`ValueError`, `KeyError`, `IndexError`, `ZeroDivisionError`, `TypeError`, `OSError`,
`FileNotFoundError` …) plus `UnitError` and `ShapeError`; errors Apsi raises itself match them too.

## Generators

```apsi
fn naturals():
    var n = 0
    while true:
        yield n                         # values are produced one at a time; endless is fine
        n += 1
```

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

```apsi
fn total(*xs, **opts) = sum(xs) * opts.get("scale", 1)   # any number of arguments
total(1, 2, 3)  total(*nums, **options)                  # spread a list / dict into a call
```

Decorators: `@memo` (cache results), `@native` (compile to machine code), `@native("fast")`, `@precision(f32)` / `@precision(exact)`.

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

Methods named `add sub mul div eq lt le gt ge neg len get set contains iter to_str call hash bool` make `+ - * /
== < … len() [] in for print()` work on your structs (Python's `__add__`, `__str__` … under friendlier names).

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
or `--allow-all`. `apsi audit app.apsi` prints exactly the flags a program needs. Pure standard modules (`pyimport("collections")`, heapq, itertools, re, datetime …) need no flag.


## Built-in functions and values (298)

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
| `argsort(xs, reverse=False)` | function |
| `ArithmeticError(/, *args, **kwargs)` | Base class for arithmetic errors. |
| `ascii(obj, /)` | Return an ASCII-only representation of an object. |
| `asin(x)` | inverse sine, radians |
| `AssertionError(/, *args, **kwargs)` | Assertion failed. |
| `atan(x)` | inverse tangent, radians |
| `atan2(y, x)` | angle of the point (x, y), radians |
| `AttributeError(/, *args, **kwargs)` | Attribute not found. |

### B

| Name | What it is |
|---|---|
| `band(a, b)` | `a & b` of Python: bits, set intersection, element-wise and |
| `bf16` | value |
| `bf8` | value |
| `bin(number, /)` | Return the binary representation of an integer. |
| `bnot(a)` | `~a` |
| `bool(/, *args, **kwargs)` | true/false of a value |
| `bor(a, b)` | `a | b` of Python: bits, set union, dict merge, element-wise or |
| `bxor(a, b)` | `a ^ b` of Python |
| `bytearray(/, *args, **kwargs)` | bytearray(iterable_of_ints) -> bytearray bytearray(string, encoding[, errors]) -> bytearray bytearray(bytes_or_buffer) -> mutable copy of bytes_or_buffer bytearray(int) -> bytes array of size given by the parameter initi |
| `bytes(/, *args, **kwargs)` | bytes(iterable_of_ints) -> bytes bytes(string, encoding[, errors]) -> bytes bytes(bytes_or_buffer) -> immutable copy of bytes_or_buffer bytes(int) -> bytes object of size given by the parameter initialized with null byte |

### C

| Name | What it is |
|---|---|
| `callable(obj, /)` | Return whether the object is callable (i.e., some kind of function). |
| `cbrt(x)` | cube root |
| `ceil(x)` | round up to an integer |
| `chain_lists(xss)` | join a list of lists into one list |
| `chars(s)` | the characters of a string as a list |
| `chr(i, /)` | Return a Unicode string of one character with ordinal i; 0 <= i <= 0x10ffff. |
| `chunks(xs, n)` | function |
| `clamp(x, lo, hi)` | function |
| `classmethod(/, *args, **kwargs)` | Convert a function to be a class method. |
| `clip(x, lo, hi)` | limit values to [lo, hi] |
| `cload(path)` | load a C library (needs --allow-ffi) |
| `clock()` | time since the program started, in seconds (a Quantity) |
| `combinations(xs, k)` | function |
| `complex(/, *args, **kwargs)` | Create a complex number from a string or numbers. |
| `concat(items, axis=0)` | join tensors along an axis |
| `ConnectionError(/, *args, **kwargs)` | Connection error. |
| `cos(x)` | cosine (radians or an angle quantity) |
| `cosh(x)` | hyperbolic cosine |
| `count_by(xs, key=None)` | function |
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
| `dir(obj=None)` | function |
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
| `EnvironmentError(/, *args, **kwargs)` | Base class for I/O related errors. |
| `EOFError(/, *args, **kwargs)` | Read beyond end of file. |
| `erf(x)` | error function |
| `error(msg)` | stop the program with an error message |
| `Error(/, *args, **kwargs)` | An error raised from Apsi code with `raise "message"` or `raise Error("message")`. |
| `exact(x)` | switch to exact rational arithmetic in a block or for a value |
| `Exception(/, *args, **kwargs)` | An error raised from Apsi code with `raise "message"` or `raise Error("message")`. |
| `exists(path)` | true when a file exists (needs read permission) |
| `exit(code=0)` | Stop the program with an exit status (0 = success). |
| `exp(x)` | e to the power x |
| `expm1(x)` | function |
| `eye(n)` | identity matrix |

### F

| Name | What it is |
|---|---|
| `f16` | value |
| `f32` | value |
| `f64` | value |
| `FileExistsError(/, *args, **kwargs)` | File already exists. |
| `FileNotFoundError(/, *args, **kwargs)` | File not found. |
| `filter(f, xs)` | items of a list for which a function returns true |
| `flatten(x)` | tensor as one dimension |
| `float(x)` | convert to a floating-point number |
| `FloatingPointError(/, *args, **kwargs)` | Floating-point operation failed. |
| `floor(x)` | round down to an integer |
| `flush_out()` | function |
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
| `frozenset(/, *args, **kwargs)` | an unchangeable set |
| `fsum(x)` | Correctly rounded sum (no accumulated rounding error), for lists and tensors of any size. |
| `full(shape, value)` | tensor filled with one value |

### G

| Name | What it is |
|---|---|
| `gelu(x, approximate='tanh')` | GELU. approximate="tanh" (default, fast) or "none" (exact: x·Φ(x), as PyTorch's default). |
| `getattr(obj, name, *default)` | getattr(obj, "field"[, default]); names starting with _ are never reachable. |
| `getenv(name, default=None)` | Read an environment variable (needs --allow-env); the place for secrets like API keys. |
| `gpu(*args, **kwargs)` | run work on the GPU when one is available |
| `gpu_info()` | The GPUs Apsi can use (needs the NVIDIA driver; no CUDA toolkit). |
| `grad(f, wrt=0)` | gradient of a function (automatic differentiation) |
| `group_by(xs, key)` | function |

### H

| Name | What it is |
|---|---|
| `hasattr(obj, name)` | true when a value has a field or method |
| `hash(x)` | A stable hash for numbers, strings, tuples and struct instances; tensors hash their bytes. |
| `hessian(f)` | hessian(f)(x) -> matrix of second derivatives of a scalar f. |
| `hex(number, /)` | Return the hexadecimal representation of an integer. |
| `http_get(url, timeout=30)` | download a URL (needs --allow-net) |
| `http_post_json(url, data, timeout=30)` | POST JSON to a URL (needs --allow-net) |
| `hypot(*xs)` | √(Σ x²) without overflow/underflow; works with quantities of the same unit and with tensors. |

### I

| Name | What it is |
|---|---|
| `i32` | value |
| `i64` | value |
| `id(obj, /)` | Return the identity of an object. |
| `ImportError(/, *args, **kwargs)` | Import can't find module, or can't find name in module. |
| `IndexError(/, *args, **kwargs)` | Sequence index out of range. |
| `inf` | constant |
| `input(prompt='')` | Read one line from standard input (without the newline); returns none at end of input. |
| `int(x, base=None)` | convert to an integer |
| `int4` | value |
| `int8` | value |
| `IOError(/, *args, **kwargs)` | Base class for I/O related errors. |
| `IsADirectoryError(/, *args, **kwargs)` | Operation doesn't work on directories. |
| `isfinite(x)` | function |
| `isinstance(x, t)` | type test |
| `isnan(x)` | function |
| `issubclass(cls, class_or_tuple, /)` | Return whether 'cls' is derived from another class or is the same class. |
| `iter(…)` | iter(iterable) -> iterator iter(callable, sentinel) -> iterator |

### J

| Name | What it is |
|---|---|
| `jacobian(f)` | jacobian(f)(x) -> d f_i / d x_j (f: Tensor[M] -> Tensor[M], x: Tensor[N]) as Tensor[M, N]. |
| `join(xs, sep='')` | join strings with a separator |
| `json_parse(text)` | parse JSON text into lists, dicts, numbers and strings |
| `json_str(value, indent=None, sort_keys=False, ensure_ascii=False, separators=None, default=None)` | turn a value into JSON text (indent, sort_keys …) |

### K

| Name | What it is |
|---|---|
| `KeyError(/, *args, **kwargs)` | Mapping key not found. |
| `keys(d)` | keys of a dict as a list |

### L

| Name | What it is |
|---|---|
| `laplace(f)` | laplace(f)(x) -> Σ ∂²f/∂x_i² (trace of the Hessian), e.g. for PDE residuals. |
| `len(x)` | number of items |
| `lerp(a, b, t)` | function |
| `library_path(path)` | The actual file cload() would load (e.g. vecmath -> vecmath.dll on Windows). |
| `linspace(start, stop, n=50)` | n evenly spaced values from start to stop |
| `list(x=())` | make a list from any sequence |
| `load_csv(path, skip_header=0)` | read a CSV file into columns/rows |
| `load_json(path)` | read a JSON file |
| `log(x)` | natural logarithm |
| `log10(x)` | base-10 logarithm |
| `log1p(x)` | function |
| `log2(x)` | base-2 logarithm |
| `log_softmax(x, axis=-1)` | log of softmax, numerically stable |
| `logsumexp(xs, axis=None)` | function |
| `LookupError(/, *args, **kwargs)` | Base class for lookup errors. |

### M

| Name | What it is |
|---|---|
| `magnitude(x)` | the number without its unit (in SI base units) |
| `map(f, xs)` | apply a function to every item |
| `matmul(a, b)` | matrix product (also `a @ b`) |
| `max(*args, axis=None, keepdims=False, key=None, default=…)` | largest value (of a list/tensor, or of the arguments) |
| `mean(x, axis=None, keepdims=False, start=None)` | average |
| `measure(value, sigma)` | value ± sigma -> an independent measurement with standard uncertainty sigma. |
| `median(xs)` | function |
| `memo(fn)` | cache a function's results (@memo) |
| `MemoryError(/, *args, **kwargs)` | Out of memory. |
| `meshgrid(x, y)` | coordinate grids from 1-D axes |
| `min(*args, axis=None, keepdims=False, key=None, default=…)` | smallest value |
| `mode(xs)` | function |
| `ModuleNotFoundError(/, *args, **kwargs)` | Module not found. |

### N

| Name | What it is |
|---|---|
| `NameError(/, *args, **kwargs)` | Name not found globally. |
| `nan` | constant |
| `nan_to_num(x, nan=0.0, posinf=None, neginf=None)` | function |
| `native(*args, **kwargs)` | compile a function to machine code (@native) |
| `native_info()` | What the native compiler sees on this machine (CPU, SIMD, threads, cache). |
| `native_threads(n=None)` | Number of CPU threads `parallel for` and large tensor kernels use (set it with an argument). |
| `nbytes(x)` | memory a tensor uses, in bytes |
| `ndim(x)` | number of tensor dimensions |
| `next(…)` | next(iterator[, default]) |
| `nominal(x)` | central value of an uncertain quantity |
| `norm(x, axis=None)` | Euclidean (or other) norm |
| `NotImplementedError(/, *args, **kwargs)` | Method or function hasn't been implemented yet. |

### O

| Name | What it is |
|---|---|
| `object(/, *args, **kwargs)` | The base class of the class hierarchy. |
| `oct(number, /)` | Return the octal representation of an integer. |
| `ones(*shape, dtype=None)` | tensor of ones |
| `ones_like(x)` | ones with the shape of another tensor |
| `open(file, mode='r', *args, **kwargs)` | function |
| `ord(character, /)` | Return the ordinal value of a character. |
| `OSError(/, *args, **kwargs)` | Base class for I/O related errors. |
| `outer(a, b)` | outer product |
| `OverflowError(/, *args, **kwargs)` | Result too large to be represented. |

### P

| Name | What it is |
|---|---|
| `pairwise(xs)` | function |
| `param_count(tree)` | number of trainable numbers in a model |
| `PermissionError(/, *args, **kwargs)` | Not enough permissions. |
| `permissions()` | the permissions the program runs with |
| `permutation(n, seed=None)` | random permutation of 0..n |
| `permutations(xs, k=None)` | function |
| `permute(x, *dims)` | Reorder the axes (PyTorch's x.permute(...)). |
| `pi` | constant |
| `platform()` | "windows", "linux" or "macos". |
| `pmap(f, xs, workers=None)` | Parallel map on a thread pool (numpy/BLAS work releases the GIL). |
| `pow(base, exp, mod=None)` | Equivalent to base**exp with 2 arguments or base**exp % mod with 3 arguments |
| `precision(dtype)` | @precision(f32) / @precision(exact): numeric precision of a function. |
| `print(*args, sep=' ', end='\n', flush=False, file=None)` | write values to the output, separated by spaces |
| `prod(x, axis=None)` | product of all values |
| `product(*lists, repeat=1)` | function |
| `property(/, *args, **kwargs)` | Property attribute. |
| `ptr_f32` | value |
| `ptr_f64` | value |
| `ptr_i32` | value |
| `ptr_i64` | value |
| `py_type_name(x)` | The name Python would give the type of x (int, float, str, list, dict, or the struct's name). |
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
| `RecursionError(/, *args, **kwargs)` | Recursion limit exceeded. |
| `relative_uncertainty(x)` | uncertainty divided by the value |
| `relu(x)` | max(x, 0) |
| `repr(x)` | text form of a value as code would write it |
| `reshape(x, *shp)` | same data, new shape |
| `reversed(xs)` | items in reverse order |
| `roll(x, shift, axis=None)` | shift values cyclically |
| `round(x, digits=0)` | round to n digits |
| `round_to(x, fmt, scale='auto', axis=None)` | Round x to what the format can represent and return it as a normal tensor ("fake quantisation"). Gradients pass straight through, so this works inside training. |
| `RuntimeError(/, *args, **kwargs)` | Unspecified run-time error. |

### S

| Name | What it is |
|---|---|
| `safe_div(a, b, default=0.0)` | function |
| `save_csv(path, data, header='')` | write rows to a CSV file |
| `save_json(path, value)` | write a value as a JSON file |
| `security_policy()` | the active security profile |
| `seed(n)` | make random numbers reproducible |
| `set(/, *args, **kwargs)` | a set: unique items, fast `in` |
| `set_recursion_limit(n)` | Allow deeper recursion (Python's sys.setrecursionlimit), capped at 1 000 000; the stack grows with it. |
| `setattr(obj, name, value)` | function |
| `shape(x)` | shape of a tensor |
| `shl(a, b)` | `a << b` |
| `shr(a, b)` | `a >> b` |
| `sigma(x)` | standard uncertainty of a value |
| `sigmoid(x)` | 1 / (1 + e^-x) |
| `sin(x)` | sine (radians or an angle quantity) |
| `sinh(x)` | hyperbolic sine |
| `slice(/, *args, **kwargs)` | slice(stop) slice(start, stop[, step]) |
| `softmax(x, axis=-1)` | normalised exponentials (probabilities) |
| `softplus(x)` | log(1 + e^x) |
| `sort(x)` | sorted copy of a tensor |
| `sorted(xs, key=None, reverse=False)` | sorted copy of a list (key=, reverse=) |
| `sqrt(x)` | square root (units become their square root) |
| `squeeze(x, axis=None)` | Remove axes of length 1 (only `axis` when given, and only if it has length 1, as in PyTorch). |
| `stack(items, axis=0)` | stack tensors along a new axis |
| `staticmethod(/, *args, **kwargs)` | Convert a function to be a static method. |
| `std(x, axis=None)` | standard deviation |
| `stop_gradient(x)` | use a value without differentiating through it |
| `StopIteration(/, *args, **kwargs)` | Signal the end from iterator.__next__(). |
| `str(x)` | text form of a value |
| `sum(x, axis=None, keepdims=False, start=None)` | sum of all values (or along an axis) |
| `super(/, *args, **kwargs)` | super() -> same as super(__class__, <first argument>) super(type) -> unbound super object super(type, obj) -> bound super object; requires isinstance(obj, type) super(type, type2) -> bound super object; requires issubcla |
| `swapaxes(x, a, b)` | Exchange two axes (PyTorch's x.transpose(a, b)). |

### T

| Name | What it is |
|---|---|
| `tan(x)` | tangent |
| `tanh(x)` | hyperbolic tangent |
| `tau` | constant |
| `tensor(x, dtype=None)` | make a tensor from numbers or nested lists (dtype=…) |
| `testing()` | true while `apsi test` runs |
| `timed(fn)` | @timed: print how long each call takes. |
| `timeit(f, repeat=5, warmup=1)` | Run f() repeat times after warm-up runs; seconds as a record(best, median, mean, runs). |
| `TimeoutError(/, *args, **kwargs)` | Timeout expired. |
| `to_float(x)` | plain float from a quantity or 1-element tensor |
| `to_py(x)` | Apsi value -> plain Python/JSON-friendly value (units become {value, unit}). |
| `to_torch(x, device=None)` | turn an Apsi tensor into a PyTorch tensor (needs --allow-python) |
| `transpose(x, axes=None)` | swap tensor axes |
| `tree_map(f, *trees)` | apply a function to every leaf of nested lists/dicts/records |
| `tuple(/, *args, **kwargs)` | make a tuple |
| `type(v)` | type name of a value ("Int", "Tensor" …) |
| `TypeError(/, *args, **kwargs)` | Inappropriate argument type. |

### U

| Name | What it is |
|---|---|
| `u` | `u.km`, `u.eV` ... -> one of that unit as a Quantity. |
| `uint8` | value |
| `UnicodeDecodeError(/, *args, **kwargs)` | Unicode decoding error. |
| `UnicodeEncodeError(/, *args, **kwargs)` | Unicode encoding error. |
| `UnicodeError(/, *args, **kwargs)` | Unicode related error. |
| `unique(xs)` | distinct values |
| `unit(x)` | the unit of a quantity |
| `unsqueeze(x, axis)` | Insert an axis of length 1 (negative axes count from the end, as in PyTorch). |

### V

| Name | What it is |
|---|---|
| `value_and_grad(f, wrt=0)` | value and gradient of a function (automatic differentiation) |
| `ValueError(/, *args, **kwargs)` | Inappropriate argument value (of correct type). |
| `var(x, axis=None)` | variance |
| `vars(obj)` | The fields of a struct or record as a dict. |
| `void` | value |

### W

| Name | What it is |
|---|---|
| `where(cond, a, b)` | choose element-wise: where(cond, a, b) |
| `windows(xs, n)` | function |
| `write_err(text)` | function |
| `write_out(text)` | Write text without a line break (Python's sys.stdout.write). |
| `write_text(path, text)` | write a text file (needs write permission) |

### Z

| Name | What it is |
|---|---|
| `ZeroDivisionError(/, *args, **kwargs)` | Second argument to a division or modulo operation was zero. |
| `zeros(*shape, dtype=None)` | tensor of zeros |
| `zeros_like(x)` | zeros with the shape of another tensor |
| `zip(*xs)` | pairs items of several lists |

## Standard modules (43)

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

### `use dataframe` — 12 names

| Name | What it is |
|---|---|
| `col(name)` | function |
| `concat(frames)` | function |
| `count()` | function |
| `DataFrame(data=None, columns=None)` | type |
| `Expr(fn, name, agg=None)` | A column expression, evaluated against a DataFrame: col("a") * 2 > col("b"). |
| `frame(data=None, columns=None)` | type |
| `from_csv_text(text, sep=',', columns=None)` | function |
| `from_records(rows, columns=None)` | function |
| `lit(v)` | function |
| `read_csv(path, sep=',', columns=None, interp=None)` | function |
| `read_json(path, interp=None)` | A JSON array of objects, or an object of columns → a DataFrame. |
| `read_jsonl(path, interp=None)` | One JSON object per line (JSON Lines / NDJSON) → a DataFrame (missing keys become nulls). |

### `use db` — 10 names

| Name | What it is |
|---|---|
| `DatabaseError(msg, code=None, detail=None)` | An error reported by the database server (has .code: the SQLSTATE for PostgreSQL). |
| `mariadb(url=None, **kw)` | function |
| `mongo(url=None, **kw)` | function |
| `mysql(url=None, **kw)` | function |
| `pool(url, size=8, **kw)` | function |
| `postgres(url=None, **kw)` | function |
| `redis(url=None, **kw)` | function |
| `RedisError(msg, code=None, detail=None)` | type |
| `Result(rows=(), columns=(), status='', rowcount=0)` | Rows of a query (a list), with .columns, .rowcount (rows affected) and .status ("INSERT 0 3" …). |
| `sqlite(path=':memory:')` | function |

### `use env` — 12 names

| Name | What it is |
|---|---|
| `bool(name, default=False)` | function |
| `float(name, default=None)` | function |
| `get(name, default=None)` | function |
| `has(name)` | function |
| `int(name, default=None)` | function |
| `list(name, sep=',', default=None)` | function |
| `load(path='.env', override=True, export=False)` | Read a .env file (needs --allow-read for it). Returns the names it defined. |
| `parse(text)` | Parse .env text into a dict (the same rules as python-dotenv / docker compose). |
| `require(*names)` | function |
| `secret(name, default=None)` | function |
| `Secret(value, name='')` | A value that does not show itself: print, str, f-strings and logs give ******. |
| `values` | value |

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
| `gpu_info()` | GPU 0: name, compute capability, memory. Works with CuPy, or with just the NVIDIA driver. |
| `hardware()` | function |
| `to_cpu(x)` | function |
| `to_device(x, dev=None)` | function |
| `to_gpu(x)` | function |

### `use hp` — 36 names

| Name | What it is |
|---|---|
| `acos(x)` | function |
| `asin(x)` | function |
| `atan(x)` | function |
| `atan2(y, x)` | function |
| `cbrt(x)` | function |
| `cos(x)` | function |
| `cosh(x)` | function |
| `det(a)` | function |
| `digits(n=None)` | Set (or read) the number of significant digits of hp results (default 50, at most 100 000). |
| `dot(a, b)` | function |
| `e()` | function |
| `equal(a, b, digits_=None)` | True when a and b agree to `digits` significant digits (default: the current precision minus 2). |
| `exp(x)` | function |
| `factorial(n)` | function |
| `float(x)` | function |
| `gamma(x)` | Γ(x) for positive integers and half-integers (exact formulas). |
| `inv(a)` | function |
| `ln(x)` | function |
| `log(x)` | function |
| `log10(x)` | function |
| `log2(x)` | function |
| `matmul(a, b)` | function |
| `mean(xs)` | function |
| `num(x)` | An hp number. Strings keep every digit you write ("0.1" is exactly one tenth); floats keep their exact binary value (0.1 becomes 0.1000000000000000055511…), integers and fractions are exact. |
| `pi()` | π to the current number of digits (Chudnovsky-free: Machin's formula with binary splitting of atan). |
| `pow(x, y)` | function |
| `prod(xs)` | function |
| `root(x, n)` | The real n-th root (negative x allowed for odd n). |
| `sin(x)` | function |
| `sinh(x)` | function |
| `solve(a, b)` | x with a·x = b, by LU factorisation with partial pivoting at the current precision (+ guard digits). |
| `sqrt(x)` | function |
| `str(x, n=None)` | x as text with exactly n significant digits (default: the current precision). |
| `sum(xs)` | The exact sum (no rounding error accumulates), rounded once to the current precision. |
| `tan(x)` | function |
| `tanh(x)` | function |

### `use hub` — 15 names

| Name | What it is |
|---|---|
| `cache_dir()` | function |
| `clear_cache(repo=None)` | function |
| `config(repo, revision='main', token=None)` | function |
| `download(repo, filename, revision='main', token=None, kind='model', progress=True)` | One file → its local path (cached by commit; resumed; SHA-256 checked for large files). |
| `files(repo, revision=None, token=None, kind='model')` | function |
| `HubError(msg, line=None, col=None)` | type |
| `info(repo, revision=None, token=None, kind='model')` | function |
| `load(repo, filename=None, revision='main', token=None, dtype=None)` | Weights as {name: tensor}: model.safetensors (also sharded), or a named .safetensors / .gguf file. |
| `login(token)` | function |
| `logout()` | function |
| `offline(on=True)` | function |
| `search(query='', task=None, limit=20, sort='downloads')` | function |
| `snapshot(repo, patterns=('*.json', '*.safetensors', '*.txt', '*.model'), revision='main', token=None, kind='model')` | Download the files that match the patterns; returns the folder. |
| `tokenizer(repo, revision='main', token=None)` | function |
| `tokenizer_from_file(path)` | function |

### `use io` — 9 names

| Name | What it is |
|---|---|
| `exists(path)` | function |
| `json_parse(text)` | function |
| `json_str(value, indent=None, sort_keys=False, ensure_ascii=False, separators=None, default=None)` | function |
| `load_csv(path, skip_header=0)` | function |
| `load_json(path)` | function |
| `read_text(path)` | function |
| `save_csv(path, data, header='')` | function |
| `save_json(path, value)` | function |
| `write_text(path, text)` | function |

### `use lab` — 9 names

| Name | What it is |
|---|---|
| `experiment(name, params=None, folder='experiments')` | Start an experiment log. Returns a record with log(step, metrics), note(text), result(metrics), finish() and the run id. Every call appends one JSON line to FOLDER/NAME.jsonl. |
| `format_measurement(value, uncertainty, unit=None)` | Value with its standard uncertainty in parenthesis notation, rounded by the usual rule (uncertainty to two significant digits when they start with 1 or 2, otherwise one): 9.8123 ± 0.0312 -> "9.81(3)". |
| `load_runs(folder='experiments', name=None)` | All runs in an experiment folder (or one experiment): list of record(run, name, params, logs, result, notes, provenance, finished). |
| `provenance()` | What produced a result: software versions, machine and time (no user names, host names or paths). |
| `set_seed(seed)` | Seed every random generator Apsi uses (Apsi's random module, numpy's legacy and Python's random). |
| `sigfig(x, n=3)` | Round to n significant figures. |
| `table_csv(rows, headers=None)` | CSV text (RFC 4180 quoting). |
| `table_latex(rows, headers=None, caption=None, label=None)` | A LaTeX table (booktabs) from lists or records; text is escaped. |
| `table_markdown(rows, headers=None)` | A Markdown table from lists or records. |

### `use layers` — 70 names

| Name | What it is |
|---|---|
| `activation(fn)` | function |
| `add()` | function |
| `autoencoder(d_in, latent, hidden=(64,), act='relu')` | Returns a DAG with outputs [reconstruction, code]. |
| `avg_pool(k=2)` | type |
| `batch_norm_layer(d, eps=1e-05)` | Normalises over the batch (and spatial) axes with the current batch's statistics. |
| `bilinear(n1, n2, n_out)` | type |
| `check_torch(*a, **k)` | Compare every layer type on NumPy and PyTorch (outputs and gradients, float64). |
| `checkpoint(layer)` | Gradient checkpointing: checkpoint(layer) gives the same results with much less activation memory. |
| `cnn(in_channels, classes, image=28, channels=(16, 32), dense_width=64)` | Small convolutional classifier for images shaped [N, C, H, W]. |
| `concatenate()` | function |
| `conv2d_layer(c_in, c_out, k=3, stride=1, padding='same')` | type |
| `count_params(params)` | function |
| `custom_layer(init, forward, name='model', parts=None)` | A layer from your own functions: init(seed) returns the parameter tree, forward(p, x, train) the output. |
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
| `load_torch_weights(model, params, state, prefix='')` | Copy a PyTorch state_dict (or a flat record/dict of arrays with the same names, e.g. from load_safetensors(path, nested=false)) into an Apsi parameter tree of the same architecture. Linear weights are transposed, LSTM/GR |
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

### `use math` — 35 names

| Name | What it is |
|---|---|
| `abs(x)` | function |
| `acos(x)` | function |
| `asin(x)` | function |
| `atan(x)` | function |
| `atan2(y, x)` | function |
| `besselj(n, x)` | Bessel function of the first kind J_n(x), integer n. |
| `beta(a, b)` | B(a, b) = Γ(a)Γ(b)/Γ(a+b). |
| `cbrt(x)` | function |
| `ceil(x)` | function |
| `comb(n, k)` | Binomial coefficient C(n, k) (exact for integers). |
| `cos(x)` | function |
| `cosh(x)` | function |
| `e` | constant |
| `erf(x)` | function |
| `erfc(x)` | function |
| `erfinv(y)` | Inverse error function. |
| `exp(x)` | function |
| `factorial(n)` | n! (exact integer for integers). |
| `floor(x)` | function |
| `gamma(x)` | Γ(x). |
| `inf` | constant |
| `legendre(n, x)` | Legendre polynomial P_n(x). |
| `lgamma(x)` | ln\|Γ(x)\|. |
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

### `use mechanics` — 22 names

| Name | What it is |
|---|---|
| `beam(length, E, I, support='cantilever', loads=None, distributed=0.0, n=100)` | Euler–Bernoulli beam by finite elements (cubic Hermite elements, exact at the nodes for these loads). |
| `hamiltonian_flow(H, q0, p0, dt, steps, method='leapfrog', every=1)` | Hamilton's equations q' = ∂H/∂p, p' = −∂H/∂q with gradients from automatic differentiation. "leapfrog" is symplectic for separable H = T(p) + V(q); "rk4" works for any H. Returns record(t, q, p, energy). |
| `hohmann(r1, r2, mu)` | Hohmann transfer between circular orbits: record(dv1, dv2, dv_total, time). |
| `inertia_tensor(masses, positions, about='com')` | Inertia tensor of point masses about their centre of mass (about="com") or the origin ("origin"). |
| `kepler_solve(M, e)` | Eccentric anomaly E from the mean anomaly M (Kepler's equation M = E − e sin E), 0 ≤ e < 1. |
| `lagrangian_flow(L, q0, qdot0, dt, steps, every=1)` | Euler–Lagrange equations of L(q, qdot), derived by automatic differentiation: M(q, q̇) q̈ = ∂L/∂q − (∂²L/∂q̇∂q) q̇ with M = ∂²L/∂q̇². Integrated with RK4. Returns record(t, q, qdot, energy) where energy = q̇·∂L/∂q̇ − L. |
| `nbody(masses, x0, v0, dt, steps, G=6.6743e-11, softening=0.0, method='yoshida4', every=1)` | Gravitational N-body integration (direct summation, symplectic). Positions/velocities have shape [N, 3] (or [N, 2]). Returns record(t, x, v, energy, momentum, angular_momentum, energy_drift). |
| `normal_modes(K, M)` | Normal modes of M x'' + K x = 0: angular frequencies ω (rad/s), frequencies (Hz) and mass-normalized mode shapes (columns), from the generalized symmetric eigenproblem K φ = ω² M φ. |
| `orbit_elements(r, v, mu)` | Classical orbital elements from a state vector: a, e, i, raan, argp, nu (angles in radians). |
| `orbit_state(a, e, i, raan, argp, nu, mu)` | Position and velocity from classical orbital elements -> record(r, v). |
| `orbital_period(a, mu)` | T = 2π √(a³/μ) in seconds. |
| `principal_axes(I)` | Principal moments (ascending) and axes (columns) of an inertia tensor. |
| `quat(axis, angle)` | Unit quaternion [w, x, y, z] for a rotation by `angle` about `axis`. |
| `quat_conj(q)` | function |
| `quat_mul(a, b)` | Hamilton product a·b. |
| `quat_rotate(q, v)` | Rotate vector v by unit quaternion q. |
| `quat_to_matrix(q)` | function |
| `rigid_body(I, omega0, q0, dt, steps, torque=None, every=1)` | Free (or torqued) rotation of a rigid body: Euler's equations in the body frame + quaternion attitude, RK4. I: 3×3 body inertia tensor (or its 3 principal moments); torque(t, omega, q) in the body frame. Returns record(t |
| `rotation_matrix(axis, angle)` | Rotation matrix for a right-handed rotation by `angle` about `axis` (Rodrigues). |
| `simulate(accel, x0, v0, dt, steps, method='verlet', t0=0.0, every=1)` | Integrate x'' = accel(x, v, t) (accel may also take (x) or (x, v)). |
| `truss2d(nodes, members, E, A, supports, loads)` | Plane pin-jointed truss by the direct stiffness method. nodes: [[x, y], ...]; members: [[i, j], ...] (0-based node indices); E, A: numbers or one per member; supports: [[node, fix_x, fix_y], ...] with true/false; loads:  |
| `vis_viva(r, a, mu)` | Orbital speed at distance r on an orbit of semi-major axis a. |

### `use medical` — 32 names

| Name | What it is |
|---|---|
| `anion_gap(sodium, chloride, bicarbonate, potassium=None)` | Serum anion gap (mmol/L): Na − (Cl + HCO3) [+ K if given]. |
| `bmi(weight, height)` | Body-mass index in kg/m². weight in kg, height in m (numbers) or as quantities (70kg, 175cm). |
| `brier_score(prob, outcome)` | Mean squared error of predicted probabilities against 0/1 outcomes. |
| `bsa(weight, height, method='mosteller')` | Body-surface area in m² (Mosteller or DuBois & DuBois). weight in kg, height in cm (numbers) or quantities. |
| `cockcroft_gault(age, weight, creatinine, sex)` | Creatinine clearance (mL/min) by Cockcroft–Gault. weight in kg, creatinine in mg/dL. |
| `cohen_kappa(a, b, level=0.95)` | Cohen's kappa for two raters' categorical labels, with its asymptotic standard error and CI. |
| `concordance_index(time, risk, event, higher_is_worse=True)` | Harrell's C: the fraction of comparable pairs whose predicted risk orders the event times correctly (ties in prediction count ½). |
| `cox_ph(time, event, X, names=None, level=0.95, max_iter=50)` | Cox proportional-hazards regression (Efron's method for ties), Newton–Raphson. Returns coefficients, SE, z, p, hazard ratios with CI, log-likelihood, likelihood-ratio test and the concordance index. |
| `diagnostic_metrics(tp, fp, fn, tn, level=0.95)` | Accuracy of a binary test against a reference standard, with confidence intervals (Wilson for proportions, log method for likelihood ratios and the diagnostic odds ratio). |
| `DISCLAIMER` | value |
| `ecg_rpeaks(ecg, fs, min_rr=0.25)` | R-peak detection in a single-lead ECG (Pan–Tompkins style: 5–15 Hz band-pass, derivative, squaring, 150 ms integration, adaptive threshold, refinement to the true maximum). Returns record(indices, times, rr (s), heart_ra |
| `egfr_ckd_epi_2021(creatinine, age, sex)` | eGFR (mL/min/1.73 m²) by the race-free CKD-EPI 2021 creatinine equation. creatinine in mg/dL. |
| `final_size(R0)` | Final epidemic size z of the SIR model: z = 1 − exp(−R0·z) (fully susceptible population). |
| `hrv(rr, frequency_domain=True)` | Heart-rate variability from RR intervals in seconds: mean RR, SDNN, RMSSD, SDSD, pNN50, mean HR (ms and bpm) and, with ≥ 60 s of data, LF (0.04–0.15 Hz), HF (0.15–0.4 Hz) power and LF/HF from a 4 Hz resampled tachogram ( |
| `ideal_body_weight(height, sex)` | Devine ideal body weight (kg); height in cm (number) or as a quantity. |
| `incidence_rate(cases, person_time, level=0.95, per=1.0)` | Incidence rate with the exact (Poisson / chi-squared) confidence interval, scaled per `per` person-time. |
| `kaplan_meier(time, event, level=0.95)` | Kaplan–Meier estimate: record(times, at_risk, events, censored, survival, se (Greenwood), ci (log-log), median). event: 1 = event observed, 0 = censored. |
| `logrank(time, event, group)` | Log-rank test comparing survival between two or more groups: record(statistic, p, df, observed, expected, groups). |
| `mantel_haenszel(tables, level=0.95)` | Pooled odds ratio and risk ratio over strata of 2×2 tables [[a, b], [c, d]] (Mantel–Haenszel) with the Robins–Breslow–Greenland CI, the Cochran–Mantel–Haenszel test and the Breslow–Day-free homogeneity summary. |
| `mean_arterial_pressure(systolic, diastolic)` | MAP ≈ DBP + (SBP − DBP)/3 (same unit as the inputs, usually mmHg). |
| `nca(t, c, dose=None, route='extravascular', lambda_points=None)` | Non-compartmental analysis of one concentration–time profile (linear-up / log-down trapezoids). Returns Cmax, Tmax, AUC_last, AUC_inf, AUC_%extrap, AUMC, λz (with R² and points used), t½, MRT and, when dose is given, CL  |
| `pk_one_compartment(dose, V, t, CL=None, ke=None, route='iv', ka=None, F=1.0, tau=None, doses=1, infusion_time=None)` | Plasma concentration of a one-compartment model with first-order elimination. |
| `pk_two_compartment(dose, V1, k10, k12, k21, t)` | IV bolus into a two-compartment model: C(t) = A e^{−αt} + B e^{−βt}. Returns record(c, alpha, beta, A, B, t_half_alpha, t_half_beta). |
| `qtc(qt, rr=None, hr=None, method='fridericia')` | Heart-rate-corrected QT (ms). qt in ms; give rr (s) or hr (beats/min). |
| `read_dicom(path, include_identifiers=False)` | Read an uncompressed DICOM image: record(pixels (rescaled, e.g. Hounsfield units), rows, columns, pixel_spacing (mm), slice_thickness, modality, … ). Patient-identifying fields are left out unless include_identifiers=tru |
| `read_dicom_series(folder, include_identifiers=False)` | Read every DICOM slice in a folder and stack them into a volume sorted by position along the slice normal (or instance number): record(volume[z, y, x], spacing [z, y, x] mm, positions, info). |
| `risk_measures(a, b, c, d, level=0.95)` | 2×2 table: exposed (a cases, b non-cases), unexposed (c cases, d non-cases). Relative risk (log CI), odds ratio (Woolf CI), risk difference (Wald CI), NNT and the attributable fraction. |
| `roc(scores, labels, level=0.95)` | ROC analysis: record(fpr, tpr, thresholds, auc, auc_ci (DeLong, logit-free), auc_se, youden_threshold, youden_sensitivity, youden_specificity). labels: 1 = disease/positive, 0 = negative. |
| `roc_compare(scores1, scores2, labels)` | Paired DeLong test of two AUCs measured on the same subjects. |
| `seir(beta, sigma, gamma, S0, E0, I0, R0=0.0, days=200, dt=0.1)` | SEIR model with incubation rate sigma (1/latent period). Returns record(t, S, E, I, R, peak_day, R0). |
| `sir(beta, gamma, S0, I0, R0=0.0, days=160, dt=0.1)` | SIR epidemic model (counts or fractions). Returns record(t, S, I, R, peak_day, peak_infected, R0, attack_rate). |
| `window_level(image, center, width)` | Map intensities (e.g. Hounsfield units) to [0, 1] for display with a window centre and width (e.g. soft tissue 40/400, lung −600/1500, bone 400/1800, brain 40/80). |

### `use memory` — 13 names

| Name | What it is |
|---|---|
| `array(shape, dtype='float32', fill=None, ram=None)` | An array that lives in RAM while it fits the budget, else in a memory-mapped file on the SSD. ram=true/false forces the choice. |
| `best_device(need_gb=0.0)` | The GPU with the most free memory (if it has need_gb free), otherwise "cpu". |
| `budget(gb=None)` | Set (or read) the RAM budget for array(), in GB. |
| `cached(fn=None, name=None, ram_mb=128, disk_gb=5)` | Memoise fn in a two-tier store (RAM, then SSD). Use as @memory.cached or memory.cached(fn). |
| `chunks(a, rows=65536)` | function |
| `devices()` | function |
| `fit_batch(fn, batch, minimum=1)` | fn(batch); on out-of-memory the batch is halved (GPU cache emptied) and tried again. |
| `free(a)` | Release an array from array() now (RAM is returned, a spill file is deleted). |
| `map(fn, a, rows=65536, out=None, workers=None)` | function |
| `on_disk(a)` | function |
| `spill_dir()` | function |
| `status()` | function |
| `store(name='default', ram_mb=256, disk_gb=10)` | function |

### `use models` — 16 names

| Name | What it is |
|---|---|
| `dequantize_model(params)` | function |
| `gguf_info(path)` | function |
| `hf_pipeline(task, model=None, device=None, **kw)` | A Hugging Face transformers pipeline: classify = hf_pipeline("sentiment-analysis"). |
| `load_gguf(path, nested=True)` | function |
| `load_model(path, verify=True, key=None)` | function |
| `load_safetensors(path, dtype=None, names=None, nested=True)` | function |
| `model_info(path)` | function |
| `model_size(params)` | Bytes used by all tensors in a parameter tree (counts compact formats at their real size). |
| `onnx_model(path, providers=None)` | Run an ONNX model with onnxruntime: m = onnx_model("net.onnx"); m({"input": x}). |
| `qlinear(x, W, b=None)` | x @ W (+ b) where W may be a QTensor — decodes, multiplies and accumulates in FP32. |
| `quantize_model(params, dtype=None, min_size=1024)` | Store every weight matrix (>= min_size elements) in a compact format; others stay as-is. |
| `safetensors_info(path)` | function |
| `save_gguf(path, params, metadata=None, dtype='f16', name=None, architecture='apsi')` | function |
| `save_model(path, params, metadata=None, shard='4GB', sign=False)` | function |
| `save_safetensors(path, params, metadata=None)` | function |
| `torch_load(path, map_location='cpu')` | function |

### `use net` — 2 names

| Name | What it is |
|---|---|
| `http_get(url, timeout=30)` | function |
| `http_post_json(url, data, timeout=30)` | function |

### `use nn` — 57 names

| Name | What it is |
|---|---|
| `accuracy(logits, labels)` | function |
| `adagrad(lr=0.01, eps=1e-10)` | function |
| `adam(lr=0.001, b1=0.9, b2=0.999, eps=1e-08)` | function |
| `adamw(lr=0.001, b1=0.9, b2=0.999, eps=1e-08, weight_decay=0.01)` | function |
| `attention(q, k, v, mask=None)` | Scaled dot-product attention over the last two axes. |
| `batch_norm(x, gamma=None, beta=None, eps=1e-05, axis=0)` | Batch normalisation over `axis` (training-mode statistics). |
| `batches(X, y, size, seed=None)` | function |
| `bce(prob, target)` | Binary cross-entropy on probabilities (clamped like PyTorch: log ≥ −100). |
| `bce_with_logits(logits, target)` | Binary cross-entropy on logits (numerically stable): mean(softplus(z) − y·z). |
| `causal_mask(T)` | function |
| `clip_grad_norm(grads, max_norm=1.0)` | function |
| `conv2d(x, w, b=None, stride=1, padding=0)` | function |
| `conv_init(c_in, c_out, k=3, seed=None)` | function |
| `cosine_lr(step, total, base=0.001, warmup=0, minimum=0.0)` | function |
| `cross_entropy(logits, labels)` | Mean cross-entropy of integer labels; labels [B] with logits [B, C], or labels [B, T] with logits [B, T, C] (sequence models: every position counts). |
| `dense(n_in, n_out, seed=None, scale=None)` | A dense layer's parameters: record(W, b) with He initialisation. |
| `dropout(x, p=0.1, train=True, seed=None)` | function |
| `embedding_init(vocab, d, seed=None)` | function |
| `gelu(x, approximate='tanh')` | GELU. approximate="tanh" (default, fast) or "none" (exact: x·Φ(x), as PyTorch's default). |
| `global_norm(grads)` | function |
| `gru(p, xs)` | function |
| `gru_cell(p, x, h)` | function |
| `gru_init(n_in, n_hidden, seed=None)` | function |
| `huber_loss(pred, target, delta=1.0)` | Huber / smooth-L1 loss (mean), quadratic below delta and linear above. |
| `l1_loss(pred, target)` | Mean absolute error. |
| `layer_norm(x, gamma=None, beta=None, eps=1e-05)` | function |
| `linear_warmup(step, base=0.001, warmup=100)` | function |
| `load_params(path, like)` | function |
| `loader(X, y, batch_size=32, shuffle=False, seed=None)` | Iterable mini-batches (like a PyTorch DataLoader over a TensorDataset); reshuffled every epoch. |
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
| `nll_loss(log_probs, labels)` | Negative log-likelihood of integer labels given log-probabilities [N, C]. |
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

### `use pde` — 9 names

| Name | What it is |
|---|---|
| `advection1d(u0, c, dx, dt, steps, scheme='lax-wendroff', every=1)` | u_t + c u_x = 0 on a periodic grid ("upwind": first order, monotone; "lax-wendroff": second order). |
| `bicgstab(A, b, tol=1e-10, max_iter=None, x0=None)` | BiCGSTAB for general square A. Returns record(x, iterations, residual, converged). |
| `cg(A, b, tol=1e-10, max_iter=None, x0=None)` | Conjugate gradients for symmetric positive-definite A. Returns record(x, iterations, residual, converged). |
| `heat1d(u0, dx, dt, steps, alpha, bc='dirichlet', left=None, right=None, every=1)` | u_t = α u_xx by Crank–Nicolson. bc: "dirichlet" (end values fixed at left/right, default: the initial end values), "neumann" (zero flux) or "periodic". Returns record(t, x, u) with u[frame, i]. |
| `heat2d(u0, dx, dt, steps, alpha, every=1)` | u_t = α ∇²u on a square grid by the ADI method (unconditionally stable, second order). Boundary values stay at their initial values. Returns record(t, u) with u[frame, i, j]. |
| `navier_stokes2d(omega0, nu, length=6.283185307179586, dt=0.001, steps=1000, every=0)` | 2-D incompressible Navier–Stokes in vorticity form ω_t + u·∇ω = ν∇²ω on a periodic square, pseudo-spectral with 2/3 de-aliasing and RK4 (viscous term integrated exactly). Returns record(t, omega, u, v, energy) for the st |
| `poisson2d(f, dx, bc='dirichlet')` | Solve ∇²u = f on a uniform grid with the 5-point Laplacian. "dirichlet": u = 0 on the boundary (f gives the interior points); "periodic": f must have zero mean, the solution has zero mean. Fast transforms: O(N log N), ex |
| `tridiag_solve(a, b, c, d)` | Solve tridiagonal systems: a sub-diagonal (a[0] unused), b diagonal, c super-diagonal (c[-1] unused). d may have extra trailing dimensions (several right-hand sides solved at once). |
| `wave1d(u0, v0, c, dx, dt, steps, bc='fixed', every=1)` | u_tt = c² u_xx by the leapfrog scheme (needs c·dt ≤ dx). bc: "fixed", "free" or "periodic". Returns record(t, x, u). |

### `use physics` — 60 names

| Name | What it is |
|---|---|
| `a_0` | constant |
| `activity(N, half_life)` | Activity λN in becquerel. |
| `alpha_fs` | constant |
| `AU` | constant |
| `biot_savart(path, current, points)` | Magnetic field (T) of a current along a polyline `path` [[x, y, z], ...] at the given points. Each straight segment uses the exact finite-wire formula. |
| `blackbody_power(T, area=1.0, emissivity=1.0)` | Radiated power ε σ A T⁴. |
| `brewster_angle(n1, n2)` | function |
| `c` | constant |
| `carnot_efficiency(T_hot, T_cold)` | function |
| `coulomb_field(charges, positions, points, softening=0.0)` | Electric field (V/m) of point charges at the given points (2-D or 3-D), shape [points, dim]. |
| `coulomb_potential(charges, positions, points, softening=0.0)` | function |
| `critical_angle(n1, n2)` | function |
| `cyclotron_frequency(q, B, m)` | Angular cyclotron frequency \|q\|B/m (rad/s). |
| `de_broglie(m, v)` | Matter wavelength h/(m v) (non-relativistic). |
| `decay(N0, half_life, t)` | Remaining amount N0 · 2^(−t/T½) (same unit as N0). |
| `doppler(f, v_source=0.0, v_observer=0.0, c_wave=343.0)` | Classical Doppler shift; speeds positive when source and observer approach each other. |
| `double_slit(theta, spacing, width, wavelength)` | Relative intensity of two slits of finite width. |
| `epsilon_0` | constant |
| `fresnel(n1, n2, theta)` | Fresnel power coefficients for s and p polarization -> [Rs, Rp, Ts, Tp]. |
| `G` | constant |
| `g0` | constant |
| `gradient(u, dx, axis=0)` | Central-difference derivative of a field along one axis (periodic). |
| `h` | constant |
| `half_life_from_rate(lam)` | function |
| `hbar` | constant |
| `heat_conduction(k, area, dT, thickness)` | Fourier's law: heat flow k A ΔT / d (W). |
| `ideal_gas_pressure(n, T, V)` | p = n R T / V (n in mol). |
| `k_B` | constant |
| `laplacian(u, dx, boundary='periodic')` | Discrete Laplacian (∇²u) on a regular grid with spacing dx. |
| `length_contraction(L, v)` | function |
| `lorentz_boost(t, x, v)` | Coordinates (t', x') in a frame moving with speed v along x -> [t', x']. |
| `lorentz_factor(v)` | γ = 1/√(1 − v²/c²). |
| `lorentz_force(q, E, B, v)` | F = q (E + v × B). |
| `m_e` | constant |
| `M_earth` | constant |
| `m_n` | constant |
| `m_p` | constant |
| `M_sun` | constant |
| `maxwell_speed_pdf(v, m, T)` | Maxwell–Boltzmann speed distribution f(v) (s/m). |
| `mean_speed(m, T)` | function |
| `meshgrid(x, y)` | function |
| `mu_0` | constant |
| `N_A` | constant |
| `photon_energy(wavelength)` | function |
| `planck(wavelength, T)` | Spectral radiance B_λ(λ, T) in W/(m²·sr·m). |
| `q_e` | constant |
| `R_earth` | constant |
| `R_gas` | constant |
| `relativistic_doppler(f, v)` | Frequency received from a source receding at speed v (negative v: approaching). |
| `relativistic_energy(m, v)` | Total energy γ m c². |
| `relativistic_momentum(m, v)` | function |
| `rms_speed(m, T)` | function |
| `sigma_SB` | constant |
| `single_slit(theta, width, wavelength)` | Relative Fraunhofer intensity of a single slit: sinc²(π a sinθ / λ). |
| `skin_depth(f, sigma, mu_r=1.0)` | Electromagnetic skin depth √(2/(ω μ σ)) of a good conductor. |
| `snell(n1, n2, theta)` | Refraction angle (radians); NaN for total internal reflection. |
| `thin_lens(f, d_o)` | Image distance and magnification for a thin lens -> [d_i, m]. |
| `time_dilation(dt, v)` | Dilated time γ·Δt of a moving clock's proper interval Δt. |
| `velocity_add(u, v)` | Relativistic velocity addition (u + v)/(1 + uv/c²). |
| `wien_peak(T)` | Wavelength of maximum spectral radiance (Wien's law). |

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

### `use physics.formulas` — 36 names

| Name | What it is |
|---|---|
| `activity(N, half_life)` | Activity λN in becquerel. |
| `biot_savart(path, current, points)` | Magnetic field (T) of a current along a polyline `path` [[x, y, z], ...] at the given points. Each straight segment uses the exact finite-wire formula. |
| `blackbody_power(T, area=1.0, emissivity=1.0)` | Radiated power ε σ A T⁴. |
| `brewster_angle(n1, n2)` | function |
| `carnot_efficiency(T_hot, T_cold)` | function |
| `coulomb_field(charges, positions, points, softening=0.0)` | Electric field (V/m) of point charges at the given points (2-D or 3-D), shape [points, dim]. |
| `coulomb_potential(charges, positions, points, softening=0.0)` | function |
| `critical_angle(n1, n2)` | function |
| `cyclotron_frequency(q, B, m)` | Angular cyclotron frequency \|q\|B/m (rad/s). |
| `de_broglie(m, v)` | Matter wavelength h/(m v) (non-relativistic). |
| `decay(N0, half_life, t)` | Remaining amount N0 · 2^(−t/T½) (same unit as N0). |
| `doppler(f, v_source=0.0, v_observer=0.0, c_wave=343.0)` | Classical Doppler shift; speeds positive when source and observer approach each other. |
| `double_slit(theta, spacing, width, wavelength)` | Relative intensity of two slits of finite width. |
| `fresnel(n1, n2, theta)` | Fresnel power coefficients for s and p polarization -> [Rs, Rp, Ts, Tp]. |
| `half_life_from_rate(lam)` | function |
| `heat_conduction(k, area, dT, thickness)` | Fourier's law: heat flow k A ΔT / d (W). |
| `ideal_gas_pressure(n, T, V)` | p = n R T / V (n in mol). |
| `length_contraction(L, v)` | function |
| `lorentz_boost(t, x, v)` | Coordinates (t', x') in a frame moving with speed v along x -> [t', x']. |
| `lorentz_factor(v)` | γ = 1/√(1 − v²/c²). |
| `lorentz_force(q, E, B, v)` | F = q (E + v × B). |
| `maxwell_speed_pdf(v, m, T)` | Maxwell–Boltzmann speed distribution f(v) (s/m). |
| `mean_speed(m, T)` | function |
| `photon_energy(wavelength)` | function |
| `planck(wavelength, T)` | Spectral radiance B_λ(λ, T) in W/(m²·sr·m). |
| `relativistic_doppler(f, v)` | Frequency received from a source receding at speed v (negative v: approaching). |
| `relativistic_energy(m, v)` | Total energy γ m c². |
| `relativistic_momentum(m, v)` | function |
| `rms_speed(m, T)` | function |
| `single_slit(theta, width, wavelength)` | Relative Fraunhofer intensity of a single slit: sinc²(π a sinθ / λ). |
| `skin_depth(f, sigma, mu_r=1.0)` | Electromagnetic skin depth √(2/(ω μ σ)) of a good conductor. |
| `snell(n1, n2, theta)` | Refraction angle (radians); NaN for total internal reflection. |
| `thin_lens(f, d_o)` | Image distance and magnification for a thin lens -> [d_i, m]. |
| `time_dilation(dt, v)` | Dilated time γ·Δt of a moving clock's proper interval Δt. |
| `velocity_add(u, v)` | Relativistic velocity addition (u + v)/(1 + uv/c²). |
| `wien_peak(T)` | Wavelength of maximum spectral radiance (Wien's law). |

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

### `use pycompat` — 146 names

| Name | What it is |
|---|---|
| `abs(x, /)` | Return the absolute value of the argument. |
| `aiter(async_iterable, /)` | Return an AsyncIterator for an AsyncIterable object. |
| `all(iterable, /)` | Return True if bool(x) is True for all values x in the iterable. |
| `anext(…)` | Return the next item from the async iterator. |
| `any(iterable, /)` | Return True if bool(x) is True for any x in the iterable. |
| `ArithmeticError(/, *args, **kwargs)` | Base class for arithmetic errors. |
| `ascii(obj, /)` | Return an ASCII-only representation of an object. |
| `AssertionError(/, *args, **kwargs)` | Assertion failed. |
| `AttributeError(/, *args, **kwargs)` | Attribute not found. |
| `BaseException(/, *args, **kwargs)` | Common base class for all exceptions |
| `BaseExceptionGroup(/, *args, **kwargs)` | A combination of multiple unrelated exceptions. |
| `bin(number, /)` | Return the binary representation of an integer. |
| `BlockingIOError(/, *args, **kwargs)` | I/O operation would block. |
| `bool(/, *args, **kwargs)` | Returns True when the argument is true, False otherwise. The builtins True and False are the only two instances of the class bool. The class bool is a subclass of the class int, and cannot be subclassed. |
| `breakpoint(*a, **k)` | function |
| `BrokenPipeError(/, *args, **kwargs)` | Broken pipe. |
| `BufferError(/, *args, **kwargs)` | Buffer error. |
| `bytearray(/, *args, **kwargs)` | bytearray(iterable_of_ints) -> bytearray bytearray(string, encoding[, errors]) -> bytearray bytearray(bytes_or_buffer) -> mutable copy of bytes_or_buffer bytearray(int) -> bytes array of size given by the parameter initi |
| `bytes(/, *args, **kwargs)` | bytes(iterable_of_ints) -> bytes bytes(string, encoding[, errors]) -> bytes bytes(bytes_or_buffer) -> immutable copy of bytes_or_buffer bytes(int) -> bytes object of size given by the parameter initialized with null byte |
| `BytesWarning(/, *args, **kwargs)` | Base class for warnings about bytes and buffer related problems, mostly related to conversion from str or comparing to str. |
| `callable(obj, /)` | Return whether the object is callable (i.e., some kind of function). |
| `ChildProcessError(/, *args, **kwargs)` | Child process error. |
| `chr(i, /)` | Return a Unicode string of one character with ordinal i; 0 <= i <= 0x10ffff. |
| `classmethod(/, *args, **kwargs)` | Convert a function to be a class method. |
| `compile(*a, **k)` | function |
| `complex(/, *args, **kwargs)` | Create a complex number from a string or numbers. |
| `ConnectionAbortedError(/, *args, **kwargs)` | Connection aborted. |
| `ConnectionError(/, *args, **kwargs)` | Connection error. |
| `ConnectionRefusedError(/, *args, **kwargs)` | Connection refused. |
| `ConnectionResetError(/, *args, **kwargs)` | Connection reset. |
| `delattr(obj, name)` | function |
| `DeprecationWarning(/, *args, **kwargs)` | Base class for warnings about deprecated features. |
| `dict(/, *args, **kwargs)` | dict() -> new empty dictionary dict(mapping) -> new dictionary initialized from a mapping object's (key, value) pairs dict(iterable) -> new dictionary initialized as if via: d = {} for k, v in iterable: d[k] = v dict(**k |
| `dir(*obj)` | function |
| `divmod(x, y, /)` | Return the tuple (x//y, x%y). Invariant: div*y + mod == x. |
| `Ellipsis` | value |
| `EncodingWarning(/, *args, **kwargs)` | Base class for warnings about encodings. |
| `enumerate(/, *args, **kwargs)` | Return an enumerate object. |
| `EnvironmentError(/, *args, **kwargs)` | Base class for I/O related errors. |
| `EOFError(/, *args, **kwargs)` | Read beyond end of file. |
| `eval(*a, **k)` | function |
| `Exception(/, *args, **kwargs)` | Common base class for all non-exit exceptions. |
| `ExceptionGroup(/, *args, **kwargs)` | type |
| `exec(*a, **k)` | function |
| `exit(code=None)` | function |
| `FileExistsError(/, *args, **kwargs)` | File already exists. |
| `FileNotFoundError(/, *args, **kwargs)` | File not found. |
| `filter(/, *args, **kwargs)` | Return an iterator yielding those items of iterable for which function(item) is true. If function is None, return the items that are true. |
| `float(/, *args, **kwargs)` | Convert a string or number to a floating-point number, if possible. |
| `FloatingPointError(/, *args, **kwargs)` | Floating-point operation failed. |
| `format(value, format_spec='', /)` | Return type(value).__format__(value, format_spec) |
| `frozenset(/, *args, **kwargs)` | Build an immutable unordered collection of unique elements. |
| `FutureWarning(/, *args, **kwargs)` | Base class for warnings about constructs that will change semantically in the future. |
| `GeneratorExit(/, *args, **kwargs)` | Request that a generator exit. |
| `getattr(obj, name, *default)` | function |
| `globals(*a, **k)` | function |
| `hasattr(obj, name)` | function |
| `hash(obj, /)` | Return the hash value for the given object. |
| `hex(number, /)` | Return the hexadecimal representation of an integer. |
| `id(obj, /)` | Return the identity of an object. |
| `ImportError(/, *args, **kwargs)` | Import can't find module, or can't find name in module. |
| `ImportWarning(/, *args, **kwargs)` | Base class for warnings about probable mistakes in module imports |
| `IndentationError(/, *args, **kwargs)` | Improper indentation. |
| `IndexError(/, *args, **kwargs)` | Sequence index out of range. |
| `input(prompt='')` | function |
| `int(/, *args, **kwargs)` | int([x]) -> integer int(x, base=10) -> integer |
| `InterruptedError(/, *args, **kwargs)` | Interrupted by signal. |
| `IOError(/, *args, **kwargs)` | Base class for I/O related errors. |
| `IsADirectoryError(/, *args, **kwargs)` | Operation doesn't work on directories. |
| `isinstance(obj, class_or_tuple, /)` | Return whether an object is an instance of a class or of a subclass thereof. |
| `issubclass(cls, class_or_tuple, /)` | Return whether 'cls' is derived from another class or is the same class. |
| `iter(…)` | iter(iterable) -> iterator iter(callable, sentinel) -> iterator |
| `KeyboardInterrupt(/, *args, **kwargs)` | Program interrupted by user. |
| `KeyError(/, *args, **kwargs)` | Mapping key not found. |
| `len(obj, /)` | Return the number of items in a container. |
| `list(/, *args, **kwargs)` | Built-in mutable sequence. |
| `locals(*a, **k)` | function |
| `LookupError(/, *args, **kwargs)` | Base class for lookup errors. |
| `map(/, *args, **kwargs)` | Make an iterator that computes the function using arguments from each of the iterables. Stops when the shortest iterable is exhausted. |
| `max(…)` | max(iterable, *[, default=obj, key=func]) -> value max(arg1, arg2, *args, *[, key=func]) -> value |
| `MemoryError(/, *args, **kwargs)` | Out of memory. |
| `memoryview(/, *args, **kwargs)` | Create a new memoryview object which references the given object. |
| `min(…)` | min(iterable, *[, default=obj, key=func]) -> value min(arg1, arg2, *args, *[, key=func]) -> value |
| `ModuleNotFoundError(/, *args, **kwargs)` | Module not found. |
| `NameError(/, *args, **kwargs)` | Name not found globally. |
| `next(…)` | next(iterator[, default]) |
| `NotADirectoryError(/, *args, **kwargs)` | Operation only works on directories. |
| `NotImplemented` | value |
| `NotImplementedError(/, *args, **kwargs)` | Method or function hasn't been implemented yet. |
| `object(/, *args, **kwargs)` | The base class of the class hierarchy. |
| `oct(number, /)` | Return the octal representation of an integer. |
| `open(file, mode='r', *args, **kwargs)` | function |
| `ord(character, /)` | Return the ordinal value of a character. |
| `OSError(/, *args, **kwargs)` | Base class for I/O related errors. |
| `OverflowError(/, *args, **kwargs)` | Result too large to be represented. |
| `PendingDeprecationWarning(/, *args, **kwargs)` | Base class for warnings about features which will be deprecated in the future. |
| `PermissionError(/, *args, **kwargs)` | Not enough permissions. |
| `pow(base, exp, mod=None)` | Equivalent to base**exp with 2 arguments or base**exp % mod with 3 arguments |
| `print(*args, sep=' ', end='\n', file=None, flush=False)` | Prints the values to a stream, or to sys.stdout by default. |
| `ProcessLookupError(/, *args, **kwargs)` | Process not found. |
| `property(/, *args, **kwargs)` | Property attribute. |
| `pyfrom(module, name)` | `from module import name`: an attribute of the module, or its submodule. |
| `pyimport(name)` | pyimport() for Python-exact files: pure standard modules (and their submodules) need no permission; with --allow-python=all every module is the real module object. |
| `pystar(module, *names)` | `from module import *`: the public names, in one go. |
| `PythonFinalizationError(/, *args, **kwargs)` | Operation blocked during Python finalization. |
| `quit(code=None)` | function |
| `range(/, *args, **kwargs)` | range(stop) -> range object range(start, stop[, step]) -> range object |
| `RecursionError(/, *args, **kwargs)` | Recursion limit exceeded. |
| `ReferenceError(/, *args, **kwargs)` | Weak ref proxy used after referent went away. |
| `repr(obj, /)` | Return the canonical string representation of the object. |
| `ResourceWarning(/, *args, **kwargs)` | Base class for warnings about resource usage. |
| `reversed(/, *args, **kwargs)` | Return a reverse iterator over the values of the given sequence. |
| `round(number, ndigits=None)` | Round a number to a given precision in decimal digits. |
| `RuntimeError(/, *args, **kwargs)` | Unspecified run-time error. |
| `RuntimeWarning(/, *args, **kwargs)` | Base class for warnings about dubious runtime behavior. |
| `set(/, *args, **kwargs)` | Build an unordered collection of unique elements. |
| `setattr(obj, name, value)` | function |
| `slice(/, *args, **kwargs)` | slice(stop) slice(start, stop[, step]) |
| `sorted(iterable, /, *, key=None, reverse=False)` | Return a new list containing all items from the iterable in ascending order. |
| `staticmethod(/, *args, **kwargs)` | Convert a function to be a static method. |
| `StopAsyncIteration(/, *args, **kwargs)` | Signal the end from iterator.__anext__(). |
| `StopIteration(/, *args, **kwargs)` | Signal the end from iterator.__next__(). |
| `str(/, *args, **kwargs)` | str(object='') -> str str(bytes_or_buffer[, encoding[, errors]]) -> str |
| `sum(iterable, /, start=0)` | Return the sum of a 'start' value (default: 0) plus an iterable of numbers |
| `super(/, *args, **kwargs)` | super() -> same as super(__class__, <first argument>) super(type) -> unbound super object super(type, obj) -> bound super object; requires isinstance(obj, type) super(type, type2) -> bound super object; requires issubcla |
| `SyntaxError(/, *args, **kwargs)` | Invalid syntax. |
| `SyntaxWarning(/, *args, **kwargs)` | Base class for warnings about dubious syntax. |
| `SystemError(/, *args, **kwargs)` | Internal error in the Python interpreter. |
| `SystemExit(/, *args, **kwargs)` | Request to exit from the interpreter. |
| `TabError(/, *args, **kwargs)` | Improper mixture of spaces and tabs. |
| `TimeoutError(/, *args, **kwargs)` | Timeout expired. |
| `tuple(/, *args, **kwargs)` | Built-in immutable sequence. |
| `type(/, *args, **kwargs)` | type |
| `TypeError(/, *args, **kwargs)` | Inappropriate argument type. |
| `UnboundLocalError(/, *args, **kwargs)` | Local name referenced but not bound to a value. |
| `UnicodeDecodeError(/, *args, **kwargs)` | Unicode decoding error. |
| `UnicodeEncodeError(/, *args, **kwargs)` | Unicode encoding error. |
| `UnicodeError(/, *args, **kwargs)` | Unicode related error. |
| `UnicodeTranslateError(/, *args, **kwargs)` | Unicode translation error. |
| `UnicodeWarning(/, *args, **kwargs)` | Base class for warnings about Unicode related problems, mostly related to conversion problems. |
| `UserWarning(/, *args, **kwargs)` | Base class for warnings generated by user code. |
| `ValueError(/, *args, **kwargs)` | Inappropriate argument value (of correct type). |
| `vars(*obj)` | function |
| `Warning(/, *args, **kwargs)` | Base class for warning categories. |
| `ZeroDivisionError(/, *args, **kwargs)` | Second argument to a division or modulo operation was zero. |
| `zip(/, *args, **kwargs)` | The zip object yields n-length tuples, where n is the number of iterables passed as positional arguments to zip(). The i-th element in every tuple comes from the i-th iterable argument to zip(). This continues until the  |

### `use python` — 5 names

| Name | What it is |
|---|---|
| `from_py(x)` | Python/JSON value -> Apsi value ({value, unit} dicts become Quantities). |
| `from_torch(t)` | function |
| `pyimport(name)` | function |
| `to_py(x)` | Apsi value -> plain Python/JSON-friendly value (units become {value, unit}). |
| `to_torch(x, device=None)` | function |

### `use quantum` — 38 names

| Name | What it is |
|---|---|
| `anticommutator(A, B)` | function |
| `apply(state, gate, targets)` | Apply a k-qubit gate to the given target qubits (list, in the gate's qubit order) of an n-qubit state. |
| `bell(k=0)` | Bell states: 0 Φ+, 1 Φ−, 2 Ψ+, 3 Ψ−. |
| `bloch(state)` | Bloch vector [x, y, z] of a single-qubit state or density matrix. |
| `bra(bits)` | function |
| `commutator(A, B)` | function |
| `concurrence(rho)` | Wootters concurrence of a two-qubit state (0 = separable, 1 = maximally entangled). |
| `controlled(gate)` | Controlled version of a gate (control = first qubit). |
| `dagger(A)` | function |
| `density(state)` | Density matrix \|ψ⟩⟨ψ\|. |
| `entropy(rho, base=2)` | Von Neumann entropy −Tr ρ log ρ (bits by default). |
| `evolve1d(psi0, V, x, dt, steps, mass=9.1093837015e-31, hbar=1.054571817e-34, every=1)` | Time evolution with the second-order split-operator method (periodic box, norm-preserving). Returns record(t, psi[frame, i], norm, x_mean). |
| `expect_p(psi, x, hbar=1.054571817e-34)` | ⟨p⟩ = ∫ ψ* (−iħ ∂ψ/∂x) dx (spectral derivative). |
| `expect_x(psi, x)` | function |
| `expm(A)` | Matrix exponential (scaling and squaring with a Padé(13) approximant). |
| `expm_hermitian(H, t=1.0, hbar=1.0)` | exp(−i H t / ħ) for Hermitian H (exact, by diagonalization). |
| `fidelity(a, b)` | Fidelity between two pure states \|⟨a\|b⟩\|², or two density matrices (Uhlmann). |
| `gates` | A lightweight named-field container: record(W=..., b=...). |
| `gaussian_packet(x, x0, sigma, k0=0.0)` | Normalized Gaussian wave packet centred at x0 with width sigma and mean wavenumber k0. |
| `ghz(n)` | GHZ state (\|0…0⟩ + \|1…1⟩)/√2. |
| `hydrogen_energy(n, Z=1)` | Bohr energy levels −13.605693 eV · Z²/n² (infinite nuclear mass). |
| `hydrogen_radial(n, l, r)` | Normalized radial function R_nl(r) of hydrogen, r in Bohr radii (∫ R² r² dr = 1). |
| `ket(bits)` | Computational basis state, e.g. ket("010") or ket([0, 1, 0]). |
| `kron(*mats)` | Tensor (Kronecker) product of any number of matrices or vectors. |
| `measure(state, shots=1000, seed=None)` | Sample measurement outcomes: record of bitstring -> count. |
| `partial_trace(rho, keep, n=None)` | Reduced density matrix of the qubits in `keep` (list of indices). |
| `pauli()` | Pauli matrices: record(I, X, Y, Z). |
| `phase(phi)` | function |
| `probabilities(state)` | Measurement probabilities in the computational basis. |
| `probability(psi, x, a, b)` | ∫_a^b \|ψ\|² dx. |
| `purity(rho)` | function |
| `rx(theta)` | function |
| `ry(theta)` | function |
| `rz(theta)` | function |
| `schrodinger1d(V, x, mass=9.1093837015e-31, n=5, hbar=1.054571817e-34)` | Lowest n bound states of −ħ²/(2m) ψ'' + V ψ = E ψ on the grid x (ψ = 0 beyond the ends; fourth-order finite differences). V: function of x or values on the grid. Returns record(energies, energies_ev, psi[n, len(x)], x). |
| `spin_ops(s=0.5, hbar=1.0)` | Spin operators for spin s (in units of ħ by default): record(x, y, z, plus, minus, squared). |
| `time_evolution(H, psi0, times, hbar=1.0)` | ψ(t) = exp(−iHt/ħ) ψ0 for every t in `times` -> array [len(times), dim]. |
| `uncertainty(psi, x, hbar=1.054571817e-34)` | record(dx, dp, product) — standard deviations of position and momentum. |

### `use random` — 5 names

| Name | What it is |
|---|---|
| `permutation(n, seed=None)` | function |
| `rand(*shape, seed=None)` | function |
| `randint(lo, hi, *shape, seed=None)` | function |
| `randn(*shape, seed=None)` | function |
| `seed(n)` | function |

### `use rl` — 25 names

| Name | What it is |
|---|---|
| `bandit(arms=10, n=1, seed=None)` | function |
| `beam_search(step, start, beam_width=4, max_len=20, eos=None)` | Beam search: step(sequence) returns log-probabilities over the next token (1-D). Returns the best sequences as a list of record(tokens, score), best first. |
| `best_of_n(candidates, score)` | The candidate with the highest score(candidate), with all scores. |
| `cartpole(n=1, seed=None, max_steps=500)` | function |
| `categorical_entropy(logits)` | function |
| `categorical_log_prob(logits, actions)` | log π(a) for integer actions (differentiable in logits). |
| `categorical_sample(logits, seed=None, rng=None)` | Sample actions from softmax(logits) (Gumbel-max). |
| `discounted_returns(rewards, dones, gamma=0.99, last_value=None)` | Discounted returns of time-major rewards [T] or [T, N]; done[t] ends the episode after step t. |
| `dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1, label_smoothing=0.0)` | Direct Preference Optimization. Inputs are summed log-probabilities of the chosen / rejected responses under the policy and the frozen reference model. Returns record(loss, chosen_rewards, rejected_rewards, accuracy). |
| `dqn(env, model, params, total_steps=50000, buffer_size=50000, batch_size=64, lr=0.001, gamma=0.99, learning_…)` | Deep Q-learning: replay buffer, ε-greedy exploration, target network, (double) DQN target, Huber loss. model(params, obs) returns Q-values [batch, actions]. Returns record(params, episode_returns). |
| `evaluate(env, policy, episodes=10, seed=0)` | Run policy(obs) -> actions until `episodes` episodes have finished; returns their mean and all returns. |
| `gae(rewards, values, dones, last_values, gamma=0.99, lam=0.95)` | Generalized Advantage Estimation (time-major [T, N]). Returns record(advantages, returns). |
| `gaussian_entropy(log_std)` | function |
| `gaussian_log_prob(mean, log_std, actions)` | Sum over action dimensions of log N(a; mean, exp(log_std)²) (differentiable). |
| `gridworld(size=5, n=1, seed=None)` | function |
| `grpo_advantages(rewards, group_size, eps=1e-06)` | GRPO: rewards [B·G] of G completions per prompt -> (r − group mean) / (group std + eps). |
| `ipo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, tau=0.1)` | Identity Preference Optimization: mean((h − 1/(2τ))²) with h the log-ratio difference. |
| `kl_penalized_rewards(rewards, logp, ref_logp, beta=0.05)` | Per-token rewards for RLHF with PPO: −β·(log π − log π_ref) at every token plus the sequence reward on the last token. logp / ref_logp: [B, T]; rewards: [B]. Returns [B, T]. |
| `mcts(state, legal_actions, apply, is_terminal, reward, simulations=1000, c=1.41, rollout_depth=200, seed=0, t…)` | Monte-Carlo tree search (UCT). Functions: legal_actions(s) -> list, apply(s, a) -> new state, is_terminal(s) -> bool, reward(s) -> value from the point of view of the player who moved last when two_player=true (otherwise |
| `ppo(env, model, params, total_steps=100000, rollout=128, epochs=4, minibatch=256, lr=0.0003, gamma=0.99, lam…)` | Proximal Policy Optimization (clipped surrogate, GAE, value and entropy terms, gradient clipping). `model(params, obs)` returns [logits, value]. Returns record(params, episode_returns, updates). |
| `ppo_clip_loss(logp, old_logp, advantages, clip=0.2, mask=None)` | Clipped PPO surrogate for token log-probs [B, T] (mask: 1 for response tokens). |
| `preference_loss(r_chosen, r_rejected, margin=0.0)` | Bradley–Terry reward-model loss: −log σ(r_chosen − r_rejected − margin), averaged. |
| `reinforce(env, model, params, episodes=500, lr=0.01, gamma=0.99, seed=0, max_steps=1000)` | REINFORCE with a running-mean baseline on a single environment (n=1). model(params, obs) -> logits. |
| `sample_token(logits, temperature=1.0, top_k=None, top_p=None, seed=None)` | Sample from logits [..., V] with temperature, top-k and nucleus (top-p) filtering; temperature 0 = greedy. |
| `sequence_log_prob(logits, tokens, mask=None)` | Σ_t log softmax(logits[b, t])[tokens[b, t]] for each sequence (logits [B, T, V], tokens [B, T]). |

### `use rt` — 20 names

| Name | What it is |
|---|---|
| `backend(name=None)` | rt.backend() → the current frame backend; rt.backend("cpu" \| "gpu" \| "auto"). "auto" picks the GPU only for frames large enough to beat the copy to the card (≥ 1600×900); "gpu" without a usable card stays on the CPU and  |
| `box_blur(img, r=1)` | Mean filter of radius r (cost independent of r). |
| `camera(index=0, width=None, height=None, fps=None, fourcc=None)` | A camera through OpenCV (pip install opencv-python); needs --allow-python. `width` may be a size name or "WxH" ("fhd", "1600x900"). |
| `diff(a, b, threshold=0.0)` | \|a - b\| per pixel (largest channel for colour); with a threshold, a boolean mask of pixels that changed more than it. |
| `downsample(img, k=2)` | Average k×k blocks (fast, no aliasing from skipping pixels). |
| `edges(img, low=50, high=150)` | Edge map (Canny with OpenCV; gradient magnitude above `high` otherwise), uint8 0/255. |
| `ema(prev, new, alpha=0.2)` | Exponential moving average (smooths noise while staying responsive). |
| `fps_meter()` | function |
| `gray(rgb)` | Luminance of an RGB(A) image (BT.601). uint8 in → uint8 out (fast path); other types → float32. |
| `has_opencv` | OpenCV, imported on first use (it takes ~20 ms to load); falsy when it is not installed. |
| `loop(hz, step, seconds=None, steps=None, catch_up=False, on_error='raise')` | Call step(tick, t) `hz` times per second for `seconds` (or `steps` ticks). A tick that cannot start on time is counted as missed and skipped (catch_up=true runs it late instead), so the rhythm never drifts. on_error="ski |
| `motion(prev, cur, threshold=25.0, step=None)` | Did something move? record(moved, fraction, cx, cy, box): changed share of pixels, the centre of the change and its bounding box [x0, y0, x1, y1] in full-frame pixels. Works on gray or colour frames of any size; above 64 |
| `resize(img, width, height=None)` | Resize to width×height (area averaging when shrinking — no aliasing; smooth when enlarging). resize(img, 0.5) scales by a factor. |
| `scheduler()` | function |
| `size(width, height=None)` | (width, height) from 1600, 900 / "1600x900" / "fhd" / "1080p" / (w, h). |
| `sizes` | value |
| `synthetic(width=640, height=None, fps=120.0, noise=False)` | function |
| `threshold(img, t, high=255)` | Pixels above t become `high`, the rest 0 (uint8). |
| `ticks(hz, seconds=None, steps=None)` | A for-loop that runs at `hz`: `for tick in rt.ticks(1000, seconds=2):`. Yields the tick number. |
| `video(source, process, fps=None, seconds=None, frames=None, adapt=False, min_scale=0.25, on_error='skip', gray…)` | Run `process(frame, info)` on the newest frame from `source` until `seconds` or `frames` is reached. |

### `use scale` — 4 names

| Name | What it is |
|---|---|
| `gpu_table()` | function |
| `inference(params, gpu='H100', precision='bf16', context=8192, batch=1, kv_heads=None, layers=None, width=None, act…)` | Memory to serve a model: weights + KV cache, and the GPUs needed. |
| `number(x)` | '21B' -> 2.1e10, '1T' -> 1e12, '15T' tokens, or a plain number. |
| `plan(params, tokens=None, gpu='H100', gpus=None, days=None, active=None, layers=None, width=None, heads=None,…)` | Plan a training run. Returns a record with the layout, memory per GPU, GPU count and duration. |

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

### `use signal` — 24 names

| Name | What it is |
|---|---|
| `bandpass(x, low, high, fs, order=4)` | Zero-phase Butterworth band-pass between low and high. |
| `bandstop(x, low, high, fs, order=4)` | Zero-phase Butterworth band-stop. |
| `blackman(n, sym=True)` | function |
| `butter(order, cutoff, kind='lowpass', fs=2.0)` | Butterworth filter as second-order sections. cutoff in Hz (with fs) — a [low, high] pair for band filters. Default fs=2 means cutoff is a fraction of the Nyquist frequency, as in SciPy. |
| `convolve(a, b, mode='full')` | Linear convolution ("full", "same" or "valid"), FFT-based for long inputs. |
| `correlate(a, b, mode='full')` | Cross-correlation (as numpy.correlate / scipy.signal.correlate for real data). |
| `detrend(x, kind='linear')` | Remove the mean ("constant") or the least-squares line ("linear"). |
| `envelope(x)` | Amplitude envelope \|analytic signal\|. |
| `filtfilt(sos, x, padlen=None)` | Zero-phase filtering: forward and backward, odd-extended edges, steady-state start (SciPy compatible). |
| `find_peaks(x, height=None, distance=None, prominence=None)` | Local maxima (flat tops: the middle sample), filtered by minimum height, minimum distance in samples (taller peaks win) and minimum prominence — the same rules as scipy.signal.find_peaks. |
| `freqz(sos, n=512, fs=6.283185307179586)` | Frequency response: record(f, h (complex), magnitude, magnitude_db, phase). |
| `hamming(n, sym=True)` | function |
| `hann(n, sym=True)` | Hann window. |
| `highpass(x, cutoff, fs, order=4)` | Zero-phase Butterworth high-pass. |
| `hilbert(x)` | Analytic signal x + i·H[x] (FFT method). |
| `kaiser(n, beta=8.6)` | Kaiser window. |
| `lowpass(x, cutoff, fs, order=4)` | Zero-phase Butterworth low-pass. |
| `moving_average(x, n)` | Centered moving average over n samples (edges use the available samples). |
| `notch(x, freq, fs, q=30.0)` | Remove one frequency (e.g. 50/60 Hz mains) with a second-order IIR notch, zero phase. |
| `resample(x, num)` | Change the number of samples with the Fourier method (periodic signals; as scipy.signal.resample). |
| `savgol(x, window, order)` | Savitzky–Golay smoothing (least-squares polynomial of `order` over `window` samples; polynomial fit at the edges, like SciPy's mode="interp"). |
| `sosfilt(sos, x)` | Causal filtering with second-order sections (starts from rest). |
| `spectrogram(x, fs=1.0, nperseg=256, noverlap=None)` | Short-time power spectrum (Hann window, density scaling): record(f, t, power[f, t]). |
| `welch(x, fs=1.0, nperseg=256, noverlap=None, detrend='constant')` | Power spectral density by Welch's method (Hann window, one-sided, density scaling). |

### `use special` — 31 names

| Name | What it is |
|---|---|
| `assoc_legendre(l, m, x)` | Associated Legendre function P_l^m(x) with the Condon–Shortley phase (as in physics texts). |
| `besseli(n, x)` | Modified Bessel function of the first kind I_n(x), integer n. |
| `besselj(n, x)` | Bessel function of the first kind J_n(x), integer n. |
| `beta(a, b)` | B(a, b) = Γ(a)Γ(b)/Γ(a+b). |
| `betainc(a, b, x)` | Regularized incomplete beta I_x(a, b). |
| `betaincinv(a, b, p)` | x with I_x(a, b) = p. |
| `chebyshev(n, x)` | Chebyshev polynomial of the first kind T_n(x). |
| `comb(n, k)` | Binomial coefficient C(n, k) (exact for integers). |
| `digamma(x)` | ψ(x) = Γ'(x)/Γ(x). |
| `ellipe(m)` | Complete elliptic integral of the second kind E(m), m = k² ≤ 1 (AGM). |
| `ellipk(m)` | Complete elliptic integral of the first kind K(m), m = k² < 1 (AGM). |
| `erf(x)` | function |
| `erfc(x)` | function |
| `erfcinv(y)` | Inverse complementary error function. |
| `erfinv(y)` | Inverse error function. |
| `expint(x)` | Exponential integral E1(x) = ∫_x^∞ e^{-t}/t dt, x > 0. |
| `factorial(n)` | n! (exact integer for integers). |
| `gamma(x)` | Γ(x). |
| `gammainc(a, x)` | Regularized lower incomplete gamma P(a, x). |
| `gammaincc(a, x)` | Regularized upper incomplete gamma Q(a, x) = 1 − P(a, x). |
| `gammaincinv(a, p)` | x with P(a, x) = p. |
| `hermite(n, x)` | Physicists' Hermite polynomial H_n(x) (H_1 = 2x). |
| `hermite_e(n, x)` | Probabilists' Hermite polynomial He_n(x) (He_1 = x). |
| `laguerre(n, x, alpha=0.0)` | Generalized Laguerre polynomial L_n^(α)(x). |
| `lbeta(a, b)` | function |
| `legendre(n, x)` | Legendre polynomial P_n(x). |
| `lgamma(x)` | ln\|Γ(x)\|. |
| `perm(n, k)` | Number of ordered selections n!/(n-k)! (exact for integers). |
| `sinc(x)` | sin(πx)/(πx) (normalized). |
| `sph_harm(l, m, theta, phi)` | Spherical harmonic Y_l^m(θ, φ) (θ polar, φ azimuth), orthonormal, Condon–Shortley phase. Complex. |
| `zeta(s)` | Riemann zeta ζ(s) for real s > 1 (Euler–Maclaurin). |

### `use stats` — 44 names

| Name | What it is |
|---|---|
| `adjust_pvalues(p, method='holm')` | Adjusted p-values: "bonferroni", "holm", "bh" (Benjamini–Hochberg FDR) or "by" (Benjamini–Yekutieli). |
| `anova_oneway(*groups)` | One-way ANOVA: F, p, degrees of freedom, η² and ω². |
| `beta_dist(a, b)` | Beta distribution on [0, 1]. |
| `binom_test(k, n, p=0.5, alternative='two-sided')` | Exact binomial test. |
| `binomial(n, p)` | Binomial distribution: successes in n trials with probability p. |
| `bootstrap_ci(data, statistic=None, n=10000, level=0.95, seed=0)` | Percentile bootstrap confidence interval of statistic(data) (default: the mean). |
| `chi2(df)` | Chi-squared distribution. |
| `chi2_contingency(table, correction=True)` | Chi-squared test of independence for an r×c table (Yates' correction for 2×2 by default). |
| `chi2_gof(observed, expected=None, ddof=0)` | Chi-squared goodness-of-fit test (expected counts default to uniform). |
| `cohen_d(a, b)` | Cohen's d with the pooled standard deviation. |
| `corr_matrix(X)` | Pearson correlation matrix of the columns of X. |
| `cov_matrix(X)` | Sample covariance matrix (n−1) of the columns of X. |
| `describe(x)` | n, mean, std (n−1), sem, min, quartiles, max, skewness, excess kurtosis and a 95 % CI of the mean. |
| `exponential(rate=1.0)` | Exponential distribution with the given rate λ. |
| `f_dist(d1, d2)` | F distribution (Fisher–Snedecor). |
| `fisher_exact(table, alternative='two-sided')` | Fisher's exact test for a 2×2 table. Returns the sample odds ratio and the exact p-value. |
| `gamma_dist(shape, scale=1.0)` | Gamma distribution (shape k, scale θ). |
| `hedges_g(a, b)` | Hedges' g (small-sample corrected d). |
| `jarque_bera(x)` | Jarque–Bera normality test. |
| `kendalltau(x, y)` | Kendall's τ-b with the asymptotic p-value (ties handled). |
| `kruskal(*groups)` | Kruskal–Wallis H test (tie-corrected). |
| `ks_2samp(a, b, method='auto')` | Two-sample Kolmogorov–Smirnov test. method "exact" (default when m·n ≤ 10 000), or "asymptotic". |
| `logit(X, y, add_intercept=True, names=None, level=0.95, max_iter=100)` | Logistic regression by Newton–Raphson (IRLS): coefficients, SE, z, p, CI, odds ratios with CI, log-likelihood, AIC and McFadden's pseudo-R². |
| `lognormal(mu=0.0, sigma=1.0)` | Log-normal: log(X) ~ N(mu, sigma²). |
| `mannwhitneyu(a, b, alternative='two-sided', continuity=True)` | Mann–Whitney U test (normal approximation with tie correction). U is for sample a. |
| `mean_ci(x, level=0.95)` | Confidence interval of the mean (t distribution). |
| `normal(mu=0.0, sigma=1.0)` | Normal distribution N(mu, sigma²). |
| `ols(X, y, add_intercept=True, names=None, level=0.95)` | Ordinary least squares: coefficients with standard errors, t, p and confidence intervals; R², adjusted R², F test, residuals and a predict(X) function. |
| `pearsonr(x, y, alternative='two-sided', level=0.95)` | Pearson correlation with p-value (t test) and Fisher-z confidence interval. |
| `permutation_test(a, b, statistic=None, n=10000, seed=0, alternative='two-sided')` | Two-sample permutation test of statistic(a, b) (default: difference of means). |
| `poisson(rate)` | Poisson distribution with mean `rate`. |
| `power_ttest(effect, n, alpha=0.05, alternative='two-sided', kind='two-sample')` | Power of a t-test for a standardized effect size (Cohen's d), n per group (exact noncentral t). |
| `proportion_ci(k, n, level=0.95, method='wilson')` | Confidence interval of a proportion k/n: "wilson" (default) or "clopper-pearson" (exact). |
| `quantile(x, q)` | Quantiles (linear interpolation, as numpy/R type 7). |
| `sample_size_ttest(effect, power=0.8, alpha=0.05, alternative='two-sided', kind='two-sample')` | Smallest n (per group for two-sample) that reaches the requested power. |
| `sem(x)` | function |
| `spearmanr(x, y, alternative='two-sided')` | Spearman rank correlation (t approximation for the p-value, as SciPy). |
| `student_t(df, loc=0.0, scale=1.0)` | Student's t distribution with df degrees of freedom. |
| `ttest_1samp(x, mu=0.0, alternative='two-sided', level=0.95)` | One-sample t-test of mean(x) = mu. |
| `ttest_ind(a, b, equal_var=False, alternative='two-sided', level=0.95)` | Two-sample t-test (Welch by default; equal_var=true for Student's). |
| `ttest_rel(a, b, alternative='two-sided', level=0.95)` | Paired t-test. |
| `uniform(low=0.0, high=1.0)` | Continuous uniform distribution on [low, high]. |
| `wilcoxon(x, y=None, alternative='two-sided', continuity=False)` | Wilcoxon signed-rank test (normal approximation, zeros dropped, tie correction). |
| `zscore(x)` | function |

### `use sys` — 8 names

| Name | What it is |
|---|---|
| `cpu_threads()` | function |
| `device()` | "cuda:0" when a GPU meeting Apsi's target tier is usable through CuPy, otherwise "cpu". |
| `gpu_available()` | function |
| `gpu_info()` | GPU 0: name, compute capability, memory. Works with CuPy, or with just the NVIDIA driver. |
| `hardware()` | function |
| `to_cpu(x)` | function |
| `to_device(x, dev=None)` | function |
| `to_gpu(x)` | function |

### `use tokens` — 12 names

| Name | What it is |
|---|---|
| `best(value, model='gpt-4o')` | Try the transforms that apply and return (text, method) with the fewest tokens; unpack with restore(). |
| `count(text, model='gpt-4o')` | Tokens in text: exact with tiktoken installed (o200k / cl100k), otherwise an estimate (±10–15 %). |
| `dedupe(text, min_chars=24, min_count=2)` | Lines that repeat are written once in a header and referenced as §N where they occur. |
| `diff(old, new)` | Only what changed, as line operations: `=N` keep N lines, `-N` drop N lines, `+text` add a line. |
| `exact()` | True when count() is exact (tiktoken installed). |
| `expand(text)` | function |
| `pack(value)` | Compact, exactly reversible text for a JSON value. A list of records with the same keys becomes a table: `@table ["id","name"]` then one line per record (`1,"ann"`). Anything else is minified JSON. |
| `pack_jsonl(text)` | JSON Lines → pack() of the records (exact for the parsed values). |
| `patch(old, delta)` | function |
| `restore(text, method)` | function |
| `savings(before, after, model='gpt-4o')` | function |
| `unpack(text)` | function |

### `use torch` — 138 names

| Name | What it is |
|---|---|
| `Adagrad(params=None, lr=0.01, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0, eps=1e-10, foreach=None, maximize=False, differentiable=False, fused=None)` | type |
| `Adam(params=None, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False, maximize=False, foreach=None, capturable=False, differentiable=False, fused=None, decoupled_weight_decay=False)` | type |
| `Adamax(params=None, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, **_)` | type |
| `AdamW(params=None, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)` | type |
| `AdaptiveAvgPool1d(output_size)` | function |
| `AdaptiveAvgPool2d(output_size)` | function |
| `AdaptiveMaxPool1d(output_size, return_indices=False)` | function |
| `AdaptiveMaxPool2d(output_size, return_indices=False)` | function |
| `AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)` | function |
| `AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)` | function |
| `BatchNorm1d(*a, **k)` | function |
| `BatchNorm2d(*a, **k)` | function |
| `BCELoss(*args, **kw)` | function |
| `BCEWithLogitsLoss(*args, **kw)` | function |
| `Bilinear(in1_features, in2_features, out_features, bias=True)` | type |
| `binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction='mean')` | function |
| `binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)` | PyTorch's stable form: (1−y)·x + (1 + (pw−1)·y)·(log(1+e^(−\|x\|)) + max(−x, 0)). |
| `CELU(*args, inplace=False, **kw)` | function |
| `celu(x, alpha=1.0)` | function |
| `ChainedScheduler(schedulers, optimizer=None)` | type |
| `checkpoint(function, *args, use_reentrant=None, **kw)` | torch.utils.checkpoint.checkpoint(fn, *args): every differentiable value fn uses must be an argument. |
| `checkpoint_layer(layer, params, *xs, train=None, **kw)` | Apply layer(params, *xs) with gradient checkpointing (torch.utils.checkpoint): activations inside the layer are not stored but recomputed in the backward pass, which trades compute for memory. |
| `clip_grad_norm(grads, max_norm, norm_type=2.0, error_if_nonfinite=False)` | PyTorch's clip_grad_norm_: scale every gradient by max_norm / (total_norm + 1e-6) when that is below 1. |
| `clip_grad_value(grads, clip_value)` | function |
| `ConstantLR(optimizer, factor=0.3333333333333333, total_iters=5, last_epoch=-1)` | type |
| `Conv1d(*a, **k)` | function |
| `Conv2d(*a, **k)` | function |
| `cosine_similarity(x1, x2, dim=1, eps=1e-08)` | function |
| `CosineAnnealingLR(optimizer, T_max, eta_min=0.0, last_epoch=-1)` | type |
| `CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0.0, last_epoch=-1)` | type |
| `cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_s…)` | F.cross_entropy with class-index or class-probability targets, weights, ignore_index and smoothing. |
| `CrossEntropyLoss(*args, **kw)` | function |
| `CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)` | type |
| `DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, generator=None, prefetch_factor=None, persistent_workers=False, pin_memory_device='', seed=None)` | Mini-batches from a dataset (TensorDataset, or any object with __len__ / __getitem__), reshuffled every epoch when shuffle=true; batches are stacked like PyTorch's default collate. |
| `Dropout(p=0.5, inplace=False)` | type |
| `dropout(x, p=0.5, train=True, training=None, inplace=False)` | Inverted dropout (as F.dropout): zero with probability p and scale the rest by 1/(1-p) while training. |
| `ELU(*args, inplace=False, **kw)` | function |
| `elu(x, alpha=1.0)` | function |
| `Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, scale_grad_by_freq=False, sparse=False, _weight=None)` | type |
| `embedding(idx, weight, padding_idx=None)` | function |
| `ExponentialLR(optimizer, gamma, last_epoch=-1)` | type |
| `Flatten(start_dim=1, end_dim=-1)` | type |
| `flatten(x, start_dim=0, end_dim=-1)` | function |
| `GELU(*args, inplace=False, **kw)` | function |
| `gelu(x, approximate='none')` | function |
| `generate_square_subsequent_mask(sz)` | Causal mask: −inf above the diagonal (as nn.Transformer.generate_square_subsequent_mask). |
| `GLU(*args, inplace=False, **kw)` | function |
| `glu(x, dim=-1)` | function |
| `GroupNorm(num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None)` | type |
| `GRU(*a, **k)` | function |
| `Hardsigmoid(*args, inplace=False, **kw)` | function |
| `hardsigmoid(x)` | function |
| `Hardswish(*args, inplace=False, **kw)` | function |
| `hardswish(x)` | function |
| `Hardtanh(*args, inplace=False, **kw)` | function |
| `hardtanh(x, min_val=-1.0, max_val=1.0)` | function |
| `huber_loss(input, target, reduction='mean', delta=1.0, weight=None)` | function |
| `HuberLoss(*args, **kw)` | function |
| `Identity(*args, inplace=False, **kw)` | function |
| `identity(x)` | function |
| `kl_div(input, target, size_average=None, reduce=None, reduction='mean', log_target=False)` | function |
| `KLDivLoss(*args, **kw)` | function |
| `l1_loss(input, target, size_average=None, reduce=None, reduction='mean')` | function |
| `L1Loss(*args, **kw)` | function |
| `LambdaLR(optimizer, lr_lambda, last_epoch=-1)` | type |
| `layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-05)` | function |
| `LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None)` | type |
| `leaky_relu(x, negative_slope=0.01)` | function |
| `LeakyReLU(*args, inplace=False, **kw)` | function |
| `Linear(in_features, out_features, bias=True, device=None, dtype=None)` | type |
| `linear(x, weight, bias=None)` | function |
| `LinearLR(optimizer, start_factor=0.3333333333333333, end_factor=1.0, total_iters=5, last_epoch=-1)` | type |
| `load_state_dict(model, params, state, strict=True)` | Copy a PyTorch state_dict (or one saved by Apsi) into the parameter tree and the buffers of `model`. Missing or unexpected keys are errors when strict=true; shapes must match. |
| `log_softmax(x, dim=None)` | function |
| `LogSoftmax(*args, inplace=False, **kw)` | function |
| `LSTM(*a, **k)` | function |
| `MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)` | function |
| `MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)` | function |
| `Mish(*args, inplace=False, **kw)` | function |
| `mish(x)` | function |
| `Module(init, forward, name='model', parts=None)` | A model from its init(seed) and forward(p, x…, train) functions (models converted from PyTorch). |
| `ModuleList(mods=())` | function |
| `mse_loss(input, target, size_average=None, reduce=None, reduction='mean', weight=None)` | function |
| `MSELoss(*args, **kw)` | function |
| `MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)` | type |
| `MultiplicativeLR(optimizer, lr_lambda, last_epoch=-1)` | type |
| `MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)` | type |
| `NAdam(params=None, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, momentum_decay=0.004, decoupled_weight_decay=False, **_)` | type |
| `nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')` | function |
| `NLLLoss(*args, **kw)` | function |
| `normalize(x, p=2.0, dim=1, eps=1e-12)` | function |
| `one_hot(t, num_classes=-1)` | function |
| `OneCycleLR(optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, three_phase=False, last_epoch=-1)` | type |
| `pad(x, pad, mode='constant', value=0.0)` | F.pad for constant padding: pad = (last_lo, last_hi, prev_lo, prev_hi, …). |
| `parameters_count(params)` | function |
| `PolynomialLR(optimizer, total_iters=5, power=1.0, last_epoch=-1)` | type |
| `PReLU(num_parameters=1, init=0.25)` | type |
| `RAdam(params=None, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, decoupled_weight_decay=False, **_)` | type |
| `ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.0, eps=1e-08)` | type |
| `ReLU(*args, inplace=False, **kw)` | function |
| `relu(x)` | function |
| `ReLU6(*args, inplace=False, **kw)` | function |
| `relu6(x)` | function |
| `RMSNorm(normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None)` | type |
| `RMSprop(params=None, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0.0, momentum=0.0, centered=False, capturable=False, foreach=None, maximize=False, differentiable=False)` | type |
| `RNN(*a, nonlinearity='tanh', **k)` | function |
| `scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, train=True)` | softmax(Q·Kᵀ/√d + mask)·V over the last two axes (PyTorch's F.scaled_dot_product_attention). |
| `SELU(*args, inplace=False, **kw)` | function |
| `selu(x)` | function |
| `Sequential(*layers)` | type |
| `SequentialLR(optimizer, schedulers, milestones, last_epoch=-1)` | type |
| `SGD(params=None, lr=0.001, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None)` | type |
| `Sigmoid(*args, inplace=False, **kw)` | function |
| `sigmoid(x)` | function |
| `SiLU(*args, inplace=False, **kw)` | function |
| `silu(x)` | function |
| `smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean', beta=1.0)` | function |
| `SmoothL1Loss(*args, **kw)` | function |
| `Softmax(*args, inplace=False, **kw)` | function |
| `softmax(x, dim=None)` | function |
| `Softplus(*args, inplace=False, **kw)` | function |
| `softplus(x, beta=1.0, threshold=20.0)` | function |
| `Softsign(*args, inplace=False, **kw)` | function |
| `softsign(x)` | function |
| `state_dict(model, params)` | PyTorch-style state_dict of a model: parameters and buffers as one flat record of tensors ("encoder.layers.0.self_attn.in_proj_weight", "bn.running_mean", …). Saved with models.save_model it is a safetensors file PyTorch |
| `StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)` | type |
| `Tanh(*args, inplace=False, **kw)` | function |
| `tanh(x)` | function |
| `Tanhshrink(*args, inplace=False, **kw)` | function |
| `tanhshrink(x)` | function |
| `TensorDataset(*tensors)` | type |
| `Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)` | type |
| `TransformerDecoder(decoder_layer, num_layers, norm=None)` | function |
| `TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)` | type |
| `TransformerEncoder(encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True)` | function |
| `TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)` | type |
| `Unflatten(dim, unflattened_size)` | type |
| `value_and_grad_aux(fn)` | value_and_grad for a function returning [loss, aux]: returns (loss, aux, grads). aux values (e.g. the logits used to compute the loss) are returned without their gradient graph, so nothing is computed twice. |

### `use web` — 10 names

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
| `websocket_connect(url, headers=None, timeout=10.0, subprotocols=None)` | function |

## Command line (43 commands)

Every command takes `--help` (`apsi help run` works too).

| Command | What it does |
|---|---|
| `apsi run` | run a .apsi program (default: the project's main file) |
| `apsi debug` | debug a program in the terminal (breakpoints, stepping, variables) |
| `apsi dap` | the Debug Adapter Protocol server (VS Code starts it for F5) |
| `apsi native` | native compiler: info, explain FILE, ir FILE [FN], asm FILE [FN], clear-cache |
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
| `apsi plan` | plan training of a large model: GPUs, layout, memory per GPU, days |
| `apsi model` | inspect, verify and convert model files (GGUF, .apsim, safetensors) |
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
