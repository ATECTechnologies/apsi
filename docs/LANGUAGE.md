# The Apsi Language Reference

Version 2.0 · source files use the extension `.apsi` and the UTF-8 encoding.

This document is the complete reference for Apsi: how programs are written, what every construct means, every
built-in function and module, the interactive REPL and the `apsi` command line. Read it from the top to learn the
language, or jump to a chapter.

**Contents**

1. [A first look](#1-a-first-look)
2. [Running programs](#2-running-programs)
3. [Lexical structure](#3-lexical-structure)
4. [Values and types](#4-values-and-types)
5. [Variables and scope](#5-variables-and-scope)
6. [Expressions and operators](#6-expressions-and-operators)
7. [Statements and control flow](#7-statements-and-control-flow)
8. [Functions](#8-functions)
9. [Structs and records](#9-structs-and-records)
10. [Physical units](#10-physical-units)
11. [Measurement uncertainty](#11-measurement-uncertainty)
12. [Tensors and shapes](#12-tensors-and-shapes)
13. [Precision and number formats](#13-precision-and-number-formats)
14. [`evolve` — differential equations](#14-evolve--differential-equations)
15. [Automatic differentiation](#15-automatic-differentiation)
16. [Modules and imports](#16-modules-and-imports)
17. [Tests](#17-tests)
18. [The checker and errors](#18-the-checker-and-errors)
19. [Permissions and security](#19-permissions-and-security)
20. [Execution engines](#20-execution-engines)
21. [Built-in functions](#21-built-in-functions)
22. [Standard modules](#22-standard-modules)
23. [The REPL](#23-the-repl)
24. [The command line](#24-the-command-line)
25. [Projects and `apsi.toml`](#25-projects-and-apsitoml)
26. [Style guide](#26-style-guide)
27. [Grammar](#27-grammar)

---

## 1. A first look

```apsi
# orbit.apsi — a few lines of everything
use physics.constants                         # c, G, M_earth, R_earth, …

struct Body:
    mass: Mass
    radius: Length = 1m
    fn surface_gravity(self) -> Acceleration = G * self.mass / self.radius ** 2

let earth = Body(M_earth, radius=R_earth)
print(f"g = {earth.surface_gravity():.3f}")          # g = 9.820 m/s^2

fn kinetic(m: Mass, v: Velocity) -> Energy = 0.5 * m * v ** 2
print(kinetic(2kg, 3[m/s]))                          # 9.0 J
print(1[kWh] as [MJ])                                # 3.6 MJ

let squares = [i ** 2 for i in 0..10 if i % 2 == 0]
print(squares |> sum)                                # 120

test "kinetic energy":
    assert kinetic(1kg, 2[m/s]) ~= 2J
```

What to notice:

- blocks are made by **indentation**, as in Python;
- numbers can carry **units** (`2kg`, `3[m/s]`), and units are checked *before* the program runs;
- `let` declares a constant, `var` a variable;
- `fn` defines functions, `struct` defines record types with fields and methods;
- `test` blocks live next to the code and run with `apsi test`.

---

## 2. Running programs

```bash
apsi hello.apsi                 # run a file (same as: apsi run hello.apsi)
apsi run hello.apsi a b         # program arguments, read with args()
apsi check hello.apsi           # check only, do not run
apsi test                       # run test blocks (default: tests/ of the project)
apsi                            # interactive REPL
apsi -e "sqrt(2) * 3m"          # evaluate one expression
apsi new myproject              # create a project folder
```

Every run first parses the whole file, then the **checker** reports every error it can prove (unit mismatches,
undefined names, wrong arguments, …) without running a single line, then the program runs. Chapter 24 lists
every command and flag.

On Linux a file starting with `#!/usr/bin/env apsi` can be made executable with `chmod +x`.

---

## 3. Lexical structure

### 3.1 Encoding and lines

- Source files are UTF-8. A byte-order mark (as written by Windows Notepad) and CRLF line endings are accepted.
- A **comment** starts with `#` and runs to the end of the line.
- Statements end at the end of a line. `;` separates several statements on one line: `let a = 1; let b = 2`.
- Inside `()`, `[]` and `{}` lines may be broken freely.
- Outside brackets, a line ending in `\` continues on the next line.
- A line that **starts** with `|>` always continues the previous line, so pipelines can be written vertically:

  ```apsi
  let ds = from_tensors(X, y, batch=256)
      |> map(augment, workers=8)
      |> prefetch(4)
  ```

### 3.2 Indentation

A block follows a line ending in `:` and is indented further than that line. Use 4 spaces (a tab counts as 4).
The block ends where the indentation returns to an outer level. A block of a single statement may stay on the
same line: `if x < 0: return 0`.

### 3.3 Identifiers

Names start with a letter or `_` and continue with letters, digits or `_`. Any Unicode letter is allowed:
`let Δt = 1[ms]`, `let ω = 2 * pi * 50[Hz]`, `let größe = 3m`. Names starting with `_` are private to their file (not exported by
`import`). Names are case-sensitive.

### 3.4 Keywords

Reserved everywhere:

```
let var fn return if elif else while for in break continue struct
true false none and or not use import assert evolve as pass
```

Contextual (special only in one place, ordinary names elsewhere): `try`, `except`, `finally`, `raise`, `yield`
(statements, section 7.2–7.3), `test` (starts a test block),
`parallel` (only directly before `for`), `over step using tol record every into` (inside `evolve`).

### 3.5 Numbers

| Form | Examples | Type |
|---|---|---|
| integer | `42`, `1_000_000` | `Int` |
| decimal | `3.14`, `0.5`, `6.02e23`, `1e-9`, `2.5E+3` | `Float` |
| with a unit (letters right after the digits, no space) | `5m`, `2kg`, `300K`, `10ms`, `3eV`, `5µs` | `Quantity` |
| with a bracketed unit | `9.81[m/s^2]`, `8.314[J/(mol K)]`, `1[N*m]`, `5[m²]` | `Quantity` |

`_` may separate digit groups. A number written directly before a name is always read as a unit: write
`2 * x`, not `2x`.

### 3.6 Strings

| Form | Example |
|---|---|
| plain | `"text"` |
| multi-line | `"""line 1` ⏎ `line 2"""` |
| formatted (f-string) | `f"x = {x:.3f}, list has {len(xs)} items"` |
| formatted, multi-line | `f"""…{expr}…"""` |

Escapes: `\n` (newline), `\t` (tab), `\r`, `\\`, `\"`, `\0`. Any other backslash is kept as it is, so Windows paths
such as `"C:\data\x.csv"` mostly work, but `"C:\\data\\x.csv"` or `"C:/data/x.csv"` is clearer.

In f-strings, `{expression}` inserts a value and `{expression:spec}` formats it with a Python-style format spec
(`.3f`, `>8`, `,`, `e`, `%`, …). For quantities the spec formats the number and the unit is appended:
`f"{9.81[m/s^2]:.1f}"` → `9.8 m/s^2`. `{{` and `}}` produce literal braces.

### 3.7 Operator characters

`+ - * / // % ** ^ @ == != < <= > >= ~= = += -= *= /= //= %= **= @= -> => |> .. ..= . , : ' ( ) [ ] { }`

Unicode alternatives: `×` and `·` for `*`, `−` for `-`, `≤ ≥ ≠ ≈` for `<= >= != ~=`, `→` for `->`, `±` (or `+/-`)
for uncertainty.

---

## 4. Values and types

| Type | Examples | Notes |
|---|---|---|
| `Int` | `42`, `-7`, `1_000` | arbitrary precision |
| `Float` | `3.14`, `6.02e23`, `inf`, `nan` | IEEE 754 double |
| `Quantity` (shown as its dimension, e.g. `Length`) | `2kg`, `9.81[m/s^2]`, `1.5[kWh]` | number + SI dimension, see chapter 10 |
| `Exact` | `exact("1/3")`, values inside `@precision(exact)` | exact fractions |
| `Uncertain` | `9.81[m/s^2] ± 0.02[m/s^2]`, `3.0 +/- 0.1` | see chapter 11 |
| `Bool` | `true`, `false` | |
| `Str` | `"text"`, `f"x = {x}"` | immutable Unicode text |
| `None` | `none` | "no value" |
| `List` | `[1, 2, 3]`, `[]` | mutable, any element types |
| `Tuple` | `(1, 2)`, `(x,)`, `a, b` | immutable |
| `Dict` | `{"k": 1}`, `{}` | keys are usually strings or numbers |
| `Range` | `0..10` (10 excluded), `0..=10` (10 included) | lazy integer ranges, also with units: `0s..10s` |
| `Tensor` | `tensor([[1, 2], [3, 4]])`, `zeros(3, 3)`, `randn(64, 128)` | n-dimensional arrays, may carry a unit |
| `Record` | `record(W=w, b=b)` | ad-hoc container with named fields |
| struct instance | `Particle(...)` | see chapter 9 |
| `Fn` | `fn f(x) = …`, `fn(x) => x * 2`, built-ins | functions are values |

`type(x)` returns the type name as a string (`"Int"`, `"Length"`, `"Tensor"`, …).

**Truthiness.** In `if`, `while`, `and`, `or` and `not`: `false`, `none`, `0`, `0.0`, empty strings, lists, tuples
and dicts are false; everything else is true.

**Equality.** `==` compares values (lists element by element, quantities with their units). `a ~= b` compares
approximately (relative tolerance 1e-6) and also checks units.

### 4.1 Strings

Strings support `+` (concatenation), `*` with an `Int` (repetition), indexing `s[0]`, slicing `s[1:4]`, `len(s)`,
`in` (substring test) and the usual methods:

```apsi
let s = "Hello, World"
s.upper()  s.lower()  s.strip()  s.split(", ")  s.replace("World", "Apsi")
s.startswith("He")  s.endswith("!")  s.find("o")  s.count("l")  s.title()
", ".join(["a", "b"])     # or join(["a", "b"], ", ")
chars(s)                  # list of characters
str(42) + " items"        # str() converts anything, int("42"), float("1.5") parse
```

### 4.2 Lists, tuples and dicts

```apsi
var xs = [3, 1, 2]
xs.append(4)              # also extend, insert, pop, remove, index, count, sort, reverse, clear, copy
xs[0]  xs[-1]  xs[1:3]  xs[::2]
2 in xs                   # membership
[1, 2] + [3]              # concatenation
len(xs)  sorted(xs)  reversed(xs)  sum(xs)  min(xs)  max(xs)

let t = (1, "a")
let a, b = t              # unpacking

var d = {"a": 1}
d["b"] = 2
d.get("c", 0)  d.keys()  d.values()  d.items()  "a" in d  keys(d)
```

### 4.3 Comprehensions

```apsi
[x * x for x in 0..10]
[x for x in data if x > 0]
[(i, j) for i in 0..3 for j in 0..3 if i != j]
{k: v * 2 for k, v in d.items()}
```

---

## 5. Variables and scope

```apsi
let x = 10                  # constant: cannot be assigned again
var y = 0                   # variable: can change
let a, b = 1, 2             # several names at once (unpacks any sequence; one level, no nesting)
let v: Velocity = 3[m/s]    # optional type annotation, checked
y += 1                      # += -= *= /= //= %= **= @=
xs[0] = 5                   # element assignment (lists, dicts, tensors)
p.mass = 2kg                # field assignment (structs, records)
```

- A name must be declared with `let` or `var` before it is used — a typo is an error, not a new variable.
- A `var` that holds a physical quantity **keeps its dimension**: assigning a `Time` to a `Length` variable is an error.
- Blocks (`if`, `for`, `while`, function bodies) open a new scope. Names declared inside are not visible outside.
- Loop variables and function parameters are constants.
- Functions can read and modify (`var`) variables of the scopes around them (closures).
- Declaring the same name again in an inner scope shadows the outer one.

---

## 6. Expressions and operators

### 6.1 Precedence (lowest first)

| Level | Operators | Notes |
|---|---|---|
| lambda | `fn(x) => expr` | |
| pipe | `x \|> f` | `x \|> f` is `f(x)`; `x \|> f(a)` is `f(x, a)` |
| conditional | `a if cond else b` | |
| logical | `or`, `and`, `not` | short-circuit |
| comparison | `== != < <= > >= ~= in not in` | chain: `0 < x < 1` |
| range / conversion | `a..b`, `a..=b`, `x as [unit]` | |
| additive | `+ -` | |
| multiplicative | `* / // % @` | `@` is matrix multiplication |
| unary | `-x`, `+x` | |
| uncertainty | `x ± σ`, `x +/- σ` | binds tighter than `*` |
| power | `**` or `^` | right-associative: `2**3**2 == 512`, `-2**2 == -4` |
| postfix | `f(a, k=v)`, `x[i, j]`, `x[1:5:2]`, `x.field` | |

### 6.2 Arithmetic

`/` always divides exactly (`7 / 2 == 3.5`); `//` is floor division (`-7 // 2 == -4`); `%` is the remainder with
the sign of the divisor. Arithmetic on quantities follows the unit rules in chapter 10. Arithmetic on tensors is
element-wise with NumPy broadcasting.

### 6.3 Pipes

```apsi
data |> normalize |> train(model)          # train(normalize(data), model)
[1, 2, 3] |> sum                           # 6
```

### 6.4 Indexing and slicing

`x[i]`, `x[-1]` (from the end), `x[a:b]`, `x[a:b:step]`, `x[:, 0]` (tensor column), `x[i, j]` (tensor element),
`x[mask]` (tensor boolean mask), `x[idx_tensor]` (gather).

### 6.5 Conditional expression

```apsi
let label = "even" if n % 2 == 0 else "odd"
```

### 6.6 Approximate comparison

`a ~= b` is true when `|a - b| <= 1e-6 · max(|a|, |b|)` (and units match). For other tolerances use
`approx(a, b, rel=1e-9, atol=1e-12)`.

---

## 7. Statements and control flow

```apsi
if x > 0:
    print("positive")
elif x == 0:
    print("zero")
else:
    print("negative")

while n > 1:
    n = n // 2

for i in 0..10:                    # 0, 1, …, 9
    if i == 3:
        continue
    if i == 7:
        break
    print(i)

for i, item in enumerate(items):   # with an index
    pass                           # does nothing

for k, v in d.items():
    print(k, v)

assert total > 0, "total must be positive"
```

- `for` iterates over ranges, lists, tuples, strings, dict keys, tensors (first axis), data pipelines and channels.
- `break` leaves the innermost loop; `continue` starts its next iteration.
- `pass` is an empty statement for blocks that intentionally do nothing.
- `assert cond` stops with an `AssertionError` when `cond` is false. Without a message, the values of both sides
  of a comparison are shown: `assert x == 3` → `AssertionError: assertion failed: 4 == 3`.
- `return` leaves a function (chapter 8). `return` outside a function is an error.
- `error("message")` raises a `RuntimeError` with your message; `exit(code)` stops the program.

### 7.2 Errors: `try` / `except` / `else` / `finally` and `raise`

```apsi
fn parse_speed(text):
    try:
        return float(text) * 1[m/s]
    except ValueError as e:               # e.kind, e.message, e.line; str(e) is the message
        print(f"not a number: {e.message}")
        return none

try:
    let cfg = load_json("config.json")
except FileNotFoundError, PermissionError:   # several kinds; (A, B) works too
    print("using defaults")
except as e:                                 # every other error (also: except Error as e)
    print("unexpected:", e.kind)
    raise                                    # hand the same error on
else:
    print("config loaded")                   # only when nothing failed
finally:
    print("always runs")

raise ValueError("speed must be positive")   # the familiar Python kinds are built in
raise "something went wrong"                 # a RuntimeError with this message
```

Kinds use Python's names, so converted code keeps its exact behaviour: `ValueError`, `TypeError`, `KeyError`,
`IndexError`, `ZeroDivisionError`, `ArithmeticError`, `LookupError`, `AttributeError`, `NameError`,
`RuntimeError`, `NotImplementedError`, `AssertionError`, `StopIteration`, `OSError`, `FileNotFoundError`,
`PermissionError`, `TimeoutError`, `UnicodeError`, `RecursionError`, `OverflowError`, plus Apsi's `UnitError` and
`ShapeError`. Errors Apsi raises itself are matched the same way: dividing by zero is a `ZeroDivisionError`, a
missing dictionary key a `KeyError`, an index outside a list an `IndexError`, a unit mismatch at run time a
`UnitError` (which is also a `ValueError`). Parent kinds match their children (`except LookupError` catches
`KeyError` and `IndexError`).

Your own error types are structs: `struct OutOfFuel: message: Str = ""` and `raise OutOfFuel("tank empty")` —
`except OutOfFuel as e` then reads `e.message` and every other field of the struct.

### 7.3 Generators: `yield`

A function that contains `yield` produces its values one at a time, only when they are asked for — so a sequence
may be endless:

```apsi
fn naturals():
    var n = 0
    while true:
        yield n
        n += 1

fn evens(xs):
    for x in xs:
        if x % 2 == 0:
            yield x
    yield from [100, 200]                  # hand on every value of another sequence

for n in naturals():
    if n > 4:
        break
    print(n)
print(list(evens([1, 2, 3, 4])))           # [2, 4, 100, 200]
```

### 7.4 `for … else` / `while … else` in converted code

Apsi has no loop `else`; `apsi convert` rewrites Python's `for … else` with a flag so the `else` part still runs
only when the loop finished without `break`.

### 7.1 `parallel for`

```apsi
@native
fn total_energy(m, v) -> Float:
    var e = 0.0
    parallel for i in 0..len(m):           # iterations run on all CPU cores
        e += 0.5 * m[i] * v[i] * v[i]
    return e
```

`parallel for` promises that the iterations are independent, so they may run at the same time and in any order.
In a `@native` function the iterations are spread over all CPU cores; in a `@gpu` function each iteration is one
GPU thread. Everywhere else (the bytecode engine, the REPL) it runs like an ordinary `for`, so the same program
works on every engine.

- Variables declared inside the loop are private to each iteration.
- An outer number may only be updated as a **reduction**: `x += …`, `x -= …`, `x *= …`, `x = min(x, …)` or
  `x = max(x, …)`. Each thread works on its own copy, and the copies are combined in a fixed order, so the result is
  identical on every machine and for every thread count.
- Writing to tensor elements is allowed; two iterations must not write the same element.
- `return`, `break` and `print` are not allowed inside (the native compiler runs such a loop sequentially and says why).
- The loop must count over a range: `parallel for i in a..b`, `a..=b` or `range(…)`.

`parallel` is a soft keyword: a variable called `parallel` keeps working.

---

## 8. Functions

### 8.1 Definitions

```apsi
fn area(r: Length) -> Area = pi * r ** 2        # expression body

fn solve(a, b = 1.0, tol = 1e-9):               # block body
    var x = a
    while abs(x - b) > tol:
        x = (x + b) / 2
    return x

fn energy(m: Mass, v: Velocity) -> Energy =     # expression body on the next line
    0.5 * m * v ** 2

let double = fn(x) => x * 2                     # anonymous function (lambda)
```

- A block body without `return` returns `none`. `return a, b` returns a tuple.
- Parameters may have **defaults** (`b = 1.0`). Arguments may be passed by position or by name
  (`solve(1, tol=1e-6)`). Unknown names and wrong counts are reported by the checker.
- Functions are values: pass them as arguments, return them, store them in lists and records.
- Functions are **closures**: they capture the variables around them, and nested functions may modify outer `var`s.
- Recursion is allowed (the limit is deep — 20,000 frames).

### 8.2 Type annotations

Annotations on parameters, results and variables are **checked by the checker and again at run time**.

| Annotation | Means |
|---|---|
| `Float`, `Int`, `Bool`, `Str`, `List`, `Tuple`, `Dict`, `Record`, `Fn`, `Any` | basic types (`Float` accepts `Int`) |
| `Length`, `Mass`, `Time`, `Velocity`, `Force`, `Energy`, … | physical quantity of that dimension (list in 10.6) |
| `[m/s]`, `[J/K]` | a dimension written as a unit |
| `Tensor[B, T, D]` | tensor whose shape matches the symbols |
| `Tensor[N, 3, Length]` | tensor with a unit for its elements |
| `Tensor[_, 10, f32]` | `_` = any size; a dtype converts the tensor (see 13) |
| `Vec3[Velocity]` | same as `Tensor[3, Velocity]` (`Vec`, `Vec2`, `Vec4`, `Mat`, `Matrix` exist too) |
| a struct name | an instance of that struct |

### 8.3 Shape generics

```apsi
fn attention[B, T, D](q: Tensor[B, T, D], k: Tensor[B, T, D], v: Tensor[B, T, D]) -> Tensor[B, T, D]:
    let scores = q @ transpose(k, (0, 2, 1)) / sqrt(D)
    return softmax(scores) @ v
```

Shape symbols (`B`, `T`, `D`) are bound to the real sizes at the call. Every argument must agree; otherwise a
`ShapeError` names the dimension that differs. Inside the function the symbols are ordinary numbers (`sqrt(D)`).
Writing `[B, T, D]` after the name is optional — symbols used in `Tensor[...]` are declared automatically.

### 8.3b Any number of arguments: `*args` and `**kwargs`

```apsi
fn total(*xs, **opts):                     # xs is a List, opts a Dict
    var t = 0
    for x in xs:
        t += x
    return t * opts.get("scale", 1)

print(total(1, 2, 3))                      # 6
print(total(1, 2, scale=10))               # 30
let nums = [4, 5]
let o = {"scale": 2}
print(total(*nums, **o))                   # spread a list and a dict into a call: 18
```

`*name` collects the remaining positional arguments, `**name` the remaining keyword arguments; both come after
the ordinary parameters. Functions with them run on the bytecode engine (`@native` refuses them).

### 8.4 Decorators

A decorator is any function that takes a function and returns a function; `@name` above `fn` applies it.

| Decorator | Effect |
|---|---|
| `@precision(exact)` | every number in the function is an exact fraction — no rounding at all |
| `@precision(f32)` / `f16` / `f64` | tensors entering and leaving the function are converted to that dtype |
| `@precision(bf16)` / `fp8_e4m3` / `fp8_e5m2` / `fp6_e3m2` / `fp4_e2m1` / `int8` / `int4` | tensors are rounded to what the format can represent |
| `@memo` | remember results by argument values |
| `@timed` | print how long each call takes |
| `@app.get("/path")` etc. | register a web route (chapter 22, `web`) |
| `@disk_cache("dir")` | remember results on disk across runs (`use data`) |
| `@native` | compile the function to machine code (see [`NATIVE.md`](NATIVE.md)); `@native("fast")` drops the safety checks and allows fast-math, `@native("strict")` makes "cannot compile" an error |
| `@gpu` | run the function's `parallel for` on the NVIDIA GPU(s), or on all CPU cores when there is none ([`NATIVE.md`](NATIVE.md#gpu)) |

```apsi
@memo
fn fib(n: Int) -> Int = n if n < 2 else fib(n - 1) + fib(n - 2)
print(fib(90))                                   # instant
```

---

## 9. Structs and records

### 9.1 Structs

```apsi
struct Particle:
    pos: Vec3[Length]
    vel: Vec3[Velocity]
    mass: Mass = 1kg                             # default value
    fn kinetic(self) -> Energy = 0.5 * self.mass * dot(self.vel, self.vel)
    fn moved(self, dt: Time):
        return Particle(self.pos + self.vel * dt, self.vel, mass=self.mass)

let p = Particle(tensor([0.0, 0, 0]) * u.m, tensor([1.0, 0, 0]) * u.m / u.s)
p.kinetic()                                      # 0.5 J
p.mass = 2kg                                     # fields can be changed
print(p)                                         # Particle(pos=…, vel=…, mass=2.0 kg)
```

- Fields are declared as `name: Type` with an optional `= default`; fields with defaults come last.
- Construct with positional or named arguments: `Particle(pos, vel, mass=3kg)`. Field types are checked.
- Methods are `fn` definitions inside the struct; the first parameter is `self`.
- Struct names can be used as types in annotations.

### 9.1b Operators on your own structs

Methods with these names make a struct work with operators and built-ins — the same idea as Python's special
methods (`apsi convert` renames `__add__` → `add`, `__str__` → `to_str`, and so on):

| Method | Used by | | Method | Used by |
|---|---|---|---|---|
| `add sub mul div` | `+ - * /` | | `eq lt le gt ge` | `== < <= > >=` (and `sorted`) |
| `floordiv mod pow matmul` | `// % ** @` | | `neg` | unary `-` |
| `radd rmul rsub rdiv` | `3 + v` … | | `len get set contains` | `len(v)` `v[i]` `v[i] = x` `x in v` |
| `iter` | `for x in v` | | `to_str` | `print`, `str()`, f-strings |
| `call` | `v(...)` | | `hash bool` | dictionary keys, `if v:` |

```apsi
struct Vec:
    x: Float
    y: Float
    fn add(self, o) = Vec(self.x + o.x, self.y + o.y)
    fn mul(self, k) = Vec(self.x * k, self.y * k)
    fn to_str(self) = f"Vec({self.x}, {self.y})"

print(Vec(1.0, 2.0) + Vec(3.0, 4.0) * 2.0)   # Vec(7.0, 10.0)
```

### 9.2 Records

`record(a=1, b=2)` creates an untyped container with named fields — ideal for model parameters:

```apsi
let layer = record(W=randn(4, 8), b=zeros(8))
layer.W                                          # field access
```

### 9.3 Parameter trees (pytrees)

Structs, records, lists, tuples and dicts nest into trees. `grad`, `value_and_grad`, `tree_map`, optimizers and
`param_count` walk into every leaf, so a whole model is just a tree:

```apsi
let model = [record(W=randn(2, 16), b=zeros(16)), record(W=randn(16, 1), b=zeros(1))]
let doubled = tree_map(fn(x) => 2 * x, model)
print(param_count(model))                        # 65
```

---

## 10. Physical units

### 10.1 Writing quantities

- letters directly after a number: `5m`, `2kg`, `300K`, `10ms`, `3eV`, `5µs`
- brackets for compound units: `9.81[m/s^2]`, `8.314[J/(mol K)]`, `1[N*m]`, `5[m²]`, `3[kg m/s]`
- inside brackets: `*`, `·` or a space multiply, `/` divides, `^` or `**` raise to a power (negative and fractional
  powers allowed: `^-1`, `^(1/2)`), and superscripts `² ³ ⁻¹` work
- `u.<unit>` is one of that unit: `tensor([1, 2]) * u.km / u.h`, `x * u.eV`
- **SI prefixes** work on base and derived units: Y Z E P T G M k h da d c m u/µ n p f a — `km`, `µs`, `MeV`, `kWh`, `GPa`, `mmol`

### 10.2 Available units

`m g s A K mol cd Hz N Pa J W Wh C V F ohm/Ω S Wb T H eV L bar rad sr deg min h day yr au ly pc Å atm mmHg psi cal kcal Da ft inch mi lb mph kn rpm`

### 10.3 Rules

- `+`, `-`, comparisons and `%` need equal dimensions — **except the number 0**, which matches any unit (so
  `var total = 0` can accumulate quantities).
- `*`, `/` and `@` multiply and divide dimensions; `**` needs a dimensionless exponent.
- **A dimensionless result becomes a plain `Float`**: `6[km] / 2[km]` is `3.0`.
- `exp log sin cos tanh sigmoid softmax …` require dimensionless arguments; `sqrt` halves the dimension, `cbrt` thirds it.
- Values are stored in SI units; printing picks a readable unit (N, J, W, Pa, …).
- Unit errors are found by the checker before running whenever they can be proved; the rest are found at run time.

### 10.4 Conversion and access

```apsi
let d = 12.5[km]
d as [m]                 # 12500.0 m   (display unit; must be the same dimension)
(90deg) as [rad]         # 1.5707963…  — converting to a dimensionless unit gives a number
d.magnitude              # 12.5        (number in the displayed unit)
d.value                  # 12500.0     (number in SI units)
d.unit                   # "km"
unit(d)  dim(d)  magnitude(d)
```

### 10.5 Unit-aware functions

Most built-ins accept quantities and give results with the right unit: `abs`, `sqrt`, `min`, `max`, `sum`,
`mean`, `std`, `norm`, `dot`, `cross`, `linspace`, `arange`, `linalg.eigh` (eigenvalues of a Hamiltonian in J),
`integrate` (unit [f]·[x]), `fft` frequencies (Hz when the step has a time unit), `grad` (∂Energy/∂Length is a Force).

### 10.6 Quantity types for annotations

Dimensionless, Angle, Length, Mass, Time, Current, Temperature, Amount, LuminousIntensity, Area, Volume,
Velocity / Speed, Acceleration, Jerk, Force, Energy / Work / Torque, Power, Pressure / Stress, Momentum / Impulse,
AngularMomentum / Action, Frequency / AngularVelocity, Density, Charge, Voltage, Resistance, Capacitance,
Inductance, MagneticField, MagneticFlux, ElectricField, Entropy / HeatCapacity, Viscosity,
Diffusivity / KinematicViscosity, WaveNumber, Concentration, MolarMass, ThermalConductivity, SpringConstant.

### 10.7 Physical constants (`use physics.constants`)

`c G h hbar k_B N_A R_gas q_e m_e m_p m_n epsilon_0 mu_0 g0 sigma_SB a_0 alpha_fs M_sun M_earth R_earth AU`
— CODATA values with units.

---

## 11. Measurement uncertainty

```apsi
let g = 9.81[m/s^2] ± 0.02[m/s^2]
let L = 1.20m ± 0.01m
let T = 2 * pi * sqrt(L / g)
print(T)                          # (2.1975 ± 0.0094) s
print(L - L)                      # (0 ± 0) m  — correlations are tracked exactly
```

- `value ± σ` (or `value +/- σ`) creates a measured value that is an independent error source; σ is the standard
  uncertainty and must have the value's unit.
- `±` binds tighter than `*` and `/`: `2 * 3.0 ± 0.1` is `2 * (3.0 ± 0.1)`.
- Every operation (`+ - * / **`, `sqrt exp log sin cos tan tanh …`) propagates errors to first order **tracking each
  source separately**, so correlations are right: `x - x` is `0 ± 0`, `x * x` has twice the relative error of `x`.
- `nominal(x)`, `sigma(x)` (keeps the unit), `relative_uncertainty(x)`, `measure(v, s)` (same as `v ± s`).
- Comparisons use the nominal value.
- `sci.curve_fit` returns fitted parameters as correlated `±` values.

---

## 12. Tensors and shapes

```apsi
let A = tensor([[1.0, 2], [3, 4]])     # from nested lists (a list of quantities gives a tensor with a unit)
zeros(3, 3)  ones(2, 4)  full((2, 2), 7.0)  eye(3)  zeros_like(A)
arange(0, 10, 2)  linspace(0, 1, 5)  linspace(0m, 1m, 11)
randn(64, 128, seed=1)  rand(3)  randint(0, 10, 5)

A @ A            # matrix product        A * A   # element-wise
A.T              # transpose             shape(A), ndim(A), len(A)
A[0, 1]  A[:, 0]  A[A > 2]
reshape(A, 4)  flatten(A)  stack([A, A])  concat([A, A], axis=1)
sum(A, axis=0)  mean(A)  max(A, axis=1, keepdims=true)  argmax(A)
```

- Element types: `f64` (default), `f32`, `f16`, `i32`, `i64`, `bool`; pass `dtype=f32` to creation functions.
- Arithmetic broadcasts like NumPy.
- A tensor can carry one unit for all its elements: `tensor([1.0, 2.0]) * u.m`.
- Functions and methods of the underlying NumPy array that do not start with `_` are available (`A.sum()`, `A.copy()`).
- Shape annotations (`Tensor[B, T, D]`) are checked on every call (chapter 8.3).
- `use gpu` moves tensors to an NVIDIA GPU (`to_gpu(x)`); every operator and autodiff work there too.

---

## 13. Precision and number formats

| Format | Bits | Used for |
|---|---|---|
| `f64` | 64 | default; physics |
| `f32` | 32 | AI, large simulations |
| `f16`, `bf16` | 16 | inference / training |
| `fp8_e4m3` (`fp8`), `fp8_e5m2` (`bf8`) | 8 | weights & activations / gradients |
| `fp6_e3m2`, `fp6_e2m3` | 6 | OCP MX formats |
| `fp4_e2m1` (`fp4`) | 4 | highly compressed weights |
| `int8`, `int4`, `uint8` | 8 / 4 / 8 | quantization |
| `exact` | – | exact fractions |

- `@precision(fmt)` on a function (chapter 8.4); `Tensor[N, bf16]` in annotations.
- `quantize(x, fmt)` stores real-size codes, `dequantize(q)` decodes, `round_to(x, fmt)` rounds with a straight-through
  gradient (quantization-aware training), `qmatmul(a, b)` multiplies quantized operands accumulating in FP32.
- `exact("1/3")` creates an exact fraction; `to_float(x)` converts back.
- Details and measurements: [`PERFORMANCE.md`](PERFORMANCE.md).

---

## 14. `evolve` — differential equations

```apsi
var x = 1m
var v = 0[m/s]
let k = 4[N/m]
let m = 1kg
evolve x, v over t in 0[s]..10[s] step 1[ms] using rk4 record every 10 into traj:
    x' = v
    v' = -k / m * x
print(x, v)                      # the final state
print(traj.t[-1], traj.x[-1])    # recorded trajectory (tensors with units)
```

Form: `evolve STATE, … over TIME in START..END step DT [using METHOD [tol TOL]] [record [every N] into NAME]:`

- State variables must be declared with `var`; after `evolve` they hold the final state.
- Every state needs a derivative line `x' = expression`. **Its unit must be the unit of `x` divided by the unit of
  time** — checked by the checker and at run time.
- The body may contain `let`, `if`, `for` and function calls; state variables are read-only inside it. The time
  variable (`t`) can be used in the expressions.
- Methods: `euler`, `midpoint`, `rk4` (default), `rk45` (Dormand–Prince with adaptive steps: `step` is the first
  step, `tol 1e-10` the accuracy, default `1e-8`), `symplectic` (semi-implicit Euler, updating from the last state
  to the first, so order position, velocity), `verlet` (velocity Verlet; exactly two states: position and velocity).
- `record [every N] into name` gives `name.t`, `name.x`, `name.v`, … as tensors with units.
- **Speed**: the first step runs fully unit-checked; then a unit-erased version runs on plain SI numbers. Both are
  compared before the switch, and on any difference the checked version is used.

```apsi
var y = 1.0
evolve y over t in 0[s]..5[s] step 0.5[s] using rk45 tol 1e-10 record into tr:
    y' = -y / 1[s]
```

---

## 15. Automatic differentiation

```apsi
fn potential(x: Length) -> Energy = 0.5 * 10[N/m] * x ** 2
let force = grad(potential)             # a new function: dU/dx
print(force(0.1m))                      # 1.0 N  — the unit is derived

fn loss(w, b) = mean((X @ w + b - y) ** 2)
let l, gs = value_and_grad(loss, wrt=[0, 1])(w, b)    # gs[0] = ∂l/∂w, gs[1] = ∂l/∂b
```

- `grad(f, wrt=0)` returns a function computing ∂f/∂(argument `wrt`); `wrt` may be a list.
- `value_and_grad(f, wrt=0)` returns `(value, gradient)`.
- Arguments may be numbers, tensors, quantities or parameter trees (struct/record/list/dict); the gradient has the
  same shape.
- **Gradients have correct units**: if `f` returns `Energy` and `x` is a `Length`, the gradient is a `Force`.
- `if`, `for`, function calls and indexing work inside `f` (define-by-run). `f` must return a scalar.
- **Any order**: `grad(grad(f))`, `hessian(f)`, `jacobian(f)`, `laplace(f)` (Σ ∂²f/∂xᵢ²); `grad` may be used inside a
  function that is itself being differentiated — enough for physics-informed neural networks (`examples/18_pinn.apsi`).
- `stop_gradient(x)` blocks gradients; `derivative(f, x)` is the derivative of a scalar function at `x`;
  `nn.numeric_grad(f, x)` checks gradients with finite differences.
- `conv2d` and `max_pool2d` are differentiable to first order only.

---

## 16. Modules and imports

```apsi
use nn                       # bring every name of a module into scope, and also nn.name
use linalg as la             # or keep them under a short name only: la.eigh(H)
use physics.constants        # dotted module names
use geom_tools               # an installed library (apsi install geom-tools)
import "mylib.apsi"          # the public names (not starting with _) of another file
import "lib/geometry.apsi" as geo
```

- `use NAME` brings every public name of the module into scope and, unless that name is already taken, also makes
  the module itself available as `NAME`, so both `ppo(…)` and `rl.ppo(…)` work after `use rl`.
- `use` finds the standard modules listed in chapter 22 and installed libraries (`.apsil`, see
  [Libraries](LIBRARIES.md)). Dashes in a library name become underscores: `geom-tools` → `use geom_tools`.
- `import` runs another `.apsi` file once (cached) and brings in its public top-level names. The path is relative to
  the importing file. Its `test` blocks do not run on import.
- Circular imports are reported as errors.

Standard modules: `math`, `physics`, `physics.units`, `physics.constants`, `physics.fields`, `physics.formulas`, `nn`,
`layers`, `torch`, `rl`, `linalg`, `random`, `sci`, `special`, `stats`, `signal`, `mechanics`, `pde`, `quantum`,
`medical`, `lab`, `precision`, `data`, `parallel`, `scale`, `web`, `models`, `gpu`, `sys`, `crypto`, `io`, `ffi`,
`python`, `net`.

---

## 17. Tests

```apsi
fn mean_of(xs) = sum(xs) / len(xs)

test "mean of three numbers":
    assert mean_of([1, 2, 3]) == 2

test "works with units":
    assert mean_of([1m, 3m]) ~= 2m
```

- `apsi run` skips `test` blocks; `apsi test` runs the file's top-level code first, then each test in that file's scope.
- Put tests next to the code, or in `tests/*.apsi` files that `import` the code.
- `testing()` is `true` while tests run — use it to skip side effects such as starting a server.
- `apsi test` prints ✓/✗ per test with its time and exits with status 1 when something fails.

---

## 18. The checker and errors

Before running, Apsi checks the whole file and reports **every** problem it can prove, without running any code:

- undeclared names, including typos
- assignments to `let` constants, parameters or functions
- unit mismatches in `+ - < ==`, in arguments, in return values, in `var`s whose dimension changes, and in
  `exp/log/sin` of quantities with units
- wrong argument counts and unknown keyword names
- `evolve` rules (`var` states, a derivative for every state, derivative units, step unit)
- `return` outside a function, unknown types

The checker never warns falsely: what it cannot prove is left to the run-time checks.

Every error shows the file, line, column, the source line and a caret, followed by the call stack:

```
orbit.apsi:12:20: UnitError: argument 'v' of kinetic() must be Velocity [m/s], got Length [m] (2.0 m)
    let k = kinetic(3kg, 2m)
                   ^
  in simulate()
```

Error kinds: `SyntaxError`, `NameError`, `TypeError`, `UnitError`, `ShapeError`, `RuntimeError`, `AssertionError`,
`PermissionError`.

---

## 19. Permissions and security

Programs run with a permission profile. By default (`standard`) files can be read and written; Python, C, the
network and environment variables are off.

| Permission | Allows | Flag |
|---|---|---|
| `read` / `write` | files (optionally only some folders) | `--allow-read[=DIRS]`, `--allow-write[=DIRS]` |
| `net` | `http_get`, `http_post_json`, `use web` servers | `--allow-net[=HOSTS]` |
| `env` | `getenv` | `--allow-env[=NAMES]` |
| `python` | `pyimport`, `to_torch`, `from_torch`, ONNX/HF/PyTorch models | `--allow-python` (`=all` for system modules) |
| `ffi` | `cload` (C/C++/Rust libraries) | `--allow-ffi[=LIB@SHA256]` |

Profiles: `sandbox` (nothing, not even files), `strict`, `standard` (default), `developer` (everything). Set the
default with `apsi config --profile NAME`, per project in `apsi.toml`, or per run with `--profile`, `--sandbox`,
`--allow-all`. `apsi audit FILE` lists what a program needs and the minimal flags.

Code without a permission gets a `PermissionError` immediately — imported libraries cannot get around it. Attributes
starting with `_` on outside objects are blocked. **Apsi Guard** scans every program before it runs (skip with
`--trust`); `apsi scan PATH` scans on demand. Limits: `--max-memory 2GB`, `--timeout 30s`.
Full details: [`SECURITY.md`](SECURITY.md).

---

## 20. Execution engines

| Engine | How it works | Use it for |
|---|---|---|
| **native** (`@native`, `@gpu`, `apsi run --native`) | each function is specialised on its argument types and units, compiled through LLVM to machine code for the exact CPU (AVX2/AVX-512), cached on disk; `parallel for` uses all cores, `@gpu` the GPUs | numeric code: loops, tensors, physics, simulation — typically 100–3,500× faster than bytecode, on par with C/C++/Rust ([`NATIVE.md`](NATIVE.md)) |
| `compile` (default for everything else) | the checked program is compiled to Python bytecode: scopes resolved at compile time, specialized type checks, and an **erased twin** of every function used in hot loops | everything; code the native compiler does not handle (strings, lists, records, I/O) |
| `interp` | the tree-walking reference interpreter that defines the language's meaning | cross-checking the engines; the REPL |

The engines work together in one program: native functions and bytecode functions call each other freely, and
`evolve` calls native right-hand sides from its integration loop. A `@native` function that uses something the
native compiler does not support says why (once) and runs on the bytecode engine, so adding `@native` never breaks
a program. `apsi native explain app.apsi` lists which functions run as machine code.

Select the reference interpreter with `--engine interp`. The engines pass the same test suite; native results are
bit-identical to the bytecode engine for scalar arithmetic. Closures created in loops capture each iteration's
value. Run-time errors always point back to the line and column in the `.apsi` file.

Parsed programs are kept in a signed compile cache, and native machine code in a signed machine-code cache, so an
unchanged program starts without recompiling; `apsi build` turns a program into an executable (`.exe`). Both are
described under *Executables and the compile cache* in chapter 24. Neural networks built with `use layers` can also
run on PyTorch (CPU or GPU) — see [`BACKENDS.md`](BACKENDS.md).

---

## 21. Built-in functions

Available everywhere without `use`. Unless noted, functions accept numbers, tensors, quantities and values that
are being differentiated alike. `apsi doc NAME` (or `:doc NAME` in the REPL) prints a name's signature.

### 21.1 Input, output and conversion

| Function | Description |
|---|---|
| `print(*values, sep=" ", end="\n")` | write values to standard output |
| `input(prompt="")` | read one line from standard input (without the newline); `none` at end of input |
| `args()` | program arguments: `apsi run app.apsi a b` → `["a", "b"]` |
| `exit(code=0)` | stop the program with an exit status |
| `str(x)` · `int(x)` · `float(x)` · `bool(x)` | convert (`int("42")`, `float("1.5")`) |
| `list(x=())` · `tuple(x)` · `dict(...)` · `set(x)` | containers from iterables / keyword arguments |
| `any(xs)` · `all(xs)` · `iter(xs)` · `next(it[, default])` | truth tests and iterators |
| `ord(c)` · `chr(n)` · `hex(n)` · `bin(n)` · `oct(n)` · `format(x, spec)` · `repr(x)` | characters, number bases, formatting |
| `getattr(obj, name[, default])` · `hasattr(obj, name)` · `callable(x)` | reflection (names starting with `_` stay private) |
| `type(v)` | type name as a string |
| `error(msg)` | raise a `RuntimeError` with `msg` |
| `testing()` | `true` while `apsi test` runs |
| `platform()` | `"windows"`, `"linux"` or `"macos"` |
| `clock()` | a monotonic time in seconds (subtract two calls to measure) |
| `permissions()` · `security_policy()` | the permissions of the running program |

### 21.2 Sequences and collections

| Function | Description |
|---|---|
| `len(x)` | number of elements (first axis for tensors) |
| `range(a, b=none, step=1)` | integers as a list-like range (`0..n` is usually nicer) |
| `enumerate(xs, start=0)` | `(index, item)` pairs |
| `zip(*xs)` | tuples of corresponding elements |
| `map(f, xs)` · `filter(f, xs)` | apply / keep where `f` is true (also pipeline steps in `use data`) |
| `sorted(xs, key=none, reverse=false)` · `reversed(xs)` | ordering |
| `unique(xs)` | distinct values in first-seen order |
| `keys(d)` | keys of a dict, record or struct |
| `chars(s)` · `join(xs, sep="")` | split into characters / join strings |
| `record(**fields)` | a record |
| `tree_map(f, *trees)` · `param_count(tree)` | map over / count the leaves of a parameter tree |
| `pmap(f, xs, workers=none)` | parallel map on a thread pool |

### 21.3 Math

| Function | Description |
|---|---|
| `abs sqrt cbrt exp log log10 log2` | elementary functions (`sqrt` halves units) |
| `sin cos tan asin acos atan atan2(y, x) sinh cosh tanh` | trigonometry (radians; `deg` converts) |
| `floor ceil round(x, digits=0)` | rounding |
| `min max sum mean std var prod` (`axis=`, `keepdims=`) | reductions; `min`/`max` also take several arguments |
| `cumsum(x, axis)` · `argmax(x, axis)` · `argmin(x, axis)` | cumulative sum, positions of extremes |
| `fsum(xs)` | correctly rounded sum of a list or tensor (no accumulated rounding error; keeps units) |
| `divmod(a, b)` · `pow(a, b[, m])` | quotient and remainder · power (modular with 3 arguments) |
| `approx(a, b, rel=1e-6, atol=1e-12)` | approximate equality |
| `exact(x)` · `to_float(x)` | exact fraction / back to float |
| `pi tau e inf nan` | constants |

### 21.4 Units and uncertainty

| Function | Description |
|---|---|
| `u.<unit>` | one of a unit: `u.km`, `u.eV`, `u.m / u.s` |
| `unit(x)` · `dim(x)` · `magnitude(x)` | unit name, dimension name, number in the displayed unit |
| `measure(value, sigma)` | same as `value ± sigma` |
| `nominal(x)` · `sigma(x)` · `relative_uncertainty(x)` | parts of an uncertain value |

### 21.5 Tensors

| Function | Description |
|---|---|
| `tensor(x, dtype=none)` | tensor from (nested) lists; a list of quantities gives a tensor with a unit |
| `zeros(*shape)` · `ones(*shape)` · `full(shape, value)` · `eye(n)` · `zeros_like(x)` · `ones_like(x)` | constant tensors |
| `arange(start, stop, step=1)` · `linspace(start, stop, n=50)` · `meshgrid(x, y)` | ranges (with units too) |
| `randn(*shape, seed)` · `rand(*shape, seed)` · `randint(lo, hi, *shape, seed)` · `permutation(n, seed)` · `seed(n)` | random numbers (default seed 0, so runs are reproducible) |
| `shape(x)` · `ndim(x)` · `reshape(x, *shape)` · `flatten(x)` · `transpose(x, axes)` | shape handling |
| `swapaxes(x, a, b)` · `permute(x, *axes)` · `unsqueeze(x, axis)` · `squeeze(x, axis)` | axis handling with PyTorch's meaning (negative axes count from the end; all differentiable) |
| `erf(x)` · `gelu(x, approximate="tanh"\|"none")` | error function (differentiable); GELU with the tanh approximation or exact (PyTorch's default) |
| `stack(items, axis=0)` · `concat(items, axis=0)` · `roll(x, shift, axis)` · `where(cond, a, b)` · `clip(x, lo, hi)` · `sort(x)` | data manipulation |
| `dot(a, b)` · `matmul(a, b)` · `outer(a, b)` · `cross(a, b)` · `norm(x, axis)` · `einsum(spec, *xs)` | linear algebra (einsum multiplies units and is differentiable) |
| `f16 f32 f64 i32 i64` (+ `bf16 fp8 … int4`) | dtypes for `dtype=` and `@precision` |

### 21.6 AI and autodiff

| Function | Description |
|---|---|
| `grad(f, wrt=0)` · `value_and_grad(f, wrt=0)` | reverse-mode gradients (chapter 15) |
| `derivative(f, x)` · `jacobian(f)` · `hessian(f)` · `laplace(f)` | higher-order derivatives |
| `stop_gradient(x)` | block gradient flow |
| `relu sigmoid tanh softmax(x, axis) log_softmax(x, axis) gelu softplus` | activations |

### 21.7 Precision

`quantize(x, fmt, scale="auto", axis)` · `dequantize(q)` · `round_to(x, fmt)` · `qmatmul(a, b)` · `nbytes(x)` ·
`quant_error(x, fmt)` · `format_info(fmt)` · decorators `precision(fmt)`, `memo`, `timed`.

### 21.8 Files and data (permission `fs`)

`read_text(path)` · `write_text(path, text)` · `exists(path)` · `load_json(path)` · `save_json(path, value)` ·
`load_csv(path, skip_header=0)` · `save_csv(path, data, header="")` · `json_parse(text)` · `json_str(value)`.
Relative paths are resolved against the running `.apsi` file.

### 21.9 Other languages and the network

| Function | Permission | Description |
|---|---|---|
| `pyimport(name)` | python | import a Python module |
| `to_py(x)` · `from_py(x)` | python | convert values (units become `{value, unit}`) |
| `to_torch(x, device)` · `from_torch(t)` | python | exchange tensors with PyTorch |
| `cload(path)` · `library_path(path)` | ffi | load a C library; `lib.fn(name, [arg types], result type)` |
| `http_get(url, timeout=30)` · `http_post_json(url, data, timeout=30)` | net | HTTP requests |
| `getenv(name, default=none)` | env | environment variables (secrets, API keys) |

---

## 22. Standard modules

Use with `use NAME` (all names) or `use NAME as short` (`short.name`).

### `math`
`abs sqrt cbrt exp log log10 log2 sin cos tan asin acos atan atan2 sinh cosh tanh floor ceil round pi tau e inf nan`
and from `special`: `gamma lgamma erf erfc erfinv factorial comb beta besselj legendre`

### Science, engineering and medicine
`special` (special functions), `stats` (distributions, tests, regression, power), `signal` (filters, spectra,
peaks), `mechanics` (integrators, Lagrangian/Hamiltonian dynamics, N-body, orbits, rigid bodies, beams, trusses),
`pde` (heat, wave, advection, Poisson, Navier–Stokes, CG, BiCGSTAB), `quantum` (Schrödinger equation, operators,
qubits), `medical` (pharmacokinetics, epidemiology, diagnostic accuracy, survival analysis, clinical formulas,
ECG/HRV, DICOM) and `lab` (seeds, provenance, experiment logs, tables). Complete reference:
[`SCIENCE.md`](SCIENCE.md). `physics` also contains radiation, relativity, optics, electromagnetism,
thermodynamics and decay formulas with units (listed there).

### `physics`, `physics.constants`, `physics.fields`, `physics.units`
- constants (CODATA, with units): `c G h hbar k_B N_A R_gas q_e m_e m_p m_n epsilon_0 mu_0 g0 sigma_SB a_0 alpha_fs M_sun M_earth R_earth AU`
- fields: `laplacian(u, dx, boundary="periodic"|"fixed")` (∇² in any number of dimensions) · `gradient(u, dx, axis=0)` · `meshgrid(x, y)`
- `physics` contains both; `physics.units` exists for clarity (units are always available)

### `torch` — the PyTorch API, natively
Layers (`Linear`, `Conv1d/2d`, pooling, `BatchNorm1d/2d`, `LayerNorm`, `RMSNorm`, `GroupNorm`, `Embedding`, `LSTM`,
`GRU`, `RNN`, `MultiheadAttention`, `Transformer…`, `Sequential`, activations), the `F.` functions, losses with all
their options, optimizers (`SGD`, `Adam`, `AdamW`, `Adamax`, `RMSprop`, `Adagrad`, `NAdam`, `RAdam`), every common
learning-rate scheduler, `DataLoader` / `TensorDataset`, `state_dict` / `load_state_dict`, `value_and_grad_aux`,
`clip_grad_norm`, `checkpoint` / `checkpoint_layer` and `Module(init, forward)`. Layers use PyTorch's parameter names
and give the same results as PyTorch. Each layer has `init(seed)` → parameters and is called as
`layer(params, x)`. Reference: [From Python and PyTorch](PYTHON.md).

### `rl` — reinforcement learning
Environments (`cartpole`, `gridworld`, `bandit`), `gae`, `discounted_returns`, `ppo`, `dqn`, `reinforce`, `evaluate`,
preference learning (`dpo_loss`, `ipo_loss`, `preference_loss`, `grpo_advantages`, `kl_penalized_rewards`,
`ppo_clip_loss`, `sequence_log_prob`) and search (`mcts`, `beam_search`, `best_of_n`). Reference:
[Reinforcement learning](RL.md).

### `nn` — neural networks

| Function | Description |
|---|---|
| `dense(n_in, n_out, seed, scale)` | parameters `record(W, b)` (He initialization) |
| `mlp_init(sizes, seed=0)` | a list of dense layers, e.g. `mlp_init([2, 64, 3])` |
| `conv_init(c_in, c_out, k=3, seed)` · `conv2d(x, w, b, stride=1, padding=0)` · `max_pool2d(x, k=2)` | CNNs (x: N×C×H×W) |
| `layer_norm(x, gamma, beta, eps)` · `norm_init(d)` · `batch_norm(x, gamma, beta, eps, axis)` · `rms_norm(x, gamma, eps)` | normalization |
| `embedding_init(vocab, d, seed)` | embedding table (look up with `E[idx]`) |
| `attention(q, k, v, mask)` · `causal_mask(T)` | scaled dot-product attention |
| `dropout(x, p=0.1, train=true, seed)` | dropout |
| `lstm_init(n_in, n_hidden)` · `lstm_cell(p, x, h, c)` · `lstm(p, xs)` | LSTM (xs: B×T×F → B×T×H) |
| `gru_init(n_in, n_hidden)` · `gru_cell(p, x, h)` · `gru(p, xs)` | GRU |
| `mse(pred, target)` · `cross_entropy(logits, labels)` · `accuracy(logits, labels)` · `one_hot(labels, k)` | losses and metrics |
| `l1_loss` · `bce_with_logits(logits, y)` · `bce(prob, y)` · `nll_loss(log_probs, labels)` · `huber_loss(pred, target, delta)` | more losses (mean reduction, the same values as PyTorch) |
| `loader(X, y, batch_size, shuffle, seed)` | mini-batches that are reshuffled every time the loader is iterated (one pass = one epoch) |
| `sgd(lr, momentum)` · `adam(lr, b1, b2, eps)` · `adamw(…, weight_decay)` · `rmsprop(lr, decay, eps)` · `adagrad(lr, eps)` | optimizers: `params = opt.step(params, grads)` |
| `set_lr(opt, lr)` · `cosine_lr(step, total, base, warmup, minimum)` · `step_lr(step, base, every, factor)` · `linear_warmup(step, base, warmup)` | learning-rate schedules |
| `clip_grad_norm(grads, max_norm=1)` · `global_norm(grads)` | gradient clipping |
| `train(loss_fn, params, data, opt, epochs, batch_size, seed, verbose, clip, schedule, eval_fn)` | ready-made training loop; returns `record(params, history)` |
| `save_params(path, params)` · `load_params(path, like)` | save / load parameters (.npz) |
| `numeric_grad(f, x, eps)` | finite-difference gradient check |
| `spiral_data(n_per_class, classes, noise, seed)` · `make_moons` · `make_blobs` · `make_regression` | sample datasets |
| `train_test_split(X, y, test=0.2, seed)` · `standardize(X, axis)` · `batches(X, y, size, seed)` | data preparation |
| `relu gelu sigmoid tanh softmax log_softmax` | activations (also built-ins) |

### `layers` — any neural-network architecture

Layers, composition (`sequential`, `residual`, `parallel`, `dag`), graph neural networks (`graph`, `gcn`, `sage`,
`gat`, `gin`, `graph_pool`, `segment_sum/mean/max/softmax`), transformers, a model zoo (`mlp`, `resnet_mlp`, `cnn`,
`lstm_classifier`, `transformer_lm`, `transformer_classifier`, `autoencoder`, `vae`, `gcn_net`, `gat_net`, `gin_net`,
`siamese`) and training helpers (`fit`, `predict`, `summary`, `count_params`). `custom_layer(init, forward, name,
parts)` turns two functions into a layer (models converted from PyTorch use it), and
`load_torch_weights(model, params, state_dict)` copies PyTorch weights into an Apsi parameter tree. Every layer has `init(seed)` and is
applied as `layer(params, x, train=false, graph=…)`. More layers: `mean_pool(axis)`, `bilinear`, `scale`, `select`.

Backends: `fit(…, backend="numpy"|"torch"|"auto", device="auto"|"cpu"|"cuda"|"mps"|"xpu",
precision="fp64"|"fp32"|"tf32"|"bf16"|"fp16", compile=false)`, `predict(…, backend=…)`, `devices()`,
`torch_module(model, params)` → `torch.nn.Module`, `params_from_torch(model, module)`, `check_torch(device)`.
The torch backend needs `--allow-python`.

Complete reference with shapes, parameters and recipes: [`LAYERS.md`](LAYERS.md); backends: [`BACKENDS.md`](BACKENDS.md).

### `linalg`
`inv det solve eigh eig svd trace norm` — units are preserved (eigenvalues of a Hamiltonian are energies).

### `random`
`seed randn rand randint permutation`

### `sci` — scientific computing

| Function | Description |
|---|---|
| `einsum("ij,jk->ik", A, B)` | Einstein summation (differentiable, multiplies units) |
| `fft ifft rfft irfft(x, n, axis)` · `fftfreq(n, d)` · `rfftfreq(n, d)` · `power_spectrum(x, d)` · `real imag conj` | Fourier transforms (frequencies in Hz when `d` is a time) |
| `integrate(f, a, b, points=64)` | Gauss–Legendre quadrature (unit [f]·[x]) |
| `trapz(y, x, dx)` · `cumtrapz(y, x)` | integrate sampled data |
| `interp(x, xp, fp)` | linear interpolation |
| `find_root(f, x0, tol, max_iter)` · `bisect(f, a, b, tol, max_iter)` | root finding (Newton with exact autodiff derivatives) |
| `minimize(f, x0, method="lbfgs"\|"adam", steps, tol, lr)` | minimize over any parameter tree; returns `record(x, value, steps, converged)` |
| `curve_fit(model, x, y, p0, sigma)` | Levenberg–Marquardt; returns `record(params, cov, chi2, dof)` with correlated `±` parameters |
| `solve_ivp(f, [t0, t1], y0, method="bdf"\|"rk45", rtol, atol, t_eval)` | initial value problems `y' = f(t, y)`: `bdf` is TR-BDF2 (implicit, L-stable, adaptive) for **stiff** systems such as chemical kinetics and circuits; `rk45` is adaptive Dormand–Prince; units are kept; returns `record(t, y, steps, rejected, evaluations, success)` |

### `precision`
`quantize dequantize round_to qmatmul nbytes quant_error format_info` and the formats
`f64 f32 f16 bf16 fp8_e4m3 fp8_e5m2 fp6_e3m2 fp6_e2m3 fp4_e2m1 int8 int4 uint8` (short names `fp8 bf8 fp6 fp4 e4m3 e5m2 fp16 fp32 fp64`).

### `data` — pipelines and storage

| Kind | Functions |
|---|---|
| sources | `from_tensors(*arrays, batch, shuffle, seed, drop_last)` · `from_list(xs)` · `from_range(n)` · `from_generator(f)` · `from_csv(path, skip_header, chunk, delimiter)` · `from_jsonl(path)` · `from_parquet(path, columns, rows)` (needs `apsi pkg install data`) · `from_npy(path)` · `from_files(pattern)` · `interleave([pipelines])` |
| steps (methods or `\|>`) | `map(f, workers, ordered)` · `filter(f)` · `batch(size, drop_last)` · `unbatch()` · `shuffle(buffer, seed)` · `prefetch(n)` · `take(n)` · `skip(n)` · `repeat(times)` · `cache(directory, shard)` · `window(size, shift)` (sliding windows for time series) · `shard(count, index)` (split across workers or machines) |
| results | `.to_list()` · `len(ds)` · `throughput(ds, limit)` · iterate with `for` |
| storage | `open_tensor(path, shape, dtype, mode)` (memory-mapped) · `load_tensor(path, mmap)` · `save_tensor(path, x)` · `flush(x)` · `@disk_cache(directory)` |

### `parallel` — concurrency

`spawn(f, *args)` → task (`.wait()`, `.done()`) · `wait_all(tasks, timeout)` · `wait_any(tasks, timeout)` ·
`channel(capacity)` (`send`, `recv`, `close`, iterate with `for`) · `lock()` (`acquire`, `release`, `run(f)`) ·
`atomic(value)` (`add`, `get`, `set`) · `parallel_for(items, f, workers)` · `parallel_reduce(items, f, combine, init)` ·
`set_threads(n)` · `cpu_count()` · `sleep(t)` · `bench(f, repeat, warmup)`

### `web` — HTTP servers

`server(name, max_body, workers, max_connections, log, handler_timeout, header_timeout, body_timeout, keepalive_timeout, security_headers)` ·
`response(body, status, headers, type)` · `redirect(location, status)` · `abort(status, message)` ·
`jwt_sign(claims, secret, expires_in)` · `jwt_verify(token, secret, leeway)` · `bearer_token(req)` ·
`require_jwt(secret, exempt)` · `require_api_key(keys, header, exempt)`.

Server methods: `get post put patch delete route(method, path)` (as calls or decorators), `before`, `after`,
`rate_limit`, `cors`, `static`, `health`, `expose_metrics`, `stats`, `handle` (call without the network, for tests),
`run(host, port, tls_cert, tls_key, processes, workers)`, `run_background(port)`.
Requests: `req.method path params query headers header(name) body text json() client id user`.
Full guide: [`SERVER.md`](SERVER.md).

### `models` — model files

`save_model(path, params, metadata, shard, sign)` · `load_model(path, verify, key)` · `model_info(path)` (Apsi
checkpoints) · `save_gguf(path, params, metadata, dtype, name, architecture)` · `load_gguf(path, nested)` ·
`gguf_info(path)` · `load_safetensors(path, dtype, names, nested)` · `save_safetensors(path, params, metadata)` ·
`safetensors_info(path)` ·
`quantize_model(params, dtype, min_size)` · `dequantize_model(params)` · `model_size(params)` · `qlinear(x, W, b)` ·
`onnx_model(path, providers)` · `hf_pipeline(task, model, device)` · `torch_load(path)` (the last three need
`--allow-python`). Guides: [`AI.md`](AI.md), [`MODEL-FILES.md`](MODEL-FILES.md).

### `gpu` and `sys`
`device()` · `to_device(x, dev)` · `to_gpu(x)` · `to_cpu(x)` · `gpu_available()` · `gpu_info()` · `hardware()` (CPU,
SIMD, GPU, RAM) · `cpu_threads()`

### `crypto`

| Function | Description |
|---|---|
| `hash(data, algo="sha256")` | sha256, sha512, sha3_256, sha3_512, blake2b, blake2s |
| `hmac_sign(key, message)` · `hmac_verify(key, message, signature)` | message signatures |
| `password_hash(password)` · `password_verify(password, stored)` | salted scrypt |
| `secure_equal(a, b)` | constant-time comparison |
| `random_token(n)` · `random_hex(n)` · `secure_randint(lo, hi)` | randomness from the operating system's CSPRNG |
| `b64encode(text)` · `b64decode(text)` | base64 |

### `io`, `ffi`, `python`, `net`
Group the built-ins of 21.8 and 21.9 so they can be used under a name: `use io as fs` → `fs.read_text(path)`.
`ffi` also has the C types `f64 f32 i32 i64 void cstr ptr_f64 ptr_f32 ptr_i32 ptr_i64`.

---

## 23. The REPL

Start it with `apsi` (or `apsi repl`). Type Apsi code; the value of an expression is printed, and `_` holds the last value.

```text
C:\Users\me apsi> let g = 9.81[m/s^2]
C:\Users\me apsi> 0.5 * g * (2s) ** 2
19.62 m
C:\Users\me apsi> cd C:\Users\me\project
C:\Users\me\project apsi> ls
  Folder: C:\Users\me\project

  Modified               Size  Name
  ----------------  ---------  ----
  2026-11-08 13:40      <dir>  src\
  2026-11-08 13:41      412 B  apsi.toml

  1 folder, 1 file (412 B)
C:\Users\me\project apsi> run src\main.apsi
```

The prompt shows the current folder, as a shell does (`~` stands for your home folder on Linux and macOS; long
paths are shortened in the middle). `APSI_PROMPT=short` shows only `apsi>`.

**Multi-line input** continues automatically after a line ending in `:` (a block — finish it with an empty line),
inside open brackets or a triple-quoted string, and after a trailing `\`, `=`, `,`, `|>` or operator. `:paste`
accepts any block until a line containing `:end`.

**Shell words** behave like a terminal when they are not names you defined: `cd PATH` (Windows paths, quotes,
`~`, `%VAR%`/`$VAR` and `cd -` work; the prompt shows the new folder), `pwd`, `ls`/`dir [PATH]` (folders first,
then files, with modification time, size and a summary; `ls -a` includes hidden files), `clear`/`cls`, `run FILE`,
`help`, `exit`/`quit`.
`!COMMAND` runs any operating-system command (`!git status`).

**Commands** (`:help` lists them; most have short aliases):

| Command | Does |
|---|---|
| `:help [NAME]` (`:h`, `:?`) | help, or documentation for NAME |
| `:quit` (`:q`, `exit`, Ctrl+D) | leave |
| `:vars` (`:v`) | variables and functions of the session with their types |
| `:type EXPR` (`:t`) | type of an expression |
| `:doc NAME` | signature and documentation |
| `:time [CODE]` | time one input, or toggle timing of every input |
| `:bench [N] CODE` | benchmark code (N runs, or about one second) |
| `:load FILE` (`:l`) · `:reload` (`:r`) | run a file inside the session, keeping its definitions |
| `:run FILE` | run a file in a fresh interpreter |
| `:check FILE…` · `:test [PATH…]` · `:fmt FILE…` | toolchain commands |
| `:edit [FILE]` | open `$EDITOR` (Notepad on Windows), then run what you wrote |
| `:paste` | paste a large block, finish with `:end` |
| `:save [FILE]` | save every successful input of the session as a `.apsi` file |
| `:history [N]` (`:hist`) | the last N inputs |
| `:ast CODE` · `:tokens CODE` | syntax tree / tokens |
| `:reset` | clear all variables and functions |
| `:perms` · `:env` | permissions / versions and current folder |
| `:cd` `:pwd` `:ls` `:clear` | the shell words, as commands |

**Editing**: Tab completes names, fields (`p.<Tab>`), `:commands` and paths; Up/Down recall history (kept in
`~/.apsi/repl_history`); Ctrl+C cancels the current input. Colours turn off with `NO_COLOR=1`.
The REPL uses the reference interpreter and the same permissions flags as `apsi run`
(`apsi repl --allow-net`).

---

## 24. The command line

```text
apsi [COMMAND] [OPTIONS]
```

### 24.1 Shortcuts

| You type | Means |
|---|---|
| `apsi` | REPL (or, when a program is piped in, run it: `cat app.apsi \| apsi`) |
| `apsi FILE.apsi [ARGS…]` | `apsi run FILE.apsi ARGS…` |
| `apsi -e "CODE"` (`-c`, `--eval`) | `apsi eval "CODE"` |
| `apsi -i` | `apsi repl` |
| `apsi TASK [ARGS…]` | `apsi task TASK ARGS…` when TASK is in `apsi.toml` |
| `apsi -V`, `apsi --version` | version |

A mistyped command prints suggestions ("did you mean: test").

### 24.2 Running

| Command | Description |
|---|---|
| `apsi run [FILE] [ARGS…]` | check, then run. Without FILE: the project's `main`. `-` reads the program from stdin. Options: `--no-check`, `--time`, `-w/--watch`, `--native` (machine code for every function that allows it), `--threads N`, `--engine compile\|interp`, security flags |
| `apsi eval CODE…` | evaluate code; the value of a final expression is printed. `;` or `\n` separate statements |
| `apsi repl` | interactive REPL (chapter 23) |
| `apsi watch [PATHS…] [--do run\|test\|check] [--ext .apsi,.csv] [--interval S] [--no-clear] [-- FLAGS]` | re-run whenever a watched file changes |
| `apsi bench FILE [-n RUNS] [--warmup N]` | best / median / mean ± sd / worst over several runs (output suppressed) |
| `apsi build FILE [-o DIR] [--name N] [--allow-…] [--standalone] [--include PKG] [--add PATH] [--windowed]` | make an executable — see *Executables and the compile cache* below |
| `apsi compile [PATHS] [--runtime] [--stats] [--clear]` | parse, check and compile programs now, so later runs start from the compile cache |
| `apsi serve FILE [--host H] [--port P] [--only NAMES…]` | expose the program's functions as a JSON HTTP API |

#### Executables and the compile cache

**How a program runs.** `apsi app.apsi` parses the program, checks it, compiles it to Python bytecode
(engine `compile`, the default) and runs it. Parsed programs are kept in the **compile cache** — in memory, so
the Guard scan, the checker, imports and the run parse each file only once, and on disk in
`~/.apsi/cache/compiled`, so the next run of an unchanged file skips parsing. Disk entries are signed with your
per-user key (`~/.apsi/cache.key`, HMAC-SHA256); an entry that was modified or copied from another computer is
deleted, never loaded. The cache is keyed by the file's contents and by the Apsi version, so it can never serve a
stale result. `apsi compile PATHS` fills it ahead of time (and checks every file), `apsi compile --runtime` also
compiles Apsi itself to bytecode, `apsi compile --stats` / `--clear` show and empty it, and `APSI_NO_CACHE=1`
turns it off.

**Executables.** `apsi build app.apsi` turns a program into one executable file:

```bash
apsi build app.apsi                              # app.exe next to app.apsi (Windows), ./app (Linux)
apsi build train.apsi --allow-python -o dist     # the permissions are fixed when it is built
apsi build app.apsi --standalone                 # Windows: dist folder with its own Python, for PCs without Apsi
apsi build server.apsi --windowed --allow-net    # no console window
apsi build app.apsi --add data                   # pack a data folder (it sits next to the program when it runs)
```

The executable contains the program and every `.apsi` file it imports, the already parsed and checked program,
Apsi itself compiled to bytecode (so it runs exactly the Apsi version it was built with) and the permissions given
to `apsi build`. A program with check errors is never built. Arguments given to the executable go to the program
(`args()`), all of them — `app.exe --verbose x` passes `["--verbose", "x"]`. On the first start the program's
files are unpacked once into `~/.apsi/apps/NAME-ID`; later starts use them directly.

| Option | Meaning |
|---|---|
| `-o DIR`, `--name N` | output folder and name |
| `--allow-read`, `--allow-net`, `--allow-python`, `--allow-all`, `--profile`, … | permissions of the built program |
| `--standalone` | Windows: `DIR\NAME\NAME.exe` plus `python\` with the Python runtime, numpy and the Python packages the program uses (`pyimport("…")`, the torch backend — detected automatically; `--include PKG` adds more). Copy the whole folder to another PC. |
| `--windowed` | Windows: no console window |
| `--add PATH` | pack a data file or folder (inside the program's folder) |

Without `--standalone` the executable uses the Python runtime of the Apsi installation that built it, so it runs on
computers where Apsi is installed in the same place.

### 24.3 Checking, testing, formatting

| Command | Description |
|---|---|
| `apsi check [PATHS…] [-q]` | check files and folders without running (default: current folder) |
| `apsi test [PATHS…]` | run test blocks (default: the project's `tests/`); accepts security flags and `--engine` |
| `apsi fmt [PATHS…] [--check]` | format in place; `--check` only reports (exit 1 when something would change) |
| `apsi audit FILES…` | the capabilities a program uses and the minimal `--allow-*` flags |
| `apsi ast FILE` · `apsi tokens FILE` | syntax tree / tokens |

### 24.4 Projects and documentation

| Command | Description |
|---|---|
| `apsi new NAME [-t app\|lib\|physics\|ai\|server] [--no-vscode] [--force]` | create a project folder |
| `apsi init [PATH] [--name N] [-t TEMPLATE]` | turn a folder into a project |
| `apsi task [NAME] [ARGS…]` | list or run `apsi.toml` tasks |
| `apsi doc [NAME] [FILTER]` | modules, built-ins, or one name: `apsi doc`, `apsi doc nn`, `apsi doc nn adam`, `apsi doc sqrt` |
| `apsi help [COMMAND]` · `apsi COMMAND --help` | help |
| `apsi completion bash\|zsh\|fish\|powershell` | shell completion script |
| `apsi convert PATHS… [-o OUT] [--run] [--stdout] [--no-check]` | translate Python files or folders into Apsi, mapping numpy/math/torch/… onto Apsi (see [`PYTHON.md`](PYTHON.md)) |
| `apsi editors [list\|install\|uninstall\|path] [NAMES…] [--json]` | Apsi support for VS Code, Cursor, Windsurf, VSCodium, VS Code Insiders, Trae, Sublime Text, Neovim, Vim and Notepad++ (see [`EDITORS.md`](EDITORS.md)) |

Install completion: bash `eval "$(apsi completion bash)"` in `~/.bashrc`; zsh `eval "$(apsi completion zsh)"`;
fish `apsi completion fish > ~/.config/fish/completions/apsi.fish`; PowerShell
`apsi completion powershell | Out-String | Invoke-Expression` in `$PROFILE`.

### 24.5 Security, packages and the machine

| Command | Description |
|---|---|
| `apsi scan PATHS… [--av] [--json] [--fail-on LEVEL]` | malware scan of programs and folders |
| `apsi verify` | compare the installed runtime with its release manifest |
| `apsi config [--profile sandbox\|strict\|standard\|developer]` | show or set the default profile |
| `apsi pkg list \| install NAMES \| remove NAMES \| lock \| audit` | optional Python libraries (`gpu onnx torch transformers sci data plot crypto audit`, groups `ai ai-gpu science all`) |
| `apsi doctor [--json]` | check this machine against Apsi's hardware targets |
| `apsi plan PARAMS [--active N] [--gpu G] [--gpus N\|--days D]` | plan a large-model training run: layout, memory per GPU, GPU count, days ([`SCALE.md`](SCALE.md)) |
| `apsi model info\|verify\|convert FILE [OUT] [--dtype T] [--shard S] [--key K]` | inspect, verify and convert model files: GGUF, `.apsim`, safetensors ([`MODEL-FILES.md`](MODEL-FILES.md)) |
| `apsi native info\|explain\|ir\|asm\|clear-cache` | the native compiler: this machine's CPU/SIMD/threads/GPUs; which functions of a program run as machine code and why not; the LLVM IR or assembly of a function; empty the machine-code cache |
| `apsi lsp` | language server over stdio (for editors) |
| `apsi version` | version |

### 24.6 Security flags (run, test, repl, eval, bench, serve)

`--profile NAME` · `--allow-read[=DIRS]` · `--allow-write[=DIRS]` · `--allow-net[=HOSTS]` · `--allow-env[=NAMES]` ·
`--allow-python[=all]` · `--allow-ffi[=LIB@SHA256]` · `--allow-all` · `--sandbox` · `--trust` (skip Apsi Guard) ·
`--max-memory SIZE` · `--timeout TIME` · `--engine compile|interp`

### 24.7 Exit status

`0` success · `1` errors (check failed, test failed, program error) · `2` usage error (bad arguments, missing file) ·
`3` blocked by Apsi Guard · `130` interrupted (Ctrl+C) · any value passed to `exit(code)`.

### 24.8 Environment variables

| Variable | Effect |
|---|---|
| `APSI_HOME` | settings folder (default `~/.apsi`: config, REPL history, cache key) |
| `NO_COLOR` / `APSI_NO_COLOR` | no colours; `FORCE_COLOR` forces them |
| `EDITOR` / `VISUAL` | editor for `:edit` |

---

## 25. Projects and `apsi.toml`

`apsi new orbit --template physics` creates:

```text
orbit/
  apsi.toml              project settings and tasks
  src/main.apsi          entry point
  tests/main_test.apsi   tests (import "../src/main.apsi")
  README.md  .gitignore  .editorconfig
  .vscode/               tasks.json (run, test, check, format, watch), settings.json, extensions.json
```

Templates: `app` (default), `lib`, `physics`, `ai`, `server`.

```toml
[project]
name = "orbit"
version = "0.1.0"
description = ""
main = "src/main.apsi"          # what `apsi run` runs when no file is given

[run]
profile = "standard"            # sandbox | strict | standard | developer
allow = ["net"]                 # extra permissions for run/test in this project: read write net env python ffi

[tasks]                         # apsi task NAME — or simply apsi NAME
start = "apsi run src/main.apsi"
test = "apsi test tests"
check = "apsi check src"
fmt = "apsi fmt src tests"
ci = ["apsi check src", "apsi fmt --check src tests", "apsi test tests"]
```

- `apsi run`, `apsi test` and `apsi task` find `apsi.toml` in the current folder or any parent folder.
- A task is a command line or a list of command lines run in order (stopping at the first failure), in the project folder.
- Command lines that start with `apsi` use the same Apsi that runs the task, so tasks work even when `apsi` is not on PATH.
  Other command lines run in the system shell.
- Extra arguments go to the last step: `apsi task start --time`.
- Flags given on the command line override `[run]`.

---

## 26. Style guide

`apsi fmt` applies these rules automatically.

- 4 spaces per indentation level; no tabs.
- `snake_case` for variables and functions, `PascalCase` for structs, `UPPER_CASE` for module-level constants that
  you want to stand out; single-letter physics names (`m`, `v`, `F`, `T`) are fine.
- spaces around `= == -> => |> ±` and binary operators; none inside unit literals (`9.81[m/s^2]`), ranges
  (`0..10`) or keyword arguments (`f(x=1)`).
- annotate the units of public functions: `fn kinetic(m: Mass, v: Velocity) -> Energy`.
- put tests next to the code they test.
- prefer `let`; use `var` only for values that change.

---

## 27. Grammar

The grammar in EBNF (`{ }` = repeat, `[ ]` = optional, `|` = choice). `INDENT`/`DEDENT`/`NEWLINE` come from the
indentation rules of chapter 3.

```ebnf
program     = { statement } ;
statement   = let | fn_def | struct_def | if | while | for | evolve | test
            | "return" [ expr_list ] | "break" | "continue" | "pass"
            | "use" dotted [ "as" NAME ] | "import" STRING [ "as" NAME ]
            | "assert" expr [ "," expr ] | NAME "'" "=" expr
            | expr_list [ ( "=" | augop ) expr_list ] ;
let         = ( "let" | "var" ) NAME { "," NAME } [ ":" type ] "=" expr_list ;
fn_def      = { "@" expr NEWLINE } "fn" NAME [ "[" names "]" ] "(" params ")" [ "->" type ]
              ( "=" expr | block ) ;
params      = [ param { "," param } ] ;
param       = NAME [ ":" type ] [ "=" expr ] ;
struct_def  = "struct" NAME ":" NEWLINE INDENT { NAME ":" type [ "=" expr ] NEWLINE | fn_def } DEDENT ;
if          = "if" expr block { "elif" expr block } [ "else" block ] ;
while       = "while" expr block ;
for         = "for" names "in" expr block ;
evolve      = "evolve" names "over" NAME "in" expr "step" expr [ "using" NAME [ "tol" expr ] ]
              [ "record" [ "every" expr ] "into" NAME ] block ;
test        = "test" STRING block ;
block       = ":" NEWLINE INDENT { statement } DEDENT | ":" statement ;
type        = NAME [ "[" type_arg { "," type_arg } "]" ] | "[" unit "]" ;
augop       = "+=" | "-=" | "*=" | "/=" | "//=" | "%=" | "**=" | "@=" ;

expr        = "fn" "(" params ")" "=>" expr | pipe ;
pipe        = ternary { "|>" ternary } ;
ternary     = or [ "if" or "else" ternary ] ;
or          = and { "or" and } ;
and         = not { "and" not } ;
not         = "not" not | comparison ;
comparison  = range { cmp_op range } ;
cmp_op      = "==" | "!=" | "<" | "<=" | ">" | ">=" | "~=" | "in" | "not" "in" ;
range       = additive [ ( ".." | "..=" ) additive ] { "as" "[" unit "]" } ;
additive    = term { ( "+" | "-" ) term } ;
term        = unary { ( "*" | "/" | "//" | "%" | "@" ) unary } ;
unary       = ( "-" | "+" ) unary | plusminus ;
plusminus   = power [ ( "±" | "+/-" ) power ] ;
power       = postfix [ ( "**" | "^" ) unary ] ;
postfix     = primary { "(" args ")" | "[" subscripts "]" | "." NAME } ;
primary     = NUMBER [ unit ] | STRING | FSTRING | NAME | "true" | "false" | "none"
            | "(" [ expr_list ] ")" | "[" list_or_comp "]" | "{" dict_or_comp "}" ;
args        = [ arg { "," arg } ] ;
arg         = expr | NAME "=" expr ;
```
