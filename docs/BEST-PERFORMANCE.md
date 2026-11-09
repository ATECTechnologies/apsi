# Maximum performance — what to do, and why it matters

Apsi programs are correct by default. Being **fast** as well depends on a handful of habits. Each one below says
what to do, **why** (what happens inside Apsi), and how big the difference is. They are not style preferences:
skipping them is the difference between C-like speed and Python-like speed — often **100 to 3,500 times**.

## The short version

1. Put `@native` on the functions that do the heavy loops.
2. Give those functions simple, stable types: numbers, tensors, structs of numbers — not mixed lists.
3. Work on tensors in place, inside the native function, instead of building Python lists.
4. Use `parallel for` when iterations are independent.
5. Keep units at the edges; they cost nothing inside `@native`.
6. Measure with `apsi bench` and read `apsi native explain` when something stays slow.

## 1. Mark hot functions `@native`

**Do:** `@native` above functions with loops over numbers or tensors.

**Why:** without it, every operation goes through the bytecode engine, which checks types and units on each step.
With it, Apsi compiles the function through LLVM into machine code for your exact CPU (AVX2, AVX-512, FMA),
specialised for the argument types it receives; the code is cached on disk, so later runs start instantly.

**Difference:** the eight [benchmarks](/benchmarks) run about 500× faster than Apsi 1.10.0 (12× to 2,400× per program) and land next to C,
C++ and Rust.

**If it does not compile** (an unsupported construct), Apsi falls back to the bytecode engine and prints one note
with the line — the program keeps working. `@native("strict")` turns that into an error so you notice;
`apsi native explain file.apsi` lists every decision.

## 2. Keep types simple and stable inside hot code

**Do:** pass numbers, tensors (`zeros(n)`, `tensor(...)`) and structs of those. Keep one type per variable.

**Why:** the native compiler infers a machine type for every variable. `var s = 0` that later becomes a float, or
a list holding numbers *and* strings, cannot be given one machine type, so that function stays on the bytecode
engine.

**Difference:** all-or-nothing for that function — typically 50–500×.

## 3. Prefer tensors to lists in numeric loops

**Do:** `let x = zeros(n)` and `x[i] = …` instead of `xs.append(…)` on a list.

**Why:** a tensor is one contiguous block of memory of one element type; the compiler reads and writes it
directly, removes bounds checks it can prove safe, and vectorises the loop. A list holds separate objects.

**Difference:** 10–100× in tight loops, plus far less memory.

## 4. Use `parallel for` for independent iterations

**Do:** `parallel for i in 0..n:` when iteration `i` does not read what iteration `j` writes.

**Why:** the iterations are split across all CPU cores. Sums, products, `min` and `max` of outer variables are
combined safely and give bit-identical results on any number of cores; a loop that writes other outer variables
runs sequentially with a note explaining why — so you cannot get a wrong answer silently.

**Difference:** close to the number of cores (8 cores → about 7×).

## 5. Units: declare them at the boundaries

**Do:** annotate the function's parameters and result (`fn step(x: Length, dt: Time) -> Velocity`).

**Why:** units are checked when the function is compiled and once per call; the machine code itself works on plain
SI numbers. You keep the safety and pay nothing in the inner loop.

**Difference:** zero cost — but skipping units loses the checks that catch a whole class of physics bugs.

## 6. Choose the number format deliberately

**Do:** `f64` (default) for physics; `f32` or `bf16` for large AI models (`@precision(f32)`, `dtype=f32`).

**Why:** half the bytes per number means half the memory traffic; most AI workloads are memory-bound.

**Difference:** up to 2× for memory-bound work, and models that fit in memory at all. See
[number formats](/docs/2.4.4/performance).

## 7. Measure instead of guessing

```bash
apsi bench app.apsi            # repeated timed runs
apsi native explain app.apsi   # which functions are native, and why not
apsi native asm app.apsi FN    # the machine code, if you want to see it
python3 benchmarks/heavy/run.py     # compare with C, C++, Rust, Go and Mojo on your machine — live table
```

**Why:** the slow part is rarely where it seems. Two minutes with `apsi bench` saves hours of rewriting the wrong
function.

## Checklist before you publish a library

- [ ] hot functions `@native`, and `apsi native explain` shows them compiled
- [ ] no lists of mixed types in hot paths
- [ ] units on public functions
- [ ] tests (`test "…":` blocks) pass with `apsi test`
- [ ] `apsi bench` numbers noted in the README, so users know what to expect
