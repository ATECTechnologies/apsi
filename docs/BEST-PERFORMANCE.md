# Maximum performance on 1.10.0 — what to do, and why

Apsi 1.10.0 runs every program on its bytecode engine (Apsi compiled to Python bytecode). It has no native compiler
yet — that is the headline of 2.0.0. The habits below are what make 1.10.0 fast, and **why** each one matters.

## 1. Let tensors do the loops

**Do:** `sum(x * y)` instead of a `for` loop adding `x[i] * y[i]`; `A @ B` instead of three nested loops.

**Why:** one tensor operation runs as optimised machine code inside NumPy for the whole array; a loop runs the
bytecode engine once per element.

**Difference:** 50–500×. On the [benchmarks](/benchmarks?v=1.10.0), which are written as explicit loops on purpose,
1.10.0 is hundreds of times slower than C — that is the cost of per-element loops.

## 2. Use the built-ins

**Do:** `sorted`, `argmax`, `cumsum`, `linalg.solve`, `evolve`, `nn.adam` … rather than writing them yourself.

**Why:** the built-ins are vectorised and tested; your own loop version runs element by element.

## 3. Keep units on, they are cheap

**Why:** units are checked by the checker before running and validated once inside `evolve`, which then integrates
on plain numbers. Leaving them out saves little and loses the safety.

## 4. Choose number formats deliberately

`f32` / `bf16` halve memory traffic for large AI models (`dtype=f32`, `@precision(f32)`).

## 5. Measure

```bash
apsi bench app.apsi
```

## 6. Need loops at C speed?

Upgrade to **2.0.0** and put `@native` on the function — the same loop-heavy code runs about 500× faster on the benchmarks. Programs written
for 1.10.0 run on 2.0.0 unchanged.
