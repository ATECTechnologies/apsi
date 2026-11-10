# What's new in Apsi 2.4.4

This page walks through everything 2.4.4 adds, with a working example for each part, the numbers we measured,
and the limits you should know about. Every example runs as shown with Apsi 2.4.4. The full list of changes is in
the [changelog](https://github.com/ATECTechnologies/apsi/blob/main/CHANGELOG.md); each part links to its complete
guide.

| Part | In one line | Guide |
|---|---|---|
| Python, exactly | any Python 3.13 program converts and behaves byte for byte the same | [From Python](PYTHON.md) |
| Real time | 1–2 kHz control loops and video from 640×480 to 4K | [Real time](REALTIME.md) |
| RAM, SSD, GPU | arrays that move to the SSD instead of running out of memory | [Memory](MEMORY.md) |
| Fewer AI tokens | exactly reversible packing of what you send to language models | [Tokens](TOKENS.md) |
| Hugging Face | download and run models (BERT family, GPT-2, Llama, Mistral, Qwen, Gemma) without Python packages | [Hub](HUB.md) |
| Tables | a columnar engine, 3–24× faster than pandas | [Tables](DATAFRAME.md) |
| Databases | PostgreSQL, MySQL / MariaDB, MongoDB, Redis, SQLite without drivers | [Databases](DATABASES.md) |
| Web | an HTTP server 2–3× faster than 2.0.0 (synthetic test), WebSocket | [Servers](SERVER.md) |
| Configuration | `.env` files, secrets, Python virtual environments | [Configuration](ENV.md) |
| Debugger | breakpoints and stepping in VS Code or the terminal | [Debugging](DEBUGGING.md) |
| Number formats | f64, f32, f16, bf16, FP8 and int8 tensors in `@native` and `@gpu` code | [Number formats](NUMBERS.md) |
| Long-range graphs | graph transformers, one-step global attention, k-hop and mixed-hop layers | [Graphs](GRAPHS.md) |
| GPU | one thread per element, automatic CPU/GPU choice, Tensor Cores through cuBLAS | [Native](NATIVE.md) |
| Matrix products | loops written the textbook way run on BLAS: 7–10× faster than C loops | [Performance](PERFORMANCE.md#matrix-products-written-as-loops) |
| Same speed everywhere | SIMD lanes written by Apsi itself; `@native("fma")` | [below](#16-the-same-speed-on-every-processor) |
| Speed overall | 3.83–7.31× faster than 2.0.0 on the same programs (geometric mean) | [below](#how-much-faster-than-200) |

## 1. Python, exactly

### What it does

`apsi convert program.py` writes `program.apsi` (and an `.apsi` file for every module next to it). The converted
program prints the same text, raises the same errors, writes the same files and trains to the same losses as the
Python program. The first line of a converted file is `use pycompat`: it gives the file Python's object model,
scoping and built-ins while the text stays Apsi.

```bash
apsi convert train.py            # train.apsi + the local modules it imports
apsi run train.apsi --allow-python -- --epochs 3      # arguments after -- go to sys.argv
```

### What "exactly" covers

Everything in Python 3.13: classes with metaclasses, descriptors, `__slots__`, properties and every special method;
dataclasses and enums; `with` and `async with`; `try / except / except* / else / finally`; generators with `send`
and `throw`; `async def`, `await`, async comprehensions; `match` with every pattern kind; closures with `global` and
`nonlocal`; walrus; star targets; `del`; f-strings with every conversion; generic functions and classes
(`def first[T](xs: list[T])`), `type X = …`; `from x import *`; name mangling of `__private` names; names that read
before they are bound raise `UnboundLocalError` and `NameError` exactly as in Python.

How we check it: `tests/test_stdlib_convert.py` converts **34 pure-Python modules of the standard library itself**
(about 38,000 lines: `_pydecimal`, `_pydatetime`, `argparse`, `tomllib`, `json`, `pickle`, `dataclasses`, `enum`,
`statistics`, `ipaddress`, `difflib` …) together with a program that uses them, and requires byte-identical output to
CPython. `tests/test_python_compat.py` does the same for 16 feature programs, and a 29,022-line AI program with its
modules gives identical command output, files, multi-process results and training losses.

### Speed

Converted code runs on the same bytecode interpreter as CPython. Object-heavy code is about 8 % faster (Apsi keeps
the runtime it loaded out of the cycle collector and collects the youngest generation less often); arithmetic in
Python loops runs at CPython's speed. For loops that must be fast, put `@native` on the function: it becomes machine
code (see [native code](NATIVE.md)).

### What differs on purpose

* An uncaught error names the `.apsi` file and line; the error type and message are Python's.
* The sandbox stays on: files need `--allow-read` / `--allow-write`, libraries such as `torch` need `--allow-python`,
  and modules that reach the operating system (`subprocess`, `socket`, `ctypes` …) need `--allow-python=all`.
  `apsi convert` prints the flag a program needs.

## 2. Real time (`use rt`)

```apsi
use rt

fn control(tick, t):
    motor.set(pid.update(sensor.read()))

print(rt.loop(1000, control, seconds=10))           # rate, lateness percentiles, missed ticks

let cam = rt.camera(0, "fhd", fps=60)                # or rt.synthetic("1600x900", fps=240)
fn process(frame, info):
    if info.prev_gray is not none:
        let box = rt.motion(info.prev_gray, rt.gray(frame))
print(rt.video(cam, process, fps=90, seconds=30, adapt=true))
```

* **Loops:** sleep, then spin on the high-resolution clock for the last 1.5 ms, so ticks start within microseconds.
  A tick that cannot start on time is counted and skipped instead of making the rhythm drift.
* **Video:** the newest frame is always the one processed, so latency never grows; `adapt=true` lowers the working
  resolution under sustained overload and raises it again; one failing frame never stops the pipeline.
* **Sizes:** any width × height from 16 to 16384, and names: `vga`, `svga`, `hd`, `wxga`, `hd+`, `fhd`, `qhd`, `4k` and
  more (`rt.sizes`).
* **Measured** CPU-only on 2 cores: 640×480 at 976 fps, 1600×900 at 372 fps, 1920×1080 at 264 fps, 3840×2160 at
  65 fps; capped runs reach 240/120/90/60/30 fps exactly with 0.8–12 ms latency. 2000 Hz loops: lateness p50 0.3 µs.

## 3. RAM, SSD, CPU and GPU (`use memory`)

```apsi
use memory
print(memory.status())                               # RAM, swap, SSD space, cores, load, GPUs
memory.budget(16)                                    # GB of RAM for big arrays
let x = memory.array([200_000_000, 8], "float32")   # RAM if it fits, otherwise a memory-mapped SSD file
let y = memory.map(fn(b) => b * 2.0, x)              # all cores, larger than RAM is fine
let cache = memory.store("embeddings", ram_mb=512, disk_gb=50)
let loss, batch = memory.fit_batch(fn(b) => train_step(b), 512)   # halves the batch on out-of-memory
```

Spill files and stores live in `~/.apsi` and need no permission; a store at a path you choose needs `--allow-write`.
Pickled cache entries are signed, so a file changed outside Apsi is never loaded.

## 4. Fewer AI tokens (`use tokens`)

```apsi
use tokens
let text = tokens.pack(rows)                         # a list of records → a compact table
assert tokens.unpack(text) == rows                   # always exact
print(tokens.savings(json_str(rows, indent=2), text))
```

| transform | removes | measured |
|---|---|---|
| `pack` / `unpack` | repeated JSON keys and quoting | 57–63 % fewer tokens |
| `dedupe` / `expand` | repeated lines | 59 % on a log |
| `diff` / `patch` | resending an edited document | 99.7 % |

Nothing is summarised or approximated: each transform has an exact reverse. Text without redundancy does not shrink,
and `tokens.best()` then returns it unchanged.

## 5. Tables (`use dataframe`)

```apsi
use dataframe as df
let sales = df.read_csv("sales.csv")
let top = sales.group_by("customer").agg(total=df.col("amount").sum()).sort("total", descending=true).head(10)
print(top.join(df.read_csv("customers.csv"), on="customer", how="left"))
```

Against pandas 2 on 2 million rows: filters 24×, group-by 4.2–4.3×, joins 3.1×, sorting 1.8× faster; 4× less memory.

## 6. Databases (`use db`)

```apsi
use db
use env
let pg = db.postgres(env.secret("DATABASE_URL"))     # needs --allow-net=HOST:PORT
for row in pg.query("select id, name from item where price < $1", 10):
    print(row.id, row.name)
let r = db.redis("redis://localhost:6379/0")
```

PostgreSQL (SCRAM, TLS, prepared statements, LISTEN/NOTIFY, pools), MySQL / MariaDB (MySQL 8 and 5.7
authentication, TLS), MongoDB (BSON, SCRAM-SHA-256, cursors), Redis (pipelines, pub/sub) and SQLite. Values never
become SQL text unescaped.

```apsi
let my = db.mysql("mysql://shop:secret@localhost/shop")
let items = db.mongo("mongodb://app:secret@localhost/shop").collection("items")
```

## 7. Web

* The HTTP server runs fast routes directly on its event loop and keeps slow or blocking ones on worker threads,
  judged by CPU time so a busy server is not misled: 20,300–20,700 requests per second from 50 to 1,000 connections
  on 2 cores (2.0.0: 6,700–7,700). This is a synthetic test of the server's own cost per request (a fixed JSON
  reply over loopback); on the same test Node.js's bare `http` module is about 2.7× faster. See
  [Servers](SERVER.md) for what it does and does not measure.
* WebSocket routes, rooms and a client: `@app.websocket("/chat")`, `app.room("lobby").send(msg)`,
  `web.websocket_connect(url)`.

## 8. Configuration and the debugger

* `use env`: `.env` files, typed settings (`env.int("PORT", 8080)`), secrets that print as `******`;
  `apsi run --env-file .env`, `--venv .venv`.
* `apsi debug app.apsi` in the terminal, or F5 in VS Code: breakpoints with conditions, hit counts and log points,
  stepping, variables, tensor summaries, stop on errors.

## 9. Licence

The Apsi License 2.0 allows any use, including commercial work, and keeps one rule: Apsi itself may not be sold or
offered as a paid service without a licence from ATEC. The programs, libraries and models you build are yours. See
[Licensing](LICENSING.md) for examples.

## 10. Faster tools

* A cold compile of a 20,000-line program: 3.7 s → 2.5 s; unchanged programs skip the checker (signed marker).
* `apsi convert` of 29,022 lines: 4.5 s.

## 11. Hugging Face without Python

`use hub` downloads models from the Hugging Face Hub (with a token for private and gated models), tokenizes text
(WordPiece, BPE, Unigram) and runs the models natively: embeddings, classification, named entities, fill-mask, and
text generation with GPT-2, Llama, Mistral, Qwen and Gemma. See [Hub](HUB.md).

```apsi
use hub
let sentiment = hub.pipeline("sentiment-analysis", "distilbert/distilbert-base-uncased-finetuned-sst-2-english")
print(sentiment("Apsi is a joy to use"))
print(hub.model("HuggingFaceTB/SmolLM2-135M").generate("The three laws of motion are", max_new_tokens=40))
```

## 12. Data-parallel training

`fit(model, params, X, y, …, workers=4)` trains on four processes: each batch is split into four shards, the
gradients are averaged by shard size (the same update as one process) and one optimizer step is taken.

## 13. Matrix products written as loops

```apsi
fn matmul_into(a, b, c):
    for i in 0..a.shape[0]:
        for k in 0..a.shape[1]:
            let aik = a[i, k]
            for j in 0..b.shape[1]:
                c[i, j] += aik * b[k, j]
```

Loop nests like this one (any loop order, or the dot-product form with `var s = 0.0`) are recognised as a matrix
product and run with the tuned BLAS kernel, in the compiled engine and in `@native` code. On a 2-core VM a 2048×2048
product takes 234 ms (73 GFLOP/s) — the same loops in C with `gcc -O3 -march=native` take 2,409 ms, and in Python
they would take minutes. When the fast path would not give the loops' exact behaviour (integer arrays, `c`
overlapping an input, bounds beyond the arrays) the loops run as written. See
[Performance](PERFORMANCE.md#matrix-products-written-as-loops) and `benchmarks/matmul`.

## 14. Number formats down to FP8, and a smarter GPU

```apsi
@gpu
fn matmul(a, b, c):                       # a, b, c: f16 or bf16 tensors
    parallel for i in 0..a.shape[0]:
        for j in 0..b.shape[1]:
            var acc: f32 = 0.0            # 16-bit data, 32-bit sums
            for k in 0..a.shape[1]:
                acc += a[i, k] * b[k, j]
            c[i, j] = acc
```

* f64, f32, f16, bf16, FP8 (E4M3, E5M2) and int8 tensors work in `@native` and `@gpu` code. Small formats are
  stored small and computed in f32, with correct rounding when stored ([Number formats](NUMBERS.md)).
* The loop above gets one GPU thread per element of `c`; with cuBLAS installed it runs as a single GEMM on the
  Tensor Cores. Launches too small for the GPU run on the CPU automatically.
* Outer loops with a reduction inside run four iterations at once in SIMD registers, with bit-identical results
  ([Performance](PERFORMANCE.md#outer-loops-four-at-a-time-exact-simd-for-reductions)).

## 15. Graphs that reach far

```apsi
let g = graph(edges=edges, n=n, types=kinds, reverse=true, batch=graph_of_node)
let model = graph_transformer_net(16, 64, 2, layers=4, heads=4, relations=g.relations, global_every=2)
```

Message passing moves one hop per layer and squeezes what a node heard into one vector. `graph_transformer`
(pre-norm residual neighbour attention, per relation and per hop distance) keeps values alive over many layers;
`global_attention` lets every node read any node of its graph in one step; `k_hop` and `mix_hop` read several
hops at once; `graph_memory` hands a decoder the most salient nodes of each graph. See [Graphs](GRAPHS.md).

## 16. The same speed on every processor

`@native` already ran independent outer iterations four at a time ([Performance](PERFORMANCE.md#outer-loops-four-at-a-time-exact-simd-for-reductions)),
but left the packing into SIMD registers to LLVM. With LLVM 22 that packing failed on AVX2 processors (AMD Zen 2
and 3, Intel without AVX-512 — most desktop PCs): the n-body benchmark took 127 ms on an EPYC 7763, almost three times
as long as C. Apsi now writes the vector instructions itself. Every lane still performs exactly its own operations in its
own order, so results are bit-for-bit what they were.

| n-body (2,000 bodies × 10 steps) | before | now |
|---|---:|---:|
| AMD EPYC 7763 (Zen 3, AVX2), same runner | 127.5 ms | 46.0 ms (gcc -O3: 44.2 ms) |
| Intel Xeon (AVX-512) | 74.5 ms | 70.9 ms |

`@native("fma")` is new: every check stays on and the order of operations is kept, but `a * b + c` may become one
fused multiply-add, which is what gcc and clang do for C by default. The ODE benchmark (one long chain of dependent
arithmetic) takes 288 ms instead of 440 ms, against 387 ms for gcc -O3.

Every commit of this work was measured on one machine against the ones before it
(`benchmarks/commits/measure.py`, Actions → commits; [results](https://github.com/ATECTechnologies/apsi/blob/main/benchmarks/commits/results-2.4.4.md)).
From the start of the work to now, on the same AMD EPYC 7763: n-body **4.0×**, the ODE **1.5×**, k-means **1.27×**
and compiling **1.21×** faster, and no row slower. Numbers from different GitHub runs are not comparable with each
other: GitHub picks the processor, and the same C program takes 14 ms on one runner model and 44 ms on another.

## How much faster than 2.0.0

The same programs on both versions, on the same machine, with identical results. Measured on a GitHub Actions runner
(AMD EPYC, native compiler installed) and on a 2-core VM without the native compiler
([`benchmarks/versions`](https://github.com/ATECTechnologies/apsi/tree/main/benchmarks/versions)):

| Program | 2.0.0 | 2.4.4 | runner | 2-core VM ¹ |
|---|---:|---:|---:|---:|
| matmul 300, loops (compiled engine) | 37,909 ms | 5.0 ms | **7658.43×** | 6374.16× |
| 200k objects: create and sum | 3,793 ms | 419 ms | **9.05×** | 9.26× |
| fib(25), recursive calls | 82.3 ms | 24.4 ms | **3.37×** | 2.26× |
| matmul 512, @native loops | 26.7 ms | 9.8 ms | **2.73×** | 300.48× |
| web server, 64 connections | 5,184 req/s | 12,554 req/s | **2.42×** | 3.06× |
| web server, 256 connections | 6,037 req/s | 12,250 req/s | **2.03×** | 3.18× |
| word count, 260k words | 324 ms | 202 ms | **1.6×** | 1.41× |
| start-up: apsi run hello.apsi | 232 ms | 173 ms | **1.34×** | 6.69× |
| 200k f-strings and join | 234 ms | 189 ms | **1.23×** | 1.26× |
| integer loop, 3M iterations | 198 ms | 171 ms | **1.16×** | 0.99× |
| apsi check, 43 example programs (1,594 lines), empty cache | 9,967 ms | 9,613 ms | **1.04×** | 4.9× |
| harmonic sum 50M, @native | 70.1 ms | 70.1 ms | **1.0×** | 1.03× |

Geometric mean: **3.83× faster** (runner), **7.31×** (2-core VM). `@native` code was already at C
speed in 2.0.0 and stays there. Reproduce with
`python3 benchmarks/versions/compare.py --old ../apsi-2.0.0 --new . --web`.

¹ Without the native compiler; 2.0.0 also started much more slowly there.

## Upgrading from 2.0.0

Install 2.4.4 over 2.0.0 and run `apsi test`; everything written for 2.0.0 runs unchanged. Python files converted with
2.0.0 can be converted again (`apsi convert file.py`) to get the exact behaviour; a converted module records the Apsi
version that wrote it and is converted again automatically when the version changes.
