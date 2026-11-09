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
| Tables | a columnar engine, 3–24× faster than pandas | [Tables](DATAFRAME.md) |
| Databases | PostgreSQL, Redis, SQLite without drivers | [Databases](DATABASES.md) |
| Web | a 3× faster HTTP server, WebSocket | [Servers](SERVER.md) |
| Configuration | `.env` files, secrets, Python virtual environments | [Configuration](ENV.md) |
| Debugger | breakpoints and stepping in VS Code or the terminal | [Debugging](DEBUGGING.md) |

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

PostgreSQL (SCRAM, TLS, prepared statements, LISTEN/NOTIFY, pools), Redis (pipelines, pub/sub) and SQLite. Parameters
are never pasted into SQL. MySQL and MongoDB are not built in; use their Python drivers through `pyimport`.

## 7. Web

* The HTTP server runs fast routes directly on its event loop and keeps slow or blocking ones on worker threads,
  judged by CPU time so a busy server is not misled: 20,300–20,700 requests per second from 50 to 1,000 connections
  on 2 cores (2.0.0: 6,700–7,700).
* WebSocket routes, rooms and a client: `@app.websocket("/chat")`, `app.room("lobby").send(msg)`,
  `web.websocket_connect(url)`.

## 8. Configuration and the debugger

* `use env`: `.env` files, typed settings (`env.int("PORT", 8080)`), secrets that print as `******`;
  `apsi run --env-file .env`, `--venv .venv`.
* `apsi debug app.apsi` in the terminal, or F5 in VS Code: breakpoints with conditions, hit counts and log points,
  stepping, variables, tensor summaries, stop on errors.

## 9. Faster tools

* A cold compile of a 20,000-line program: 3.7 s → 2.5 s; unchanged programs skip the checker (signed marker).
* `apsi convert` of 29,022 lines: 4.5 s.

## Upgrading from 2.0.0

Install 2.4.4 over 2.0.0 and run `apsi test`; everything written for 2.0.0 runs unchanged. Python files converted with
2.0.0 can be converted again (`apsi convert file.py`) to get the exact behaviour; a converted module records the Apsi
version that wrote it and is converted again automatically when the version changes.
