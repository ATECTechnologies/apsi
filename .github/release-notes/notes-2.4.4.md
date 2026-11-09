### Python conversion: exact, complete
- **`apsi convert` keeps Python's semantics byte for byte.** Converted files start with `use pycompat`, which gives
  them Python's object model, scoping and built-ins. Real classes (inheritance from any library class, metaclasses,
  descriptors, `__slots__`, `__init_subclass__`, `__class_getitem__`, name mangling of `__private` names),
  dataclasses, enums, `with`, `try/except/else/finally`, `raise … from`, generators with `send`/`throw`, closures,
  `global`/`nonlocal`, chained comparisons, walrus, star targets, `del`, f-strings with every conversion and nested
  spec — all with Python's behaviour.
- **Every Python 3.13 construct converts:** `match`/`case` (all pattern kinds and guards), `except*` and exception
  groups, `async def`/`await`/`async for`/`async with` and async comprehensions, comprehensions that assign to
  attributes or indexes, `type X = …` and generic functions/classes (`def f[T]`), `from x import *`, `a[*idx]`,
  `a[1,]`, error kinds computed at run time (`except KINDS:`), names that are Apsi keywords (written in backticks
  and kept as the same name), names read before they are bound (`UnboundLocalError`, `NameError` as in Python).
- **Verified on the standard library:** 34 pure-Python standard modules (`_pydecimal`, `_pydatetime`, `argparse`,
  `tomllib`, `json`, `pickle`, `dataclasses`, `enum`, `statistics`, `ipaddress` …, about 38,000 lines) convert and
  print byte-identical output to CPython (`tests/test_stdlib_convert.py`), alongside 15 feature programs and a
  23,769-line AI training program with four companion modules (identical outputs, files, process-pool results and
  training losses).
- Faithful conversion is the default (`--idiomatic` keeps the old mapping onto Apsi built-ins); local modules and
  packages convert with the program; a converted module is reused only when it is newer than its source and was
  written by the same Apsi version; `apsi run file.apsi -- args` passes `sys.argv`; `SystemExit` behaves as in Python.
- The sandbox stays on: `sys`, `os`, `pathlib` and `numpy` are permission-checked stand-ins; interpreter internals
  (frames, `__globals__`, `__subclasses__`, `eval`, `exec` …) need `--allow-python=all`.

### Language
- New syntax (available everywhere, with Python's meaning): `class`, `with`, `del`, `global`, `nonlocal`, `match`,
  `async`/`await`, set literals and comprehensions, generator expressions, `is`/`is not`, chained comparisons,
  `| & ~ << >>` and their assignments, `:=`, `for … else`, `while … else`, bytes literals, `\x \u \U` escapes,
  positional-only and keyword-only parameters, `` `name` `` for identifiers that are keywords.

### Real time (`use rt`)
- Loops at 1–2 kHz with microsecond lateness (`rt.loop`, `rt.ticks`, `rt.scheduler` for many rates on one loop),
  missed ticks counted and skipped instead of drifting, the garbage collector paused, 1 ns timer slack on Linux and
  1 ms timer resolution on Windows while a loop runs.
- Video pipelines (`rt.video`): capture thread, newest-frame-only hand-off, adaptive resolution under sustained
  overload, error isolation, a report with fps, latency, jitter and dropped frames. **Any size from 16×16 to
  16384×16384** with 17 named sizes (`vga`, `hd`, `hd+`, `fhd`, `qhd`, `4k` …). Measured CPU-only on 2 cores:
  640×480 at 976 fps, 1600×900 at 372 fps, 1920×1080 at 264 fps, 3840×2160 at 65 fps; capped runs hit 240/120/90/60/30
  fps exactly with 1–12 ms latency.
- Frame helpers on OpenCV, NumPy or the GPU (`rt.backend("gpu")` through PyTorch or CuPy).

### RAM, SSD, CPU and GPU (`use memory`)
- `memory.array` stays in RAM while it fits a budget and moves to a memory-mapped SSD file when it does not;
  `memory.chunks` and `memory.map` process data larger than RAM on all cores with read-ahead; `memory.store` and
  `@memory.cached` are two-tier caches (RAM, then SSD) with signed entries; `memory.fit_batch` halves a batch on
  out-of-memory (CPU or CUDA) until it fits; `memory.status`, `memory.devices`, `memory.best_device`.

### Fewer AI tokens (`use tokens`)
- Exactly reversible transforms for what you send to language models: `pack`/`unpack` (lists of records as
  tables: 57–63 % fewer tokens than JSON), `dedupe`/`expand` (repeated lines once: 59 % on a log), `diff`/`patch`
  (only the changed lines: 99.7 % on an edited document), `best`, and `count` (exact with tiktoken).

### Data, databases, web
- **`use dataframe`**: a columnar table engine (filter, select, group by, join, sort, windows, CSV/JSON/JSONL, a C
  CSV parser). Against pandas on 2M rows: filters 24×, group-by 4.2–4.3×, joins 3.1×, sorting 1.8× faster, 4× less memory.
- **`use db`**: PostgreSQL (v3 protocol, SCRAM, TLS, parameters, prepared statements, LISTEN/NOTIFY, pools), Redis
  (pipelines, pub/sub) and SQLite — no driver to install.
- **WebSocket** server routes, rooms and a client (`app.websocket`, `web.websocket_connect`), RFC 6455 complete.
- **The HTTP server is about 3× faster** (an asyncio protocol; routes that prove fast run on the event loop, measured
  by CPU time so a busy server never misleads it; slow or blocking routes stay on the workers): 20,300–20,700
  requests per second from 50 to 1,000 concurrent connections on 2 cores, up from 6,700–7,700.
- **`use env`**: `.env` files, typed settings and secrets that print as `******`; `apsi run --env-file`, `--venv`
  (or a `.venv` next to the program) for `pyimport`.

### Tools
- **Debugger**: `apsi debug` in the terminal and F5 in VS Code-family editors (`apsi dap`): breakpoints with
  conditions, hit counts and log points, stepping, the call stack, variables, tensor summaries, stop on errors.
- **Faster programs:** the main program starts with the loaded runtime frozen out of the cycle collector and a
  50,000-allocation first generation: object-heavy code runs about 8 % faster than on CPython (`APSI_GC=default`
  turns it off). Module attribute access (`math.sqrt`) is a plain attribute lookup.
- **Faster start and compile:** start-up about 30 % faster (heavy modules load on first use); a cold compile of a 20,000-line program 3.7 → 2.5 s (faster
  lexer and parser, the Guard scan 3× faster, a signed "checked clean" marker so unchanged programs are not checked
  again).

**Source code only.** Installers are published at https://apsi.dev/download.
