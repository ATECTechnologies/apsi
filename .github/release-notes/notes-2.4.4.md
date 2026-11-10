## 2.4.4 — Python, exactly; real time; data, databases and the web

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

### Hugging Face without Python (`use hub`)
- Search, download (cached by commit, resumed, SHA-256 checked, offline mode) and load safetensors / GGUF weights
  from the Hugging Face Hub; access tokens for private and gated models, sent only to the Hub itself.
- Native `tokenizer.json` engines: WordPiece (BERT family), byte-level BPE (GPT-2, Llama 3, Qwen),
  SentencePiece-style BPE with byte fallback (Llama 2, Mistral) and Unigram (T5, XLM-R).
- **Graphs that reach far** (`use layers`): `graph_transformer` (pre-norm residual neighbour attention with
  relations and hop biases), `global_attention` (one hop to any node of the same graph, landmark keys above
  `max_keys`), `mix_hop` (several hop distances in parallel), `k_hop` shortcut graphs, `graph_memory` (the k most
  salient node states per graph), `graph_transformer_net`, `hop_distances`, `eccentricity`; `graph(…, types=,
  reverse=true)` for relations in both directions. `docs/GRAPHS.md`.
- **Number formats in native and GPU code.** `@native` and `@gpu` functions accept f64, f32, f16, bf16, FP8
  (E4M3, E5M2) and int8 tensors. 16- and 8-bit values are read into f32, computed in f32 and rounded once (to
  nearest, ties to even) when stored; the conversions are bit-identical to NumPy and ml_dtypes for every 16-bit
  and 8-bit pattern and work the same on every CPU and GPU. `var acc: f32 = 0.0` types a scalar's precision;
  `zeros(n, dtype=bf16)` and `dtype=fp8_e4m3` create real tensors (ml_dtypes). `docs/NUMBERS.md`.
- **GPU.** `to_gpu(x)` keeps tensors in GPU memory between calls (no copy per call; `to_host` brings them back). One thread per element for row loops with an independent inner loop (collapse); fewer than 4,096
  iterations run on the CPU and larger sizes are timed once on each and then use the faster; matrix-product loop
  nests run as one cuBLAS GEMM when cuBLAS is installed (Tensor Cores with f32 accumulation for f16/bf16).
- **Exact outer-loop vectorisation.** Independent outer iterations with a reduction along the inner loop run four
  at a time with bit-identical results (n-body forces, k-means assignment): n-body 2.8× faster relative to C,
  now ahead of Rust; k-means now faster than C. `APSI_JAM=0` turns it off.
- **Platforms.** The native compiler works on ARM64 (Linux and Apple silicon); every push is tested on linux-x64,
  linux-arm64, windows-x64, darwin-x64 and darwin-arm64.
- **Faster ordinary code.** Struct construction 9× faster (type checkers are built once per annotation, not per
  object); `Int`/`Float` argument and return checks, and `if`/`while` conditions on plain booleans, skip the checker
  call (recursive calls 2.3× faster); attribute access on structs, lists, dicts and strings takes a short path.
  Measured against 2.0.0 on the same machine: `benchmarks/versions` (geometric mean 3.83× faster on a
  GitHub runner with the native compiler, 7.31× on a 2-core VM without it).
- **Matrix products written as loops run with BLAS.** Three nested loops computing `c[i, j] += a[i, k] * b[k, j]`
  (any loop order, a hoisted `a[i, k]`, or the dot-product form with an accumulator) are recognised in the compiled
  engine and in `@native` code and run with the tuned BLAS kernel: 7–10× faster than the same loops in C with `-O3`,
  about 1,000× faster than before in the compiled engine. The loops remain the fallback (integer arrays, aliasing,
  out-of-range bounds) so behaviour never changes; `APSI_IDIOMS=0` turns it off. `benchmarks/matmul`.
- **Models run natively** (`hub.model`, `hub.pipeline`): BERT, DistilBERT, RoBERTa, XLM-R and sentence-transformers
  (embeddings, similarity, classification, named entities, fill-mask); GPT-2, Llama, Mistral, Qwen 2/3 and Gemma
  (text generation with a key/value cache, greedy or sampled, streaming).

### Fewer AI tokens (`use tokens`)
- Exactly reversible transforms for what you send to language models: `pack`/`unpack` (lists of records as
  tables: 57–63 % fewer tokens than JSON), `dedupe`/`expand` (repeated lines once: 59 % on a log), `diff`/`patch`
  (only the changed lines: 99.7 % on an edited document), `best`, and `count` (exact with tiktoken).

### Data, databases, web
- **`use dataframe`**: a columnar table engine (filter, select, group by, join, sort, windows, CSV/JSON/JSONL, a C
  CSV parser). Against pandas on 2M rows: filters 24×, group-by 4.2–4.3×, joins 3.1×, sorting 1.8× faster, 4× less memory.
- **`use db`**: PostgreSQL (v3 protocol, SCRAM, TLS, parameters, prepared statements, LISTEN/NOTIFY, pools),
  MySQL / MariaDB (caching_sha2 and native authentication, TLS), MongoDB (OP_MSG, BSON with decimal128,
  SCRAM-SHA-256/1, cursors), Redis (pipelines, pub/sub) and SQLite — no driver to install.
- **Data-parallel training:** `fit(…, workers=N)` splits every batch across N processes and averages the gradients
  (the same step as one process).
- **WebSocket** server routes, rooms and a client (`app.websocket`, `web.websocket_connect`), RFC 6455 complete.
- **The HTTP server is about 3× faster** (an asyncio protocol; routes that prove fast run on the event loop, measured
  by CPU time so a busy server never misleads it; slow or blocking routes stay on the workers): 20,300–20,700
  requests per second from 50 to 1,000 concurrent connections on 2 cores, up from 6,700–7,700.
- **`use env`**: `.env` files, typed settings and secrets that print as `******`; `apsi run --env-file`, `--venv`
  (or a `.venv` next to the program) for `pyimport`.

### Tools
- **Apsi License 2.0** replaces the Non-Commercial License 1.0: any use is allowed, including commercial work;
  only selling Apsi itself needs a licence. `docs/LICENSING.md` explains it with examples.
- The formatter keeps hand-aligned continuation lines and comment columns; `@native` and `@gpu` functions run as
  ordinary functions in the interpreter engine instead of stopping it.
- **Debugger**: `apsi debug` in the terminal and F5 in VS Code-family editors (`apsi dap`): breakpoints with
  conditions, hit counts and log points, stepping, the call stack, variables, tensor summaries, stop on errors.
- **Faster programs:** the main program starts with the loaded runtime frozen out of the cycle collector and a
  50,000-allocation first generation: object-heavy code runs about 8 % faster than on CPython (`APSI_GC=default`
  turns it off). Module attribute access (`math.sqrt`) is a plain attribute lookup.
- **Faster start and compile:** start-up about 30 % faster (heavy modules load on first use); a cold compile of a 20,000-line program 3.7 → 2.5 s (faster
  lexer and parser, the Guard scan 3× faster, a signed "checked clean" marker so unchanged programs are not checked
  again).

### Faster web server
- Less work per request (request ids without a system call, merged security headers, literal routes without a
  regular expression, one timer per connection, one-regex header validation, fast routes timed by the wall clock
  only): 19,200 → 22,400 requests/s per process on a 2-vCPU VM.
- uvloop when installed (`pip install "apsi[web]"`); `app.run(processes="auto")`; `stop()` also ends uvloop's
  `serve_forever`. Synthetic test on a 4-vCPU EPYC 7763: 1 process 14,600–15,700 req/s, `processes="auto"`
  38,800–41,500, Node.js 22 (bare `http`, one process) 23,400–24,100.

### Syntax and library
- `a ?? b`, `a?.b`, `a?.f(x)`, `a?[i]` in both engines, the checker, the formatter and the editor grammars.
- `use collections`: `Counter`, `DefaultDict`, `Deque`, `Heap`, `LRUCache`, `OrderedSet`, `BiMap`; built-in
  `first`, `last`, `partition`, `take_while`, `drop_while`, `scan`, `fold`, `flat_map`, `zip_longest`, `interleave`,
  `distinct_by`, `min_by`, `max_by`, `sum_by`, `frequencies`, `compose`, `partial` (`tests/test_new_syntax.py`).

### Smarter memory (`use memory`)
- Container-aware: the memory cgroup's limit (v1 and v2) is used when it is smaller than the machine's, for the
  budget, `array()`, `status()` and the new `limits()`.
- `pressure()` (ok / moderate / high / critical, from free memory and Linux PSI); under high pressure big arrays go
  to the SSD and stores and pools give RAM back by themselves; `relieve()`; `on_pressure(fn)` watcher.
- `pool()`: reusable buffers for big temporaries (`take` / `give`), 1.36× faster for 256 MB temporaries.
- `chunks()` and `map()` stream memory-mapped files with flat memory (3 GB file: peak RSS 3,103 → 97 MB, same speed),
  with kernel read-ahead; `chunks()` of arrays in RAM returns views.
- Stores keep a running SSD total: `set` no longer lists the folder (8.6 → 0.07 ms per `set` at 8,000 entries).
- `report()`: every array from `array()`, its size and whether it is in RAM or on the SSD.

### Speed on every processor
- The jammed lanes of an outer loop (n-body forces, matrix-vector products, distances) are emitted as SIMD vector
  instructions by Apsi itself instead of relying on LLVM's SLP vectoriser, which with LLVM 22 failed on AVX2
  processors: n-body 127.5 → 46.0 ms on an EPYC 7763 (gcc -O3: 44.2 ms); results bit-for-bit unchanged (`tests/test_jam.py`).
- `@native("fma")`: checks on, `a * b + c` may fuse into one instruction (gcc's default for C); ODE benchmark
  440 → 288 ms (gcc -O3: 387 ms).
- `@gpu` CPU/GPU choice: the first GPU launch (kernel loading) is no longer counted, and the losing side is timed
  again every 64 calls (`tests/test_gpu_choice.py`).
- `benchmarks/commits/measure.py` and the `commits` workflow measure several commits on one machine, in rotating
  rounds; the native cache now also depends on `APSI_SLP`. From the start of this work to the release, on one
  EPYC 7763: n-body 4.0×, ODE 1.5×, k-means 1.27×, compiling 1.21× faster, nothing slower
  (`benchmarks/commits/results-2.4.4.md`).

