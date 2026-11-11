# RAM, SSD, CPU and GPU (`use memory`)

Big data and big models run out of memory in the middle of the night. `use memory` decides where data lives so that
does not happen: in RAM while it fits, on the SSD (memory-mapped, the same code) when it does not, and on the GPU
with the most free memory when there is one.

```apsi
use memory
print(memory.status())
# RAM 7.3/7.8 GB free (this program 0.04 GB, budget 5.1 GB) | swap 0.0 GB | SSD 29 GB free at ~/.apsi/spill |
# 2 cores, load 0.34 | no GPU
```

## The memory you really have (containers)

```apsi
print(memory.limits())     # 5.8 GB usable (container (cgroup v1)), 5.8 GB free, budget 4.0 GB
```

In Docker, Kubernetes or a systemd slice, `/proc/meminfo` shows the whole host, but the kernel kills the process
at the container's limit. Apsi reads the limit and usage of the process's memory cgroup (v1 and v2, the smallest
limit up the hierarchy) and uses whichever is smaller everywhere: the default budget, `array()`, `status()`,
`pressure()`. A program that thought it had a 64 GB host when its pod allows 4 GB no longer plans with 64.

## Memory pressure

```apsi
print(memory.pressure())   # ok: 5.8 GB free (99 %), stalls 0.0 % (some) / 0.0 % (full)
let w = memory.on_pressure(fn(p) => print("memory is", p.level), level="high")
memory.relieve()           # give cached RAM back now
w.stop()
```

The level is `ok`, `moderate`, `high` or `critical`, from the free fraction of the usable memory and, on Linux,
the kernel's pressure stall information (`/proc/pressure/memory`, or the cgroup's own `memory.pressure`): the
share of time tasks waited for memory in the last 10 seconds. Apsi reacts by itself — no thread, the level is
re-measured at most twice a second on allocation paths:

| Level | What Apsi does |
|---|---|
| high | new big `memory.array()`s go to the SSD; stores halve their RAM tier; pools drop their spare buffers |
| critical | as high, and stores empty their RAM tier (values stay on the SSD), then a garbage collection |
| back to ok | stores grow their RAM tier back to the size you asked for |

`on_pressure(fn, level, every=1, repeat=10)` adds a background watcher that does the same and calls `fn` when the
level is reached (again every `repeat` seconds while it lasts) — for example to shrink a batch or pause a producer.

## Reusing big temporaries

```apsi
let p = memory.pool()
for step in 0..1000:
    let a = p.take([4096, 8192])        # 256 MB, already mapped after the first step
    work(a)
    p.give(a)                           # do not use a after this
```

Fresh memory costs page faults and zeroing every time it is touched; a pool hands back memory that is already
mapped. On a 2-core VM, twenty 256 MB temporaries took 768 ms from a pool and 1,041 ms freshly allocated (1.36×).
`take(shape, dtype, zero=true)` clears it; spare buffers are bounded (`max_gb`, default a tenth of the budget) and
dropped first under pressure. An array that is never given back is simply freed when it is no longer used.

## What lives where

```apsi
for r in memory.report():
    print(r)                            # [200000000, 8] float32 5.960 GB SSD /home/me/.apsi/spill/tmpab12.npy
```

## Arrays that never run out of RAM

```apsi
memory.budget(16)                                   # GB of RAM for big arrays (default: 70 % of what is free)
let x = memory.array([200_000_000, 8], "float32")   # 6.4 GB: in RAM if it fits the budget, else on the SSD
x[0, 0] = 1.5                                        # the same NumPy array either way
print(memory.on_disk(x))
memory.free(x)                                       # RAM back now / the spill file deleted now
```

Spill files live in `~/.apsi/spill` (or `APSI_SPILL_DIR`) and are deleted when the array is freed or the program
ends. `APSI_RAM_BUDGET=12GB` sets the budget from outside.

## Out-of-core processing

```apsi
var total = 0.0
for block in memory.chunks("features.npy", rows=1_000_000):   # the next block is read while this one is used
    total = total + block.sum()

let y = memory.map(fn(b) => b * 2.0 + 1.0, x, rows=500_000)   # all cores; output in RAM or on the SSD
```

`chunks` hands out views (no copy). For a file it asks the kernel to read the next block ahead while the current one
is used, and releases each finished block from the process, so streaming a file much larger than RAM keeps memory
flat: a 3 GB file streamed in 1,000,000-row blocks peaked at 97 MB of RSS (before: 3,103 MB) at the same speed.
`map` runs blocks on all cores (NumPy releases the GIL), releases finished input and output blocks the same way, and
allocates its output with `memory.array`, so the output can be larger than RAM too.

## A cache with two tiers

```apsi
let cache = memory.store("embeddings", ram_mb=512, disk_gb=50)
cache.set(doc_id, vector)              # RAM (least recently used leave first) and the SSD
let v = cache.get(doc_id)              # RAM hit, or read back from the SSD (arrays come back memory-mapped)
print(cache)                           # <store …: 12000 in RAM (511.8 MB), 480000 on SSD (40960.0 MB)>

@memory.cached                          # results survive restarts
fn tokenize(path):
    pass
```

* Arrays are stored as `.npy` files and come back memory-mapped (no copy until you touch them).
* Other values are pickled and **signed** with this user's key; a file changed outside Apsi is never unpickled.
* The SSD tier is bounded (`disk_gb`); the least recently used files are removed first, down to 90 % of the
  limit. Its size is kept as a running total, so `set` costs the same with 100 or 100,000 entries (before, every
  `set` listed the whole folder: 8.6 ms per `set` at 8,000 entries, now 0.07 ms).
* A store name is a folder under `~/.apsi/store`. A store at a path you choose (`memory.store("/data/cache")`) needs
  `--allow-write` for that path.

## Batches that fit

```apsi
let loss, batch = memory.fit_batch(fn(b) => train_step(b), 512)
```

`fit_batch` calls the function and, on out-of-memory (CPU `MemoryError`, CUDA out-of-memory from PyTorch or CuPy),
empties the GPU cache, halves the batch and tries again. It returns the result and the batch size that worked, so
the next call can start there.

## Devices

```apsi
print(memory.devices())          # cpu (cores, free RAM) and every GPU with its free memory
let dev = memory.best_device(8)  # "cuda:1" if a GPU has 8 GB free, otherwise "cpu"
```

GPU memory is read through PyTorch or CuPy when either is installed. Everything else works without them.
