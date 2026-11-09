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

`chunks` overlaps SSD reads with computation (one read-ahead thread). `map` runs blocks on all cores (NumPy releases
the GIL) and allocates its output with `memory.array`, so the output can be larger than RAM too.

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
* The SSD tier is bounded (`disk_gb`); the least recently used files are removed first.
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
