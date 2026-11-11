# Model files

Apsi saves and loads models in the formats the rest of the ecosystem uses, and in its own checkpoint format
(`.apsim`) for everything those formats cannot hold: physical units, compressed tensors, the shape of the
parameter tree, integrity checks and signatures.

| Format | Save | Load | Inspect | Opens in |
|---|---|---|---|---|
| **Apsi checkpoint** `.apsim` | `save_model` | `load_model` | `model_info` | Apsi; any safetensors reader (PyTorch, JAX, Hugging Face) |
| **GGUF** `.gguf` | `save_gguf` | `load_gguf` | `gguf_info` | llama.cpp, Ollama, LM Studio, any GGUF reader |
| **safetensors** | `save_safetensors` | `load_safetensors` | `safetensors_info` | PyTorch, JAX, Hugging Face |
| ONNX / PyTorch | — | `onnx_model`, `torch_load` | — | see [`AI.md`](AI.md) |

All of them are in `use models`. Saving needs `--allow-write`, loading needs `--allow-read` (or the `fs`
permission), exactly like other file functions, and `apsi audit` reports their use.

## 1. Apsi checkpoints (`.apsim`)

```apsi
use models
use precision

let params = record(layers=[record(W=randn(4096, 4096), b=zeros(4096))],
                    mass=tensor([1.0, 2.0]) * 1kg)                 # units are kept
let small = quantize_model(params, int8)                            # compressed tensors are kept as they are

save_model("run/model.apsim", small, metadata=record(arch="mlp", step=12000))
let back = load_model("run/model.apsim")                            # memory-mapped: opens instantly
print(model_info("run/model.apsim"))
```

What the format guarantees:

- **Standard container.** Each `.apsim` file is a valid safetensors file. Any safetensors reader gets the tensors
  (compressed tensors appear as their codes plus a `NAME__qscale` tensor).
- **Exact round trip.** Records and lists come back as they were. Tensors keep their dtype and bits, `Quantity`
  values keep their unit, and QTensors (INT8, INT4, FP8, FP6, FP4, BF16) keep their codes, scales and format.
- **Integrity.** Each file stores a SHA-256 of its tensor table (names, dtypes, shapes, offsets) and its data.
  `load_model` checks it by default and refuses a damaged or modified file. Use `verify=false` to skip the check
  on trusted storage when load time matters.
- **Signatures.** `sign=true` signs with your per-user key (the same key that protects Apsi's caches), and
  `sign="shared secret"` signs with a key shared by a team. `load_model(path, key=true)` or
  `load_model(path, key="shared secret")` then refuses a file that is unsigned, was signed with another key, or
  was changed after signing. Signing uses HMAC-SHA256 over every shard digest and the Apsi metadata. Giving a key
  always turns verification on.
- **Large models.** A checkpoint larger than `shard` (default `"4GB"`) is split into
  `NAME-00001-of-0000N.apsim` files plus `NAME.apsim.index.json` in the Hugging Face layout. Load the index
  file. Each shard digest is recorded in the index, so mixing shards from different checkpoints is detected.
- **Safe parsing.** Loading never runs code from the file. Header size, every tensor offset and every
  size are checked against the file before anything is mapped. A malformed file gives a clear error, never a
  crash; this is covered by fuzz tests.
- **Atomic writes.** Files are written to a temporary name and renamed, so an interrupted save never leaves a
  half-written checkpoint under the real name.

### Layout

```
u64 little-endian   header length N
N bytes             JSON header (safetensors): tensor table + "__metadata__"
data                tensors, back to back, in table order
```

`__metadata__` keys written by Apsi (all values are strings, as safetensors requires):

| Key | Content |
|---|---|
| `format` | `"apsi-model"` |
| `version` | format version, currently `"1"` |
| `apsi` | Apsi version that wrote the file |
| `units` | JSON: tensor name → dimension vector |
| `qtensors` | JSON: tensor name → format, logical shape, axis, packing, scale shape, unit |
| `metadata` | JSON: your `metadata=` record |
| `sha256` | SHA-256 of the canonical tensor table followed by the data |
| `shard` | `"i/N"` in sharded checkpoints |

Tensor names are the parameter paths joined with dots (`layers.0.W`), so the tree is rebuilt from the names.

### Everything to resume training, in one file

```apsi
use models
save_checkpoint("run/step-12000", params, opt=opt, step=12000, config=record(d_model=512, layers=8),
                files={"tokenizer.json": "tok/tokenizer.json", "notes": "after warm-up"})
let ck = load_checkpoint("run/step-12000")       # record(params, optimizer, step, config, files, metadata)
var opt2 = ck.optimizer                          # Adam / AdamW / SGD with their moments and step count
```

`save_checkpoint` keeps the weights, the optimizer state, the step, a config record and any files (bytes, text or
a path) in one `.apsim` file; above `shard` (default 4 GB) it writes shards and an index instead. `load_checkpoint`
takes the file, the index, the name without extension or any one shard's name, and also reads files written by
`save_model` (then only `params` is set). Resuming continues exactly: the next optimizer step is bit for bit the one
the original would have taken.

## 2. GGUF

```apsi
use models
save_gguf("model.gguf", params, metadata=record(llama_context_length=4096), dtype="q8_0")
let m = load_gguf("model.gguf")          # record(params=…, metadata=…)
print(gguf_info("model.gguf"))
```

- GGUF version 3 with 32-byte alignment. Metadata of every GGUF type is supported (integers, floats, booleans,
  strings and arrays).
- `dtype` is `"f32"`, `"f16"` (default), `"bf16"`, `"q8_0"` or `"q4_0"`. Q8_0 and Q4_0 are the block formats of
  ggml and are **bit-identical** to the reference implementation (tested against the official `gguf` package).
  Tensors that cannot be block-quantized (1-D tensors, or rows that are not a multiple of 32) are written as F16
  or F32. Integer tensors keep their integer type.
- Loading memory-maps F32, F16, F64 and integer tensors and decodes BF16, Q8_0 and Q4_0 to f32.
- Units are stored as `apsi.units.<tensor>` keys, so Apsi gets them back and other readers ignore them.
- Name the keys the way your runtime expects (for example `general.architecture`, `llama.block_count`).
  `save_gguf(..., architecture="llama")` sets `general.architecture`.

## 3. Command line

```
apsi model info model.gguf                          # tensors, types, metadata (headers only)
apsi model info model.apsim.index.json
apsi model verify model.apsim                       # check SHA-256 of every shard
apsi model verify model.apsim --key "team secret"   # and the signature
apsi model convert model.safetensors model.gguf --dtype q4_0
apsi model convert model.gguf model.apsim --shard 2GB --key "team secret"
apsi model convert model.apsim model.safetensors
```

`convert` reads `.safetensors`, `.gguf` and `.apsim` (including sharded ones) and writes any of the three.
Compressed tensors are expanded to f32 when the target format cannot hold them.

## 4. Choosing a format

| Need | Use |
|---|---|
| Resume training, keep units, compressed weights and the tree; detect tampering | `.apsim` |
| Run the model in llama.cpp, Ollama or LM Studio | GGUF |
| Hand weights to PyTorch, JAX or the Hugging Face Hub | safetensors, or `.apsim` (it is safetensors) |
| Checkpoints of 70B–2T parameters | `.apsim` with sharding (and `apsi plan` for the hardware) |
