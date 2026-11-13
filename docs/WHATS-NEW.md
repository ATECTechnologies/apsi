# What's new in Apsi 2.6.0

2.6.0 is about **training you can trust**: the same weights on one card or eight, the same weights every run,
training that survives a crash and resumes to the exact bit, and the GLA 3.0 code-and-language model as an ordinary
Apsi model. It also brings a way to send language models far fewer tokens without losing anything for good, and a
tokenizer you train yourself. Everything from 2.4.4 keeps working unchanged; the new protections are on by default.
The full list of changes is in the [changelog](https://github.com/ATECTechnologies/apsi/blob/main/CHANGELOG.md);
2.4.4's page is [here](WHATS-NEW-2.4.4.md).

| Part | In one line | Guide |
|---|---|---|
| Exact on any number of cards | one card, copies of the model and a model split across cards give the same weights, bit for bit | [below](#1-exact-training-on-any-number-of-cards) |
| The same weights every run | no atomic additions anywhere in training | [below](#2-the-same-weights-every-run) |
| Training that finishes | `steps=`, checkpoints that resume exactly, a guard against NaN steps, a save on any stop | [Training](TRAINING.md) |
| GLA 3.0 | the code-and-language model of GLA.py as a model of `use layers`, with the exact parameter count | [GLA](GLA.md) |
| Fewer tokens, found by GLA | `tokens.focus`: the parts a question needs, markers for the rest, everything restorable | [Tokens](TOKENS.md#when-it-still-does-not-fit-focus-built-on-gla) |
| Your own tokenizer | `data.train_tokenizer`: byte-level BPE, the tokenizer.json format | [Training](TRAINING.md#7-your-own-tokenizer) |
| Tensors on the card | `gpu_tensor(x)`: any code on the card, the rest falls back by itself | [GPU](GPU.md) |
| Any optimizer, any loss, any layer | on the GPU trainer, frozen parts, fine-tuning | [below](#7-training-modes) |
| Physics-informed networks | `pinn_fit`: equations as the loss, derivatives of any order | [PINN](PINN.md) |
| One model across many cards | a pipeline that splits any model, copies it, and stays exact | [GPU](GPU.md) |
| The card never waits | queued copies, losses and gates on the card, one launch per reduction | [below](#10-the-card-never-waits) |
| Settings in one file | `config/apsi-training.env`: every setting, its default, what it does | [Training](TRAINING.md#8-the-settings-file) |
| Faster everyday code | checks the compiler can prove are gone: calls, loops, f-strings, structs | [below](#13-faster-everyday-code) |
| Convert one file or a whole project | `apsi convert --only`, `--project`, `--exclude`, `--plan` | [below](#14-converting-python-one-file-or-the-whole-project) |

## 1. Exact training on any number of cards

Adding cards should make training faster, not different. In 2.6.0 it does exactly that:

```apsi
let r1 = fit_batches(model, params, batches, loss=my_loss, accumulate=4, backend="gpu")                    # one card
let r8 = fit_batches(model, params, batches, loss=my_loss, accumulate=4, device="cuda:all", stages=4)      # 4 stages x 2 copies
# r1.params and r8.params are identical, bit for bit
```

**How.** The gradients of a step's batches, of the copies of a stage and of the values crossing a cut between cards
are summed in *double-float*: each sum keeps its rounding error in a second number (an error-free TwoSum on the card,
kernels `acc2` / `fin2`) and is rounded once at the end. An exact sum crosses a cut whole. Optimizers that step on the
processor keep a float64 master copy, as NumPy does.

**Why it matters.** Without it, the order in which partial sums arrive — which depends on how many cards you have —
changes the last bits of every gradient, and over thousands of steps the runs drift apart. A result you cannot
reproduce on another machine is hard to trust and impossible to debug.

**Cost and control.** 8 bytes per parameter, on by default below 2 B parameters (`exact_sums=`,
`APSI_GPU_EXACT_SUMS`). Off, training still works; one card and several cards then round differently.

## 2. The same weights every run

Train twice, get the same weights — on the real card, not only in theory. Every reduction in training is now added in
a fixed order: bias and norm gradients (blocks of 32 columns × 8 lanes of rows, the last block to finish adds the
chunks in order), embedding gradients (a sorted index, one thread per row), losses, sums of squares, split matrix
products (the last part of each tile adds the parts in order), and the fused attention's backward (dQ in its own
kernel, summed over the key tiles in registers). `apsi gpu bench check` trains GLA and a transformer twice on your
card and compares the weights; on an RTX 3060 they are identical.

`APSI_GPU_DETERMINISTIC=0` brings back the atomic dQ in attention (one kernel instead of two).

## 3. Training that finishes

```apsi
let r = fit_batches(lm, params, batches, loss="cross_entropy", opt=adamw(lr=3e-4),
                    steps=20000, checkpoint="run.npz", save_every=500)
```

- **`steps=N`** trains N optimizer steps over as many epochs as it takes.
- **`checkpoint=`** saves the weights, the optimizer's state, the carried rounding error and the position in the
  data, atomically, every `save_every` steps, at the end, and **when training stops for any reason** (an error on the
  card, in your loss, Ctrl+C). Running the same call again resumes, and the result is bit for bit the result of one
  uninterrupted run. A checkpoint of another model is refused.
- **A guard against bad steps:** a step whose gradient holds inf or NaN is skipped inside the optimizer kernel; the
  weights and moments stay as they were. One corrupt batch can no longer turn a whole model into NaN.

On the GPU trainer and on NumPy. The complete guide, with what each protection costs: [TRAINING.md](TRAINING.md).

## 4. GLA 3.0 in Apsi

`gla(size)` builds the whole model of GLA.py — node encoder, Brain A (code), Brain B (editing), Brain C (text), the
anti-forgetting bridge, the FlyPrompt expert router, the graph memory, and the writer with gated cross-attention and
tied embeddings — with exactly the parameters of GLA's own formula (`gla_count("21b")`: 20,480,386,836). It trains on
NumPy, on one card and split across cards, the same training step for step. Real Python becomes GLA input with
`gla_from_code` (GLA's code property graph: syntax, control flow, data flow, calls). The writer's cross-entropy stays
on the card (`gla_loss.on_card`): the largest output never travels to the processor. Guide: [GLA.md](GLA.md),
example: `examples/45_gla.apsi`.

## 5. Fewer tokens, found by GLA

`use tokens` already removed redundancy exactly. 2.6.0 adds **focus**, for contexts that are simply too long:

```apsi
use tokens
let f = tokens.focus(read_text("lexer.py"), "how are string escapes handled?", budget=1500)
print(f.saved)              # 3,886 → 1,493 tokens: 61.6 % saved
send(f.text)                # the relevant code whole, numbered markers for the rest
let all = tokens.unfold(f)  # the original, exactly
```

It cuts the context along its structure (Python along its syntax tree; long functions into their parts; text into
blocks and sentences), links the parts with GLA's code property graph (calls, data flow) or shared rare words, scores
them by the question and spreads the scores over the graph, then fills the budget. What it leaves out is listed by
number, so a model can ask for it and you send `unfold(f, [n])`. `fit_chat(messages, budget)` does the same for
conversations; `outline(source)` shows a file's structure. Optional: nothing changes until you call it. Works with any
model. Guide: [TOKENS.md](TOKENS.md#when-it-still-does-not-fit-focus-built-on-gla), example:
`examples/46_focus_and_tokenizer.apsi`.

## 6. Your own tokenizer

```apsi
use data
let tok = data.train_tokenizer("corpus/", vocab_size=32000)
tok.save("tokenizer.json")
let ds = data.text_dataset("corpus/", tokenizer=tok, seq_len=1024)
```

A byte-level BPE trainer (the kind GPT-2, Llama 3 and Qwen use): distinct pre-tokens counted once, a merge touches
only the words with that pair, the best pair from a heap. 3.6 MB of text to 8,000 tokens in 1.8 s in Apsi; with
Hugging Face `tokenizers` installed it runs in its Rust trainer, faster again, same file. The file loads in Apsi and in
Hugging Face. The same corpus always gives the same tokenizer.

## 7. Training modes

- **Any optimizer** on the GPU trainer: adam, adamw and sgd keep their kernels; rmsprop, adagrad or your own
  (`step(params, grads)`) steps on the processor with float64 master weights.
- **Frozen parts:** `freeze=["enc"]`, `train_only="w*"` on NumPy, one card and many — frozen weights keep their exact
  values (decay and momentum cannot move them).
- **Any layer, any loss:** a layer or a loss without a kernel runs on the processor inside a model trained on the card.
- **`gpu_tensor(x)`:** products, arithmetic, activations, softmax, norms and reductions on the card with Apsi's own
  kernels; anything else sees an ordinary tensor. Accurate exp and log, IEEE division and square roots.

## 8. Physics-informed networks

`pinn_fit(model, params, points, loss)` where the loss is any formula of the points, the network's values and its
first and second derivatives: equations, boundary and initial conditions, measurements. On the GPU in Taylor mode
(values and derivatives through the network in one pass); on the processor with autodiff of any order.
[PINN.md](PINN.md), `examples/44_physics_networks.apsi`.

## 9. One model across many cards

`fit(…, backend="gpu", device="cuda:all")` splits a model that does not fit one card into stages where the work
balances; micro-batches flow through them in a one-forward-one-backward order with every card working at the same
time; more cards than stages copy the pipeline. Transformers with grouped-query attention, rotary positions, q/k norms,
SwiGLU and mixture-of-experts blocks; graph networks; convolutions; recurrent layers. Exact, as section 1 says.

## 10. The card never waits

- Copies to the card go through pinned memory, queued behind the running kernels (`APSI_GPU_ASYNC_COPY`).
- `fit_batches` keeps each batch's loss on the card until the end of the epoch.
- Gates (`tanh` of a learned scalar) are computed, applied and differentiated on the card.
- Reductions finish in one launch; split products add their parts inside the product.
- The cross-entropy uses accurate exp and log (no approximate instructions).

On an RTX 3060, a small transformer's step went from 5.3–6.0 ms to 4.0–4.7 ms (best runs; single runs vary by about
±15 %), with identical weights every run.

## 11. Settings and tuning

Every setting is an environment variable with a safe default, collected in
[`config/apsi-training.env`](../config/apsi-training.env) with a line about each: run
`apsi run train.apsi --env-file apsi-training.env`. Temperature control (77 °C by default) and the power limit
(`APSI_GPU_POWER`, set once with administrator rights by `apsi gpu setup` or the Apsi GPU Tuner) keep long runs at a
steady speed. [TRAINING.md](TRAINING.md#9-what-happens-if-you-turn-things-off) lists what each protection costs and
what turning it off would change; turning things off never changes what a program means, only how exactly or how
safely it runs.

## 12. How it was tested

- 409 tests on every push, on Linux (x64, ARM64), macOS (Apple silicon, Intel) and Windows, Python 3.12–3.14.
- On a real NVIDIA RTX 3060: every kernel against the exact reference, GLA and a transformer trained twice with
  identical weights, steps and resume bit for bit, the NaN guard, split products, the fused attention both ways — 60
  tests (`tools\gpu_check.bat`).
- NumPy, one card and many cards compared on the same training, step for step.
- CPU programs against C, C++, Rust, Go, Java, JavaScript, C# and Julia, and 2.6.0 against 2.4.4, on a GitHub
  runner and a 2-core VM: the eight heavy programs run at the same speed as 2.4.4 (0.99×; 0.83× the time of C), and
  everyday code is faster (section 13: fib 1.5×, a word count 2.1×, f-strings 2.3×, structs 1.7×; geometric mean 1.22×).
- Every benchmark category of the website (Python, matrix products, tables against pandas, the web server, real-time
  video, models that learn, WebSocket) run again for 2.6.0 and 2.4.4 on the same machine.

**Not measured again for 2.6.0** (the code is unchanged since 2.4.4, and the website keeps 2.4.4's numbers under
"Measured with Apsi 2.4.4"): language models from Hugging Face on the card (`use hub`), the AI server, the GPU
training speed against PyTorch, tables, databases, real-time video. `tools\bench_my_pc.bat` measures the GPU training
speed and every CPU benchmark on your own computer.

## 13. Faster everyday code

Apsi checks types and units while a program runs. 2.6.0 drops the checks the compiler can prove will always pass,
so ordinary code — functions, loops, strings, structs — runs faster with the same results and the same errors:

- A typed `Int` or `Float` parameter that the function never rebinds is checked once, on the way in; after that,
  comparing it is a plain condition and returning it needs no second check.
- `for i in 0..n` with whole-number ends is Python's own `range`, and `i` is a known integer: indexing with it and
  printing it in an f-string go straight through.
- A name bound once to a list, dict, set or text literal calls its methods (`parts.append`, `counts.get`), is
  iterated and is stored into directly.
- A struct with untyped, `Int` or `Float` fields gets a constructor of its own.
- On the card, products whose sizes are not multiples of 4 (a SwiGLU width of 2730) use the fast tiled tensor-core
  kernels through zero-padded copies.

Measured against 2.4.4 on the same machine (a GitHub runner): recursive calls (fib 25) 1.5× faster, a word count
2.1×, 200k f-strings 2.3×, 200k structs 1.7×; the geometric mean of the 12 version measurements is 1.22×. The
eight heavy `@native` programs were already at C's speed (0.83× the time of C) and are unchanged. The website's benchmark page has every measurement.

## 14. Converting Python: one file or the whole project

```bash
apsi convert app.py                    # app.py and every module of the program it imports
apsi convert app.py --only             # app.py alone; the program's other modules stay Python
apsi convert app.py --project          # every .py of the project (pyproject.toml / setup.py / .git above it)
apsi convert app.py --project --exclude "tests/*" --plan    # see the list first, write nothing
```

Files that import each other are converted together, so the converted modules import each other as Apsi files. In
VS Code, *Convert Python to Apsi* asks which of these you want. The complete guide: [Python](PYTHON.md).

## Every new name in one table

| Name | Where | What it does | Default |
|---|---|---|---|
| `steps=N` | `fit_batches` (NumPy, one card) | trains N optimizer steps over as many epochs as needed | `none` (use `epochs`) |
| `checkpoint="run.npz"` | `fit_batches` | weights, optimizer state, rounding error and data position, saved atomically; resumes on the next call | off |
| `save_every=K` | `fit_batches` | a checkpoint every K steps (and always at the end and on any stop) | end and stop only |
| `resume=false` | `fit_batches` | start from the beginning even if the checkpoint exists | `true` |
| `callback=fn(r)` | `fit_batches` | after every epoch: `r.epoch`, `r.step`, `r.loss` | none |
| `freeze=` / `train_only=` | `fit`, `fit_batches` | weights that stay exactly as they are, by path or function | none |
| `exact_sums=` | GPU `fit`, `fit_batches` | double-float sums across batches, copies and cards | on below 2 B parameters |
| `stages=` | GPU `fit`, `fit_batches` with `device="cuda:all"` | how many cards one copy of the model is split across | balanced automatically |
| `gpu_tensor(x)`, `gpu_device()` | prelude | a tensor on the card; where it lives | — |
| `pinn_fit`, `pinn_derivatives` | `use layers` | physics-informed training; u, ∂u, ∂²u at points | — |
| `gla(size)`, `gla_count`, `gla_from_code`, `gla_loss` | `use layers` | GLA 3.0 as an Apsi model | — |
| `tokens.focus`, `unfold`, `fit_chat`, `outline` | `use tokens` | fewer tokens, everything restorable | — |
| `data.train_tokenizer` | `use data` | byte-level BPE, `tokenizer.json` | `vocab_size=8000` |
| `apsi convert --only` / `--project` / `--exclude` / `--plan` | command line | how much Python is converted | the file and its imports |
| `APSI_GEMM_PAD` | environment | products with sizes not multiples of 4 on the tiled kernels | `1` |
| `APSI_GPU_GUARD` | environment | skip steps with inf/NaN gradients | `1` |
| `APSI_GPU_DETERMINISTIC` | environment | fixed-order reductions everywhere, including attention's backward | `1` |
| `APSI_GPU_EXACT_SUMS` | environment | force double-float sums on (`1`) or off (`0`) | automatic |
| `APSI_GPU_ASYNC_COPY` | environment | queued copies through pinned memory | `1` |

## Recipes

**A long run that must not be lost.** Steps, a checkpoint every few hundred steps, and the guard (on by default):

```apsi
use layers
let r = fit_batches(lm, lm.init(seed=0), batches, loss="cross_entropy", opt=adamw(lr=3e-4),
                    steps=50000, checkpoint="lm.npz", save_every=500, backend="gpu",
                    callback=fn(r) => print(f"epoch {r.epoch} step {r.step} loss {r.loss:.4f}"))
```

Stop it at any moment (Ctrl+C, a power cut, an error); run the same line again and it continues from the last
save. A save takes about as long as writing the weights twice to disk; 500 steps is a good interval for models up
to a few hundred million parameters.

**Fine-tune only the head.**

```apsi
let r = fit(model, params, X, Y, loss="cross_entropy", opt=adamw(lr=1e-4), train_only="head*", backend="gpu")
# every weight outside head* is bit for bit what it was before
```

**Run NumPy-style code on the card without rewriting it.**

```apsi
let a = gpu_tensor(x)                 # x: an ordinary tensor
let h = relu(a @ w1 + b1)             # on the card
let p = softmax(h @ w2, axis=-1)      # on the card
print(p.numpy()[:3])                  # back to the processor
```

Anything Apsi has no kernel for is computed on the processor automatically and comes back as an ordinary tensor.

**A model larger than one card.**

```apsi
let r = fit_batches(model, params, batches, loss="cross_entropy", opt=adamw(lr=2e-4),
                    device="cuda:all", stages=2, accumulate=8)
# 4 cards: 2 stages x 2 copies; the weights equal those of one card with accumulate=8
```

**Send a model only what the question needs.**

```apsi
use tokens
let f = tokens.focus(read_text("server.py"), "where is the session timeout set?", budget=2000,
                     keep=["def create_app"])
send(f.text)                  # if the model asks for marker [4]:
send(tokens.unfold(f, [4]))
```

**Train a tokenizer and a dataset from the same folder.**

```apsi
use data
let tok = data.train_tokenizer("corpus/*.txt", vocab_size=16000, special_tokens=["<|endoftext|>", "<|pad|>"])
tok.save("tok.json")          # also loads in Hugging Face tokenizers
let ds = data.text_dataset("corpus/*.txt", tokenizer=tok, seq_len=512)
```

## Questions

**Is 2.6.0 slower because of exact sums and determinism?** No. Everyday code is faster than in 2.4.4 (section 13),
the heavy native programs run at the same speed, and on the card a small transformer's step went from 5.3–6.0 ms to
4.0–4.7 ms on an RTX 3060, even though the deterministic attention backward runs two kernels instead of one. Exact
sums cost 8 bytes per parameter of card memory.

**My results differ from 2.4.4 in the last digits.** Expected: 2.6.0 sums more exactly. The difference is in the
direction of the exact value; it is not a change in what the program computes.

**Can I resume a checkpoint on another machine?** Yes, with the same model and data: the file holds the weights,
the optimizer's moments in float32 and float64 and the position in the data. A different card gives the same
result because training is deterministic.

**What does the guard do when every step is bad?** It skips them all and the loss does not move; the history shows
the same value repeated. That almost always means a bug in the data or the loss (a log of 0, a division by 0).
`APSI_GPU_GUARD=0` lets the NaN through so you can see where it appears.

**Does focus need GLA weights?** No. It uses GLA's code property graph (the structure of the code), not a trained
GLA model. It runs on the processor in milliseconds.

**Does training on several cards need special code?** No: `device="cuda:all"`. With `steps=` or `checkpoint=` it
currently needs one card; the multi-card trainer raises a clear error instead of silently ignoring them.

## Limits in 2.6.0

- `steps=`, `checkpoint=` and `save_every=` work on NumPy and on one card; on several cards they are refused with a
  message (planned).
- Mixed second derivatives (∂²u/∂x∂y) are not in `pinn_fit`'s `d2u`; write that loss with `grad` on the processor.
- `focus` reads Python by its syntax tree; other languages are read as text (blocks and lines), which still works but
  links fewer parts.
- Exact sums turn themselves off at 2 B parameters (memory); `exact_sums=true` forces them on.
- AMD and Intel cards are not supported by the GPU trainer (NVIDIA, CUDA driver).

## Upgrading from 2.4.4

Nothing to change. Programs run as before; the new protections (exact sums, determinism, the NaN guard) are on by
default and change results only in the last bits, towards the exact values. A text list given to `text_dataset`
that contains `*` or `[` is now read as text (before, it could be taken for a file pattern).
