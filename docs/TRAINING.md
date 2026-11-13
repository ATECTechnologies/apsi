# Training that finishes: steps, checkpoints, exact results, no surprises

This page explains how Apsi 2.6.0 keeps a training run **correct, repeatable and alive**: training by steps,
checkpoints that resume to the exact bit, protection against crashes and bad batches, the same weights on any
number of cards, and the settings that control all of it. Every setting has a safe default; this page tells you
what each one does, why it matters, and what you give up if you turn it off.

- [1. Quick start](#1-quick-start)
- [2. Training by steps](#2-training-by-steps)
- [3. Checkpoints and exact resume](#3-checkpoints-and-exact-resume)
- [4. Crash protection](#4-crash-protection)
- [5. Exact and repeatable results](#5-exact-and-repeatable-results)
- [6. Speed: what keeps the card busy](#6-speed-what-keeps-the-card-busy)
- [7. Your own tokenizer](#7-your-own-tokenizer)
- [8. The settings file](#8-the-settings-file)
- [9. What happens if you turn things off](#9-what-happens-if-you-turn-things-off)
- [10. Checking your card](#10-checking-your-card)

---

## 1. Quick start

```apsi
use layers
use data

let tok = data.train_tokenizer("corpus/*.txt", vocab_size=16000)        # your own tokenizer, in seconds
tok.save("tokenizer.json")
let ds = data.text_dataset("corpus/*.txt", tokenizer=tok, seq_len=512)
let batches = [b for b in ds |> batch(16)]

let lm = transformer_lm(tok.vocab_size, d=512, heads=8, layers=8, context=512)
let r = fit_batches(lm, lm.init(seed=0), batches, loss="cross_entropy", opt=adamw(lr=3e-4),
                    steps=20000,                      # stop after 20 000 optimizer steps
                    checkpoint="run.npz",             # saved every 500 steps, at the end and on any stop
                    save_every=500,
                    clip=1.0)
```

Run it. Stop it at any time (Ctrl+C, a power cut, a driver reset). Run **the same line again** and it continues from
the last checkpoint with the same weights, the same optimizer state and the same position in the data: the result
is bit for bit the result of one uninterrupted run.

## 2. Training by steps

`steps=N` trains for N optimizer steps instead of a number of epochs. It is the natural unit for large data sets
(an epoch may be hours), for learning-rate schedules (`schedule=fn(s) => cosine_lr(s, N, …)`), and for comparing
runs. The data is taken in order, epoch after epoch, as long as it takes.

| Argument | Meaning |
|---|---|
| `steps=N` | train N optimizer steps (`epochs` is then ignored) |
| `accumulate=A` | one step = A batches, their gradients averaged (a larger batch in the same memory) |
| `callback=fn(r)` | after every epoch: `r.epoch`, `r.step`, `r.loss` |
| `every=` | how often a line is printed |

With `accumulate=A` the A batches of a step are summed **exactly** (see section 5), so a step of 4 batches of 8 is
the same as one batch of 32 — on one card or many.

## 3. Checkpoints and exact resume

`checkpoint="file.npz"` turns on saving. A checkpoint holds:

- the weights (and, for optimizers that step on the processor, the float64 master copy);
- the optimizer's state: Adam's two moments, SGD's velocity, the step counter (for Adam's bias correction);
- the carried rounding error of compensated updates (section 5);
- the position in the data: the epoch and the batch the next step starts with.

It is written with a temporary name and then renamed, so a crash while saving never leaves a broken file. When
`resume=true` (the default) and the file exists, training continues from it. A checkpoint of a different model is
refused with a clear message instead of loading the wrong weights.

| Argument | Default | Meaning |
|---|---|---|
| `checkpoint=` | none | the file |
| `save_every=K` | none | also save every K steps (always saved at the end and on any stop) |
| `resume=` | `true` | continue from the file when it exists; `false` starts over (and overwrites it) |

**Why it matters:** long runs fail for reasons that have nothing to do with your model (a driver update, a sleeping
laptop, a full disk somewhere else). Without a checkpoint everything since the start is lost. With one, at most
`save_every` steps are repeated, and the result is the same as if nothing had happened.

Works with `fit_batches` on the GPU trainer (one card) and on NumPy. On several cards use `epochs=` for now.

## 4. Crash protection

Three layers of protection are on by default.

**A bad batch cannot ruin the weights.** If a step's gradient contains inf or NaN (an overflow, a division by zero in
your loss, a corrupt sample), the step is skipped: the weights, the optimizer's moments and the carried rounding error
stay exactly as they were, and training continues with the next batch. On the card the check is part of the
optimizer kernel itself (one sum of squares of the gradient per step, no extra copy). `APSI_GPU_GUARD=0` turns it off.

**A stop keeps what was learned.** When training stops for any reason — an error on the card, an error in your own
loss, Ctrl+C — and `checkpoint=` is set, the last complete step is saved before the error is shown. Run the same
call again to continue.

**Errors say what to do.** Running out of memory names the memory in use, what was asked for, and the ways out
(a smaller batch, fewer layers, a shorter context, `accumulate=`, more cards). A checkpoint of another model, a
missing card, an unsupported option: each has a message that says what happened and what to change.

## 5. Exact and repeatable results

Rounding is not noise you have to accept. Apsi removes it where it changes results:

| What | How | Setting |
|---|---|---|
| **The same weights every run** | no atomic additions anywhere in training: every sum is added in a fixed order (column sums, embedding gradients, losses, sums of squares, split products, attention's backward) | `APSI_GPU_DETERMINISTIC` (default on) |
| **The same weights on 1 or 8 cards** | the gradients of a step's batches, of copies of the model and across cuts are summed in double-float (an error-free TwoSum, rounded once at the end) | `exact_sums=` / `APSI_GPU_EXACT_SUMS` (default on below 2 B parameters) |
| **Small updates are not lost** | each weight carries the rounding error of its last update into the next one (compensated summation): 300 Adam steps of 1e-9 move a weight of 1 by 3e-7, a plain float32 update not at all | `compensated=` / `APSI_GPU_COMPENSATED` |
| **Accurate math on the card** | the cross-entropy's exp and log, tanh gates and the element-wise functions use accurate algorithms (Cody-Waite reduction, a Newton step, float64 where it is one number) instead of the fast approximate instructions | always |
| **Optimizers on the processor** | keep a float64 master copy of the weights, as NumPy does | always |

What this buys you: a bug you see once you can see again; a change in your model shows up as a change in the result
and not as run-to-run noise; adding cards makes training faster without making it different; and very long runs with
small learning rates still learn.

The cost is small: a few percent of a step (measured on an RTX 3060, the deterministic reductions are now one launch
each — the parts are added by the last block to finish, in order — so they cost about the same as the atomic ones).
Each setting can be turned off; section 9 says what changes.

## 6. Speed: what keeps the card busy

The card is fastest when it never waits for the processor. In 2.6.0:

- **Copies are queued, not waited for.** Inputs go through pinned memory and are queued behind the kernels already
  running (`APSI_GPU_ASYNC_COPY`, default on), so the processor prepares the next batch while the card works.
- **Losses stay on the card** until the end of the epoch (`fit_batches`); nothing reads a number back every step.
- **Gates stay on the card:** tanh gates are computed, applied and differentiated by kernels, not read back.
- **One launch per reduction:** sums finish in the last block of the same kernel; split-k products add their parts
  in the last part of each tile (no second kernel).
- **Large outputs stay on the card with your own loss:** a loss function can say which of its outputs is a plain
  cross-entropy (`loss.on_card`); that part runs on the card and only the small outputs go to the processor. GLA's
  loss does this: the writer's logits never leave the card.

Measured on an RTX 3060 (a 2-layer transformer, 128 wide, 8 × 128 tokens a step): 5.5–6.0 ms a step before these
changes, 4.0–4.5 ms after, with the same weights every run.

## 7. Your own tokenizer

`data.train_tokenizer(source, vocab_size=…)` trains a byte-level BPE tokenizer (the kind GPT-2, Llama 3 and Qwen
use) on files, globs, folders, records or strings — the same sources `text_dataset` reads.

```apsi
use data
let tok = data.train_tokenizer("corpus/", vocab_size=32000, verbose=true)
print(tok.vocab_size, tok.encode("Hello"))
tok.save("tokenizer.json")                          # loads in Apsi (hub.tokenizer_from_file) and in Hugging Face
```

It is fast because the text is split into pre-tokens once and only distinct pre-tokens with their counts are kept;
a merge touches only the words that contain the pair, and the best pair comes from a heap. With Hugging Face
`tokenizers` installed (`pip install tokenizers`), training runs in its Rust trainer, several times faster again,
with the same file format (`engine="apsi"` keeps it in Apsi). The two agree on more than 95 % of the merges (ties
are broken differently) and give the same number of tokens within 2 %. The same corpus always gives the same
tokenizer.

| Argument | Default | Meaning |
|---|---|---|
| `vocab_size` | 8000 | including the 256 bytes and the special tokens |
| `min_frequency` | 2 | pairs seen fewer times are not merged |
| `special_tokens` | `["<\|endoftext\|>"]` | added at the end of the vocabulary |
| `lower` | false | lower-case the text first |
| `format`, `field`, `template`, `text_split` | as in `text_dataset` | how records become text |
| `engine` | `"auto"` | `"rust"`, `"apsi"` |

## 8. The settings file

Every setting on this page is an environment variable with a safe default. `config/apsi-training.env` lists them
all with their defaults and a line about each; copy it next to your program, change what you need, and run
`apsi run train.apsi --env-file apsi-training.env`. Nothing in it is required: an empty file gives the defaults.

## 9. What happens if you turn things off

| Turned off | What you gain | What you lose |
|---|---|---|
| `APSI_GPU_GUARD=0` | one sum of squares per step (a fraction of a percent) | one NaN batch can turn every weight into NaN |
| `APSI_GPU_DETERMINISTIC=0` | the fused attention's backward in one kernel instead of two (a few percent of attention's time) | two runs of the same training can end with slightly different weights |
| `APSI_GPU_EXACT_SUMS=0` | 8 bytes per parameter of memory | accumulated batches and several cards round differently from one large batch on one card |
| `APSI_GPU_COMPENSATED=0` | 4 bytes per parameter of memory | updates far below a weight's float32 step are lost (long runs, small learning rates) |
| `APSI_GPU_ASYNC_COPY=0` | 64 MB of pinned memory | every copy to the card waits for it |
| no `checkpoint=` | no files written | a crash loses the whole run |

None of these changes what your program means: the same code runs, the results differ only as the table says.
Performance can drop somewhat with protections off in unusual cases (for example a NaN batch that ruins a run is
the most expensive outcome of all), and stability is always kept as high as possible.

## 10. Checking your card

`apsi gpu bench check` (Windows: double-click `tools\gpu_check.bat`, the report goes to `gpu-check.txt`) compares
every kernel on your card with the exact reference, trains GLA twice and checks the weights are identical, trains a
transformer on the fused attention twice (identical again) and measures a step. `tools\flow_bench.bat` measures
where a step's time goes (queued against waiting copies, the kernels on the card).
