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
  everyday code is within run-to-run noise of 2.4.4 (geometric mean 0.97× on the runner, 1.05× and 0.99× on two VM
  runs). 2.6.0 changes training, not the language's speed.

**Not measured again for 2.6.0** (the code is unchanged since 2.4.4, and the website keeps 2.4.4's numbers under
"Measured with Apsi 2.4.4"): language models from Hugging Face on the card (`use hub`), the AI server, the GPU
training speed against PyTorch, tables, databases, real-time video. `tools\bench_my_pc.bat` measures the GPU training
speed and every CPU benchmark on your own computer.

## Upgrading from 2.4.4

Nothing to change. Programs run as before; the new protections (exact sums, determinism, the NaN guard) are on by
default and change results only in the last bits, towards the exact values. A text list given to `text_dataset`
that contains `*` or `[` is now read as text (before, it could be taken for a file pattern).
