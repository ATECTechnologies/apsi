## 2.6.0 — exact training on any number of cards, GLA 3.0, training that finishes, fewer tokens (13 November 2026)

**Install:** [apsi.dev/download](https://apsi.dev/download) — `ApsiSetup-2.6.0-x64.exe` (Windows 10/11),
`ApsiSetup-2.6.0-linux.sh` (any Linux, WSL), `apsi_2.6.0_all.deb` (Ubuntu / Debian), or from this source:
`pip install numpy`, then `./apsi.sh` / `apsi.bat`. What's new, with examples: [docs/WHATS-NEW.md](https://github.com/ATECTechnologies/apsi/blob/main/docs/WHATS-NEW.md);
everyday code runs 1.22× faster than in 2.4.4 (geometric mean); the training guide: [docs/TRAINING.md](https://github.com/ATECTechnologies/apsi/blob/main/docs/TRAINING.md).

Tested: 413 tests on every push (Linux x64 / ARM64, macOS Apple silicon / Intel, Windows; Python 3.12–3.14) and 81
on a real NVIDIA RTX 3060 (every kernel against the exact reference; GLA and a transformer trained twice with
identical weights; steps and resume bit for bit; the NaN guard).

### Faster everyday code
- **Facts the compiler can prove are no longer checked at run time:** a typed `Int`/`Float` parameter that is never
  rebound is a plain number for the whole call (a comparison of two such numbers is a plain condition; returning it
  needs no second check); `for i in 0..n` with whole-number ends runs on Python's own `range` and `i` is a known
  integer (indexes and f-strings use it directly); a name bound once to a list, dict, set or text literal calls its
  methods, iterates and stores directly. Same results, same errors.
- **Structs are built directly:** a struct whose fields are untyped, `Int` or `Float` gets a constructor of its own
  (no general plan per call): creating 200,000 structs is 1.8× faster.
- Measured against 2.4.4 on the same machine (GitHub runner): recursive calls (fib 25) 1.5×
  faster, a word count 2.1×, 200k f-strings 2.3×, 200k structs 1.7× (GitHub runner, geometric mean of the 12
  measurements 1.22×); the eight heavy `@native` programs are unchanged (the native code generator was already at
  C's speed).
- **Products whose sizes are not multiples of 4** (a SwiGLU width of 2730, an odd vocabulary) run on the tiled
  tensor-core kernels through zero-padded copies instead of the general kernel (`APSI_GEMM_PAD=0` turns it off).

### Converting Python: one file, its imports, or the whole project
- `apsi convert app.py` converts the file and every module of the program it imports (as before);
  **`--only`** converts just the files named and keeps the program's other modules as Python imports;
  **`--project`** converts every `.py` of the project the file belongs to (the folder with `pyproject.toml`,
  `setup.py`, `setup.cfg`, `requirements.txt`, `Pipfile` or `.git`); **`--exclude PATTERN`** leaves files out;
  **`--plan`** lists each file, where it goes and why, and writes nothing.
- VS Code: *Convert Python to Apsi* asks how much: this file and its imports, this file only, the whole project, or
  the plan first.
- `pyimport` also finds modules in the program's own folder (a module kept as Python next to a converted program).

### Exact training on many cards
- **The same weights on any number of cards, to the last bit:** the gradients of a step's batches, of the copies of
  a stage and of the values that cross a cut are summed exactly (double-float, two new kernels: `acc2`, an
  error-free TwoSum accumulation, and `fin2`, one rounding at the end), and an exact sum crosses a cut whole (its
  high and low parts). Copies of the model and cuts between nodes give one card's training bit for bit; a weight two
  stages share is summed more exactly than one card does. On by default below 2 B parameters
  (`fit_batches(…, exact_sums=)`, `APSI_GPU_EXACT_SUMS`).
- **Order-free gradients inside the model:** a value used by several layers gets their gradients' exact sum,
  rounded once, whatever order they arrive in.
- **Run-to-run determinism on the card:** no atomic additions anywhere in training. Column sums (bias and norm
  gradients) run as blocks of 32 columns x 8 lanes of rows; with several chunks of rows, the last chunk to finish adds
  them in chunk order (a counter per launch, reset by that block). Embedding gradients use a sorted index
  (`embed_bs`, one thread per row, ids in order). Losses, sums of squares and dot products write per-block parts that
  the last block adds in order. Split-k products write each slice to a workspace and the last part of each tile adds
  them in order. The fused attention's backward computes dQ in a kernel of its own (`flash_q`, a block per 64
  queries summing over the key tiles in registers) next to dK / dV (`flash_k`). The same training twice gives the
  same weights bit for bit, checked on an RTX 3060 (`APSI_GPU_DETERMINISTIC=0`: the one-kernel attention backward
  with atomic dQ).
- **The processor never waits for the card inside a step:** copies to the card go through pinned memory and are
  queued behind the kernels already launched (`APSI_GPU_ASYNC_COPY=0` turns this off; `drop_ring()` gives the memory
  back); `fit_batches` keeps each batch's loss on the card and reads them once an epoch; tanh gates (gated attention)
  are computed, applied and differentiated on the card (`tanh_s` in float64, rounded once; `axpy_s`; `gate_g`).
  A small transformer's step on an RTX 3060 (a desktop in use, best of several runs): 5.3–6.0 ms before, 4.0–4.7
  ms after; single runs vary by about ±15 %.
- **Cross-entropy on the card with accurate exp and log** (Cody-Waite exp, log with a Newton step) instead of the
  approximate hardware instructions.
- **A loss of your own can keep its large part on the card:** `loss.on_card = (output, labels_of, rest)` computes
  that output's cross-entropy on the card and sends only the other outputs to the processor. `gla_loss` uses it: the
  writer's logits never leave the card (one card and many).
- **Optimizers on the processor keep float64 weights** (a master copy, as NumPy does); the card gets them rounded
  once. Frozen weights stay frozen in the master copy too.

### Training that finishes
- **`steps=N`** (`fit_batches`, GPU trainer and NumPy): N optimizer steps over as many epochs as it takes.
- **Checkpoints that resume bit for bit:** `checkpoint="run.npz"`, `save_every=K`, `resume=true`. The weights, the
  optimizer's moments and step, the carried rounding error, the float64 master copy and the position in the data
  (epoch and batch), written atomically every K steps, at the end and **when training stops on any error or Ctrl+C**.
  Running the same call again continues exactly: the result equals one uninterrupted run, with or without
  `accumulate`. A checkpoint of another model is refused. `callback=` after every epoch.
- **A guard against non-finite steps:** a step whose gradient holds inf or NaN is skipped inside the adam / sgd
  kernels (weights, moments and the carried rounding error unchanged), on one card, copies and split models; the same
  on NumPy and for optimizers on the processor (`APSI_GPU_GUARD=0` turns it off).
- **Clearer out-of-memory message:** suggests `accumulate=` with a smaller batch.
- `docs/TRAINING.md`: steps, checkpoints, crash protection, exactness, speed, the tokenizer, every setting and what
  turning it off changes. `config/apsi-training.env`: every setting with its default.

### Tokens and tokenizers
- **`tokens.focus(context, query, budget)`** (`use tokens`): the parts of a long context a question needs, within a
  token budget, chosen on GLA's graphs — Python cut along its syntax tree (long functions into their parts) and linked
  by GLA's code property graph (calls, data flow), text cut into blocks and sentences and linked by shared rare words;
  scores from the question's words (rarer words count more, identifiers split into their pieces) spread by a
  personalised PageRank. Left-out parts become numbered markers; functions that do not fit may stay as outlines.
  `unfold(f)` gives the original exactly, `unfold(f, [n])` one part. `fit_chat(messages, budget)`,
  `outline(source)`. `examples/46_focus_and_tokenizer.apsi`.
- **`data.train_tokenizer(source, vocab_size=…)`:** byte-level BPE (GPT-2 / Llama 3 / Qwen kind) trained in Apsi —
  distinct pre-tokens counted once, merges that touch only the words with the pair, a heap with lazy deletion, ties by
  text so the same corpus gives the same tokenizer — or in Hugging Face `tokenizers`' Rust trainer when installed
  (`engine=`). Saved as tokenizer.json (loads in Apsi and Hugging Face); `text_dataset(tokenizer=tok)` takes it.
- **Fixed:** a list of texts given to `text_dataset` that contained `*`, `?` or `[` could be taken for file patterns.

### Tensors on the card
- **`gpu_tensor(x)`:** a tensor on the card with Apsi's own kernels — products (batched, transposed views without
  copies), arithmetic with tensors, row vectors and numbers, activations, softmax, layer and RMS norms, sums and
  means, row lookups, reshape. Any other function sees an ordinary tensor, so code written for the processor runs
  unchanged; without a card everything runs on the processor. `gpu_device()`. Element-wise kernels of their own:
  exp, log (Cody-Waite reduction, a Newton step: within a few units in the last place), IEEE division and square
  roots, powers, maximum, minimum, abs.
- **`apsi gpu bench check`** also checks GLA (graphs, exact gradient sums, the same weights twice) and gpu_tensor
  against the reference; `apsi gpu bench cards` checks that every card gives one card's weights bit for bit.
  `tools\gpu_check.bat`: the whole check on a Windows computer's card, written to gpu-check.txt.
- **Faster record fields:** reading a record's field in the bytecode engine is 1.8x faster.
- `.numpy()` works on every result of `gpu_tensor` code, also on a value the processor computed (a fallback, or
  no card), so the same program runs with and without a card.

### Training modes
- **Any optimizer on the GPU trainer:** adam, adamw and sgd keep their kernels; rmsprop, adagrad or an optimizer of
  your own (anything with `step(params, grads)`) steps on the processor with the weights as the model's tree, on one
  card and across cards (one optimizer per stage). The same training as NumPy.
- **Frozen parts:** `fit(…, freeze=["enc", "layers[0]"])`, `train_only="w*"` (names by path, `*` at the end, or a
  function of the path) on NumPy, one card and many: the frozen weights get no gradient, do not count in `clip`,
  and keep their exact values (weight decay and momentum cannot move them). Fine-tuning, or GLA's phases (one brain
  at a time).
- **Frozen means untouched, also in its type:** the result's frozen weights are the very arrays given (float64 stays
  float64), on every backend; before, training in float32 could hand them back rounded to float32.
- **Any loss name on the card:** a loss without a kernel (`"vae"`, …) is evaluated on the processor, like a loss
  function.
- **Variants without a kernel run inside the model on the card** instead of refusing (an activation such as mish,
  `graph_pool("max")`, `sage(aggregate="max")`, `parallel(combine="list")`, …).
- **`fit()` on many cards** trains a loss function, another optimizer or frozen parts through `fit_batches`' cards
  instead of refusing. `fit_batches(…, accumulate=A)` on many cards takes A batches a step (default 1, as on one
  card).

### Graph-and-language models
- **GLA 3.0 as a model of `use layers`:** `gla(size)` ("21b", "3b", "small", "tiny") builds the whole model of
  GLA.py / gla_brain_dual.py — node encoder, Brain A, Brain B, Brain C, bridge, FlyPrompt router, graph memory, the
  writer with gated cross-attention and tied embeddings — with exactly the parameters of GLA's own formula, part by
  part (`gla_count`, `gla_parts`; 21b: 20,480,386,836). `gla_batch`, `gla_loss` (GLA 3.0's losses), `gla_task`,
  `gla_accuracy`. It trains on NumPy, on one card and split across cards, the same training step for step.
  `docs/GLA.md`, `examples/45_gla.apsi`.
- **Real code as GLA input:** `gla_from_code(source, text, answer)`, `gla_code_graph(source)` (GLA's Python code
  property graph builder: AST, CFG, scope- and order-aware DFG / PDG, calls, SDG; GLA's structural codes),
  `gla_text_graph(text)`.
- **Brains that own node types and relations:** `types=[…]` and `relations=[…]` on `hgt`, `typed_linear`,
  `typed_norm`, `global_attention`; nodes of other types pass through, edges of other relations are not read, so
  several brains share one graph. `hgt_brain` (input projection, hgt layers, global attention, GLA's centred graph
  context with one shared projection and a gate per layer, final norm), `typed_pool` (a learned weight per node type).
- **`global_attention` as GLA's:** `ratio=` (fewer heads of the same size), `learned_keys=true` (values scaled by
  2 sigmoid(score); above `max_keys` the best-scored keys), `bias=`, `zero_init=`, `types=`.
- **Attention:** `multi_head_attention(…, bias=false, gate=0.25, memory_mask=true)` — no output bias, the output
  scaled by tanh of a learned gate, a memory whose last column says which slots may be read; `tied_output(owner)`,
  `slice_layer`. `node_encoder`, `brain_memory`, `expert_router`, `gated_fusion`, `param_bank`.
- **GPU trainer:** kernels for all of the above (typed groups by owned types, edges by owned relations, learned keys,
  graph context and typed pooling with the per-graph numbers on the processor, gated and masked cross-attention, the
  tied output); a `dag` with several outputs and a loss of your own over all of them, targets given as a record, on
  one card and across cards; a weight held by two stages (a tied output on another card than its embedding) gets the
  sum of both gradients; the cut of a dag balances the work each node really does (rows followed from the inputs:
  graph nodes, tokens, one row per graph after pooling). GLA's node encoder and graph memory have kernels of their
  own; each brain is a node per layer, so a cut can fall inside a brain (its shared weights summed across the cards).
- **`accumulate=A` across cards is A batches a step** whatever the number of cards and copies of the pipeline: the
  training no longer depends on how many cards there are.
- **Layers written in Apsi:** a `custom_layer`'s forward function gets only the keywords it takes (`train`, `graph`,
  `mask`), never the dag's own context; such layers train inside a model on one card or split across cards.
- **Fixed:** a lambda's parameter named like something outside it (`fn(p, x, train=true) => …` after `use nn`) could
  not be passed by keyword.
- **Heterogeneous graphs:** `graph(…, node_types=)`, `typed_linear`, `typed_norm` and `hgt` (the layer of GLA's brains:
  weights per node type, keys through each relation and head, one softmax over a node's in-edges) on NumPy and the
  GPU trainer (rows grouped by type, one product per type; the relations in one batched product).
- **Any model split across cards in batches:** `fit_batches(…, device="cuda:all", accumulate=M, stages=)` cuts a
  dag (several inputs, several values crossing every cut) or a sequence into stages, with a different graph per
  micro-batch; the same training as one card, step for step.
- **`fit_batches`** (`use layers`, every backend): batches that each bring their own shape and graph; `dag` models
  with several inputs (token ids and node features) on the GPU trainer; `graph_memory` on the card; cross-attention
  from text to a graph's memory. A GLA-shaped model (graph transformer, graph memory, a writer with rotary positions,
  grouped heads and SwiGLU) trains on the card exactly as on NumPy.

### Physics-informed networks
- **`pinn_fit` and `pinn_derivatives`** (`use layers`): the loss is any formula of the points and the network's
  values, first and second derivatives along each input (`loss(x, u, du, d2u)`): equations, boundary and initial
  conditions, measurements, systems of equations. On the GPU in Taylor mode (values and derivatives through the
  network in one stack, new kernels with the activations' first three derivatives); on the processor with autodiff of
  any order, for any model. The two give the same training. `docs/PINN.md`,
  `examples/44_physics_networks.apsi` (the heat equation, error 0.001; a damped oscillator).
- **`sin` and `cos` activations** on every backend and in the GPU trainer.

### One model across many cards
- **Model parallelism on Apsi's own GPU trainer:** `fit(..., backend="gpu", device="cuda:all")` splits a model that
  does not fit one card across the cards (`parallel="auto"`), or on request (`parallel="pipeline"`). The model is cut
  into stages where the work balances and every stage's weights, gradients and optimizer state fit its card; each
  card holds only its stage, so the trainable size grows with the cards, with no limit on their number. The default
  `device="cuda"` takes every card by itself when the first one cannot hold the model.
- **A pipeline, not a relay:** a batch flows through the stages as micro-batches in a one-forward-one-backward order,
  every card on its own thread, so the cards work at the same time; only the activations at the cuts travel between
  them (card to card where the driver allows).
- **Split and copied at once:** more cards than stages copy the pipeline (stages x replicas); the copies of a stage
  sum their gradients and share the optimizer's work.
- **Your own cut:** `stages=4`, or `stages=[6, 4, 4, 6]` (the units in each stage); `micro_batches=`. `r.stages`,
  `r.replicas`, `r.micro_batches`.
- **Modern transformer blocks on the GPU trainer:** grouped-query attention (`kv_heads=`), rotary positions
  (`rope=true`, a kernel of its own, in place), per-head query / key RMS norms (`qk_norm=true`), `swiglu` feed-forward
  blocks, `transformer_block(…, norm="rms", ffn="swiglu")`; `transformer_lm(…)` takes them all. The same layers on
  NumPy autodiff and PyTorch.
- **Sparse graphs on the GPU trainer:** `gcn`, `sage`, `gin`, `gat`, `graph_pool` with `fit(…, graph=g, mask=…)`, node
  and graph tasks. The edges are kept in compressed rows (CSR) and every node sums its edges in a register: no atomic
  additions, the same result every run. New kernels: sparse products with per-edge, per-head weights, the edge
  softmax and its gradient, edge dot products.
- **Mixture of experts:** `moe(d, experts=8, top_k=2)` and `transformer_block(…, ffn="moe")` on NumPy, PyTorch and
  the GPU trainer: the router's top-k choice and its softmax in one kernel, the rows grouped by expert, every expert
  on its own rows (no padding, no row dropped), the outputs summed back by the gates without atomics.
- **Long-range graph layers on the GPU trainer:** `graph_transformer` (attention along the edges, keys and values
  by relation, biases by relation and hop), `global_attention` (every node over the nodes of its graph), `mix_hop`,
  `graph_transformer_net`: the encoder of a graph-language model trains on the card.
- **Convolutions on the GPU trainer:** `conv2d` (windows unfolded into rows, one product, folded back by a thread
  per input value: no atomics), `max_pool` (ties share the gradient, as on NumPy), `avg_pool`, `global_avg_pool`;
  `cnn` trains on the card. **Recurrent layers:** `lstm`, `gru` (a kernel per cell, the input products of all the
  steps at once), `last_step`.
- **Any layer and any loss with the GPU trainer:** a layer with no kernel of its own (a custom layer, a lambda,
  batch_norm, bilinear, graph_memory) runs on the processor with Apsi's autodiff inside a model trained on the card,
  and a loss written as a function is evaluated there; the rest stays on the card.
- **More exact weights:** the optimizer kernels carry the rounding error of every update to the next
  (compensated summation), so updates far below float32's step at a weight's size still add up (300 Adam steps of
  1e-9 move a weight of 1 by 3e-7; a plain float32 update leaves it unchanged). On by default below 2 B parameters
  (`compensated=`), on one card, several cards and split models.
- **More activations on the card:** elu, softplus, mish. **More losses:** `mae` and `bce` (one kernel, with a weight
  per element for masks), on one card, several cards and split models.
- **`apsi gpu bench`:** tokens per second of the 127 M GPT (or `11m`, `modern`), the kernels' shares, temperature,
  power and energy per token, a JSON line to send back; `apsi gpu bench check` compares every new kernel on the card
  with the exact reference; `apsi gpu bench cards` checks a model split across all cards against one card.
- **Any architecture on the GPU trainer:** `dag(...)` (values by name, gradients added where a value feeds several
  layers), `parallel`, `add` / `multiply` / `concatenate` / mean, `scale`, `identity`, `reshape`, `flatten`,
  `mean_pool` (so `transformer_classifier`), and cross-attention: `multi_head_attention` on `["query", "memory"]` with
  any memory length, grouped heads, rotary positions and q/k norms.
- The same result as one card with the whole batch (tested with Adam, AdamW, SGD, clipping over the whole model, EMA,
  EWC, low-bit precision); a failure on any card stops all of them at once with that card's error.
