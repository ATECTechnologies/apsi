# Changelog

## 2.6.0 — exact training on any number of cards, GLA 3.0, training that finishes, fewer tokens (13 November 2026)

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
- **bfloat16 training no longer keeps compensated weights by default:** the compensation (4 bytes per parameter and
  a fifth array for AdamW to read and write every step) stays on by default for fp32 training, where it keeps
  updates smaller than float32's step; in bf16 the products round every gradient far more coarsely, so it only cost
  memory and time: 0.5 GB less on the card for the 127 M GPT, and a lighter optimizer step. `compensated=true` or
  `APSI_GPU_COMPENSATED=1` turns it on.
- `APSI_SPLITK=off|2|4|auto` fixes how the weight-gradient products split their long k (for tuning a card);
  `tools/tune_gpu.bat` measures every setting on this card.

### Converting Python: one file, its imports, or the whole project
- `apsi convert app.py` converts the file and every module of the program it imports (as before);
  **`--only`** converts just the files named and keeps the program's other modules as Python imports;
  **`--project`** converts every `.py` of the project the file belongs to (the folder with `pyproject.toml`,
  `setup.py`, `setup.cfg`, `requirements.txt`, `Pipfile` or `.git`); **`--exclude PATTERN`** leaves files out;
  **`--plan`** lists each file, where it goes and why, and writes nothing.
- VS Code: *Convert Python to Apsi* asks how much: this file and its imports, this file only, the whole project, or
  the plan first.
- `pyimport` also finds modules in the program's own folder (a module kept as Python next to a converted program).
- **Real libraries convert and run:** checked on a downloader that brings datasets, huggingface_hub, fsspec, pyarrow,
  multiprocess and dill with it (603 library modules): `from . import Name` takes a name the package re-exports (or
  provides at run time) instead of failing; `import pkg.sub` inside pkg uses the converted package, never a second
  Python copy; an import cycle sees the partly run module, as in Python; a module is in `sys.modules` before it runs
  and becomes its parent's attribute however it was first loaded; one module per file however its path is spelled;
  fields declared under an `if` of a class body; `globals()` / `locals()` / one-argument `eval` / `exec` see the
  caller's namespace; `with (a / b).open() as f:`; a bare `return` in an async generator.

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

## 2.4.4 — Python, exactly; real time; data, databases and the web (10 November 2026)

### Training on the GPU with Apsi alone, low-bit formats, an AI server
- **`fit(..., backend="gpu")`: Apsi's own GPU trainer** — CUDA kernels written in Apsi's code generator and loaded
  through the driver: no PyTorch, no CUDA toolkit, nothing to download. Matrix products on the tensor cores (any
  strides, so transposes and attention heads need no copies), layer/RMS norm, attention softmax, embeddings,
  cross-entropy, AdamW / SGD over all parameters in one kernel, gradient clipping on the card, the data set on the
  card. Gradients equal Apsi's autodiff (tested to 1e-5) and the training curve equals the NumPy backend's.
  `backend="auto"` uses it whenever there is an NVIDIA card and the model fits its layers.
- **Low-bit training:** `precision="fp8" | "fp6" | "fp6_e2m3" | "fp4" | "int8" | "int4"` on the GPU trainer, NumPy and
  PyTorch; real fp8 products on Ada / Hopper / Blackwell tensor cores with PyTorch.
- **fp6 on the card:** `APSI_GPU_WEIGHTS=fp6` (E3M2) and `fp6_e2m3` (E2M3: as close as fp8 at 6 bits), bit-exact
  with ml_dtypes.
- **GPU temperature control:** long training, long answers and busy servers keep the card near 77 °C (75–78) by
  pacing the work while it is above the band (the driver's NVML gives the temperature); a cool card is never slowed.
  `APSI_GPU_TEMP=72` / `off`; `r.gpu_temperature`, `/metrics`, `hub.gpu_status()`.
- **The installer sets up full GPU speed:** an option (on when an NVIDIA card is found) installs the GPU helper with
  one administrator prompt, so ordinary, non-administrator runs hold the temperature by the power limit too.
- **Less work per step:** the optimizer kernel clears the gradient it has used (no separate pass over all the
  gradients each step), and each weight matrix is rounded to bfloat16 once per update instead of once per product.
- **A fourth build of the products:** the operands rounded to bfloat16 once (one pass each), then copied into shared
  memory with cp.async four slices ahead (no registers, no conversion in the loop); the card times it against the
  others per shape, its rounding pass included, and keeps the fastest.
- **Several cards:** `device="cuda:all"` (or "cuda:0,1,…") trains data-parallel: reduce-scatter of the gradient
  (each card sums its slice, all cards at once), the optimizer's state and update sharded, all-gather of the
  weights; the same result as one card with the whole batch (tested). No size is refused in advance (a warning
  with the numbers; a clear error if the card runs out).
- **`apsi gpu hold 77`:** holds the card's temperature by its power limit for any program (used to measure PyTorch
  under the same conditions as Apsi).
- **Large models:** a third build of the tiled products (4 warps of 64 x 64 over slices of 16), tiles visited in
  groups of 8 rows (large weights hit the L2 cache), and a size check before training starts.
- **Measured after the fix (RTX 3060): 151,008 tokens/s for the 10.9 M GPT (PyTorch bf16 133,491), 17,134 for
  127 M parameters (PyTorch bf16 20,111).**
- **Fused attention works again** (a build error had left the trainer on the slower separate kernels); every kernel
  module is now built by the test suite, without a card, so such a mistake fails there.
- **The Linux release zip carries the Apsi GPU Tuner** (gpu-tuner/), built offline by packaging/make_release_zip.sh.
- **GPU trainer:** `ema=` (moving average of the weights) and `ewc=` (elastic weight consolidation); each tiled
  product shape runs the faster of two builds, measured on the card; split-k sized from the real occupancy; the
  temperature pacing counts the card's queued work (large models were paced too little and ran hot).
- **Faster matrix products on the card:** the loads of the next slice now overlap the tensor-core work (LLVM had
  moved them after it, so every slice waited the full memory latency), the rounding to bfloat16 is one
  cvt.rn.bf16x2 per pair, and split-k fills whole waves of the card (float32 on the tiled kernels as three bfloat16
  products is there with APSI_FP32_TILED=1; the default stays the faster separate kernel).
- **Level pacing:** the temperature control follows the smoothed temperature and gathers its pauses (a card whose
  die jumps with each change of load was paced between none and most of the time: 62-77 C and half the speed).
- **Apsi GPU Tuner** (tools/gpu-tuner, on the download page): Apply / Restore files for Windows and Linux that set
  every NVIDIA card to the recommended sustained-training power limit, kept after restarts, and undo it.
- **Steady speed at a steady temperature:** with the right to set the card's power limit Apsi holds the temperature
  with it (PI control, no pauses); `apsi gpu setup` (once) installs a helper so no program ever needs to run as
  administrator for that; `apsi gpu` / `apsi gpu watch` / `apsi gpu power 80%`.
- **A model works from any thread:** every call to the card makes its context current on the calling thread first.
  (Before, a server's request threads could not use the card and the model quietly moved to the processor.)
- **`docs/GPU.md`:** everything on NVIDIA cards in one guide — formats, training, serving, temperature, measured
  numbers, and troubleshooting.
- **`apsi serve-model` / `hub.serve(lm)`: the OpenAI API** (`/v1/chat/completions` with streaming, `/v1/completions`,
  `/v1/models`, `/health`, `/metrics`), first-come-first-served queue with a limit and 429, a client that leaves frees
  the model at once, time limits, API keys. `examples/31_ai_server.apsi` adds your own routes in Apsi.

### Long context, logs, training data, side-by-side installs
- **Twice the context in the same GPU memory:** the key/value cache is float16 by default (`APSI_KV_CACHE=fp32` for
  the old one), read four values at a time; long prompts split attention over more blocks of the card.
- **Long-context models work past their training length:** YaRN rope scaling (Qwen2.5 `rope_scaling`, Llama 3.1
  style) with the attention temperature, on the GPU and the processor alike.
- **`use log` and `apsi log`:** logging to the terminal (coloured) and to `.log` files, text or JSON lines, rotation
  by size or by day, named loggers with fields, `with log.timer("step"):`; `apsi log FILE --level warn --grep x -f`
  reads, filters and follows them (also across rotations). See [`docs/LOG.md`](docs/LOG.md).
- **Faster tokenizers:** BPE merges with a heap (long words were quadratic), Unigram results cached; the same tokens.
- **Downloads that do not give up:** an interrupted download resumes by itself (up to 8 times); large files use
  several connections only when that is faster on your network.
- **Training text in any shape:** `data.text_dataset` reads txt/md/jsonl/json/csv/tsv/parquet and plain text,
  chat (`messages`, ShareGPT), Alpaca, prompt/completion and question/answer records, or your own template; any
  tokenizer (hub, `"bytes"`, `"chars"`, a function); packed windows; the prompt part labelled -100, which
  `cross_entropy` now leaves out.
- **Mixed precision chosen by itself:** `fit(..., backend="torch")` defaults to `precision="auto"` — bf16 on GPUs
  that have it, fp16 with loss scaling on older ones, fp32 on the CPU; any name or `{store, compute}` pair to choose
  yourself, `APSI_PRECISION` for every program.
- **`apsi convert --idiomatic` never changes what a program does:** a file that uses something the idiomatic form
  cannot express exactly (match, async, metaclasses, `__exit__`, exception subclasses, loop variables read after the
  loop, late-binding closures, shared list defaults, complex numbers, a line it could not translate) is converted with
  the exact form instead, and the converter says why. Keyword-only parameters after `*args` are kept in place.
  24 test programs give the same output as Python.
- **Several versions side by side:** 2.4.4, 2.5 … each install in its own folder (`<Apsi folder>/<version>`), none
  removed or overwritten; `apsi` runs the newest (or the one chosen with `apsi use 2.4.4`), `apsi-2.4.4` always that
  one, `apsi versions` lists them. Removing a version keeps the others; an install from before 2.4.4 is moved into
  its own version folder. Both the Windows/Linux setup app and the Linux `.sh` installer do this.
- Launchers start Python with `-P`, so a folder named `apsi` in the current directory can no longer replace Apsi.
- **Models on the processor twice as fast and half the RAM:** int8 matrices (Q8_0 layout, error below bfloat16's
  rounding; made once and memory-mapped afterwards) and a native thread team that starts a product on every core in
  about a microsecond. A Qwen2.5-0.5B-sized model: 96 → 52 ms per token on 2 cores. `APSI_CPU_WEIGHTS=bf16` for the
  exact weights.
- **Large models whole on the GPU:** weights are quantized on the card itself to int8 or int4 (32 values per scale;
  int4 as ggml's Q4_0) with new matrix-vector kernels that read them directly and a dequantize step for the
  tensor-core prompt product. `gpu_weights="auto"` (`APSI_GPU_WEIGHTS`) keeps bfloat16 when the model fits, else
  int8, else int4, and turns layers back into int8 where room is left — a 14B model on a 12 GB card runs all 48
  layers on the GPU instead of 16.
- **Split models done properly:** the processor's layers use the native int8 products (they fell back to float32
  NumPy before), their int8 copy is cached on disk and memory-mapped (also for memory-mapped checkpoints, whose copy
  stayed in RAM and could fill it), long prompts through them run on the GPU, and loading shows progress lines.
  `lm.plan` tells what runs where, in which format, from RAM or SSD.
- **`@gpu` calls reuse device buffers** instead of allocating and freeing them on every call (cuMemFree waits for
  the GPU): faster video frames and loops.
- **Benchmarks in four more languages:** Java, JavaScript (Node.js), C# (.NET) and Julia join C, C++, Rust and Go in
  `benchmarks/heavy` (identical checksums); `benchmarks/tokens` and `benchmarks/gpu` added.
- **Prompts on tensor cores** (RTX 30 series and newer): bfloat16 `mma.sync` with the activations split in two
  bfloat16 parts (matches float32 to about 1e-5), checked against the float32 kernel at start-up.
- **`save_checkpoint` / `load_checkpoint`:** weights, optimizer state, step, config and files (tokenizer …) in one
  `.apsim`, or shards above 4 GB; loads from the file, the index, the bare name or any shard; resuming is exact.
- **`use config`:** settings in their own files (TOML, JSON, YAML, INI, .env), layered, overridden by environment
  variables and `--set key=value`, types checked against defaults, `${other.key}` references, `config.save`.
- **`diagnose(model, params, X, y)`:** checks a model built from scratch before a long run (data, shapes, first loss,
  dead or exploding gradients, whether it can fit a small batch) and suggests a learning rate.
- **`apsi cache`:** every cache and its size, downloaded models, and `apsi cache clean` for what can never be used
  again (or one model, or everything).

### Language models on the GPU (`use hub`)
- **Decoders run on an NVIDIA graphics card by themselves, with nothing to install but the display driver.** The
  whole network runs on the card (weights as bfloat16, key/value cache, norms, rotary embeddings, attention, MLP) in
  kernels Apsi generates as PTX and the driver compiles: no CUDA Toolkit, CuPy or PyTorch. Measured on an RTX 3060
  with Qwen2.5-0.5B: 193 tokens/s greedy, 190 sampled, 211 through `lm.chat` (7.7 on the processor before this
  release); a prompt of 512 tokens in 118 ms, of 2,048 in 0.81 s; a token after 2,000 tokens of context in 5.5 ms;
  the model ready in 0.86 s.
- **Starts in a fraction of the time:** bfloat16 checkpoints go from the file to the card as they are (no float32
  copy); the driver starts while the file is read; kernels are compiled once and cached; a model downloaded before
  starts with no network request (the Hub is asked for a newer version at most once a day; without a connection the
  cached copy is used).
- **Prompts in passes of up to 512 positions** with a tiled matrix product; single tokens with a product that normalises
  its input on the fly; attention with a warp per key.
- **Models larger than the card:** the first layers that fit run on the GPU, the rest on the processor, from RAM or
  memory-mapped from the SSD (`gpu_layers=`, `memory="disk"`); `device="cuda"` insists on the card alone.
- **Steady under failure:** out of GPU memory frees scratch space and continues in smaller passes; any other driver
  error moves the model to the processor, which carries on from the tokens already produced, reported once.
  One model serves several threads.
- **`lm.chat(messages)` and `lm.conversation()`:** the model's own chat format (ChatML, Llama 3, Gemma, Mistral),
  stops at the end of the turn, oldest turns left out when the window is full; `hub.devices()` lists the GPUs.
- **The next token is chosen on the card:** penalty, temperature and softmax there; only the candidates come back.
- **Conversations reuse their cache:** a prompt that starts like the previous generation computes only the new part.
- **On the processor, bfloat16 products on all cores** (native code from Apsi's compiler): half the memory of
  float32 NumPy and faster per token (124 → 91 ms on the same machine).
- Faster sampling (top-p sorts only the likely tokens), streamed text that never splits a character, the output head
  computed for the last position only (long prompts on the processor were paying for every position), no forward
  pass after the last token, weights read from the file only when needed (a GPU model never holds a copy in RAM).

### Faster everyday code, clearer errors
- **Objects 4–6× faster:** `self.x` inside a method of an Apsi class compiles to Python's own attribute access;
  other attribute reads and writes on ordinary objects, text, lists and dicts skip straight to it after the first
  check. Creating objects and calling methods: 0.646 s → 0.102 s for 300,000 (CPython: 0.041 s).
- **`str()` / f-strings / `print` of numbers and text, conditions, dict and list stores** take their common case
  first: f-string building 1.8×, dict updates 1.7×, float loops 1.5× faster.
- **Every error says how to fix it, and the caret points at what failed** (not the start of the line):
  `'ad' is not defined (did you mean 'add'?)`, `Str has no attribute 'uper' (did you mean 'upper'?)`,
  `cannot join Str and Int with '+'; … write str(x)`, `'[' is not closed: expected ']' … at the end of line 1`,
  `'abc' is not a whole number`, `key not found: 'b' (use d.get(key, default) …)`, index, tuple, `len()`,
  unpacking (`for k, v in d.items()`) and classes without `__init__`.
- **Habits from other languages are translated:** `def` → `fn`, `else if` → `elif`, `print "x"` → `print(x)`,
  `x++` → `x += 1`, `&&` / `||` / `!` → `and` / `or` / `not`, `if x = 5` → `==`, `True` / `null` → `true` / `none`,
  `import os` → `use os`, `3[m] + 1` → give the number a unit.
- A parameter without a default after one with a default is reported at the parameter (it used to stop with a
  Python traceback); no program can show a Python traceback for a compile problem any more.

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
  Measured against 2.0.0 on the same machine: `benchmarks/versions` (geometric mean 3.84× faster on a
  GitHub runner with the native compiler, 7.31× on a 2-core VM without it).
- **Matrix products written as loops run with BLAS.** Three nested loops computing `c[i, j] += a[i, k] * b[k, j]`
  (any loop order, a hoisted `a[i, k]`, or the dot-product form with an accumulator) are recognised in the compiled
  engine and in `@native` code and run with the tuned BLAS kernel: 4–11× faster than the same loops in C with `-O3`,
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

## 2.0.0 — native machine code, all CPU cores and NVIDIA GPUs (9 November 2026)

Apsi 2.0 replaces "everything runs on the Python VM" with a native compiler. Python stays where it helps (the
tooling, libraries, the rest of the language), while numeric code runs as machine code built through LLVM.

### Python compatibility — write Apsi the way you write Python
- **Exceptions:** `try / except Kind, Other as e / else / finally` and `raise` (with Python's kind names; errors Apsi
  raises itself — division by zero, missing keys, bad indices, unit mismatches — match them too). Own error types
  are structs with a `message` field.
- **Generators:** `yield` and `yield from` produce values lazily; endless generators work.
- **`*args` / `**kwargs`** in definitions and `*xs` / `**d` in calls.
- **Operators on structs:** methods `add sub mul div eq lt len get contains iter to_str call hash …` are used by
  `+ - * / == < len() [] in for print() …` (the conversion of Python's special methods).
- **Pure standard modules without permissions:** `pyimport("collections")`, heapq, itertools, functools, bisect,
  re, string, copy, datetime, fractions, decimal and more need no `--allow-python` (only public, non-module
  members are exposed, so nothing reaches files, network or processes).
- New built-ins `bor band bxor shl shr bnot` (Python's `| & ^ << >> ~` for ints, sets, dicts and boolean tensors),
  `chain_lists`, `set`, `frozenset`; `json_str(…, indent=, sort_keys=)`; named tuples print as `Point(x=1, y=2)`.
- **`apsi convert` keeps behaviour instead of commenting it out:** `try/except` blocks, `raise`, generators, variadic
  functions, `for/while … else`, class attributes, special methods, `field(default_factory=…)`, nested
  comprehensions, `filter(None, …)`, `sum(xs, start)`, `nonlocal`. `tests/test_python_compat.py` converts real
  Python programs and requires identical output.

### Website commands (from any computer)
- `apsi release upload FILE… --version V` (platform detected from the file name, resumable 32 MB pieces),
  `apsi release list / delete ID / hide / unhide`, `apsi release bench results.json --version V`.
- `apsi yank NAME VERSION [--undo]`, `apsi unpublish NAME VERSION`.
- `benchmarks/heavy/run.py`: one command, live-updating table in the terminal, `--publish V` updates
  apsi.dev/benchmarks while it runs, `--apsi-root` measures another Apsi version.

### Native compiler (`@native`) — new architecture
- **Checked AST → specialised, unit-erased LLVM IR → machine code for the host CPU** (`-O3`, loop and SLP
  vectorisation, AVX2 / AVX-512 / FMA / NEON as available). Each function is specialised on the types, tensor
  dtypes/ranks/layouts and physical units of its arguments; types are inferred by abstract execution.
- **Units at zero cost:** dimensional analysis happens at compile time (wrong units are a `UnitError`), the machine
  code works on SI numbers, and results get their units back. Argument units are still checked on every call.
- **Safe by default:** 64-bit integer overflow, index range and integer/float floor-division-by-zero checks with
  source lines; Python semantics for `//`, `%`, negative indices, `min`/`max`, `round`. Strict IEEE arithmetic, so
  results are bit-identical to the bytecode engine. Index checks are **hoisted out of inner loops** by proving
  their range before the loop (the loop is versioned, behaviour unchanged), which lets LLVM vectorise.
- **`@native("fast")`**: checks off, `nsw` integers, fast-math — for proven code. **`@native("strict")`**: an error
  instead of a fallback.
- **Tensors:** direct element access on contiguous and strided tensors, row and slice views without copies,
  **fused** element-wise expressions with NumPy broadcasting (one pass, no temporaries), fused reductions
  (`sum mean min max argmin argmax prod dot norm`), `zeros/ones/full/*_like`, zero-copy tensor returns, and
  per-iteration memory arenas so temporaries created in loops are freed every iteration.
- **Never breaks a program:** unsupported constructs or argument kinds fall back to the bytecode engine with one
  note naming the construct and line; `grad` of a native function, `evolve` calling native right-hand sides, and
  native ↔ bytecode calls all work.
- **Signed machine-code cache** (`~/.apsi/cache/native`, HMAC-SHA256 with the per-user key), keyed by IR, CPU,
  CPU features, LLVM and Apsi versions.
- **Benchmarks** (`benchmarks/`, [`docs/NATIVE.md`](docs/NATIVE.md#performance)): 100–3,500× faster than 1.10;
  `@native("fast")` fastest or tied single-threaded on Mandelbrot, spectral-norm, sieve and fused-array sums
  against GCC, G++, Clang, Rust and Go, within 6–20 % on n-body and matmul.

### Exact semantics
- Native code now follows the bytecode engine **exactly**, including where Python numbers and NumPy scalars
  (tensor elements) obey different rules (division by zero, Int overflow, `//` and `%` by zero, `0.0 ** -1`): every
  value's origin is tracked at compile time, or with a one-bit run-time flag when it depends on the path taken,
  through variables, branches, `min`/`max` and function returns. Results come back as the same Python class.
- `tests/test_native_fuzz.py`: seeded differential fuzzing of random programs with Python and NumPy arguments —
  values, errors and result classes must all match (0 differences in 14,400 calls). Bugs it found and that are fixed:
  `-0.0 // nan` (CPython treats NaN as true), `floor`/`ceil` return Ints like the bytecode engine, provenance lost
  through `abs` and Bool → Int.

### Parallelism
- **Native thread runtime**: dispatcher and workers in machine code, spinning workers with `pause`, a seqlock-style
  job hand-off, one cache line per shared field and guided chunk claiming — a parallel region starts in ~5 µs
  instead of ~70 µs; parallel speed-ups improved accordingly (spectral-norm 15.9 → 12.6 ms, matmul 20.9 → 15.0 ms
  on 2 cores). Stress-tested with hundreds of thousands of regions, errors mid-region, nesting, thread-count changes
  and concurrent callers, including under gdb; this found and fixed a use-after-return race in the error path and
  an out-of-bounds read.
- **Call latency** from bytecode into native code: ~0.9 µs (was ~8 µs) through a signature cache, a direct typed
  entry point and tensors passed as the NumPy object (struct layout measured and verified at start-up); Ints beyond
  64 bits are routed to the bytecode engine instead of being truncated.
- **`parallel for`** (soft keyword): iterations on all CPU cores in native code; deterministic reductions
  (`+= -= *=`, `x = min(x, …)`, `x = max(x, …)`) that give **bit-identical results for any thread count**; one
  thread wake-up per core per loop (chunks are claimed by native workers with an atomic counter). An ordinary `for`
  on the other engines, so programs stay portable.
- **Automatic multi-core kernels:** fused tensor expressions and `sum`/`mean`/`prod`/`min`/`max` over 131,072 or more
  elements run on all cores without any code change, with deterministic chunked combination;
  `native_threads(n)`, `APSI_THREADS`, `apsi run --threads N`.

### GPUs (`@gpu`)
- A `parallel for` becomes a CUDA kernel: Apsi generates **PTX for sm_75** (GTX 1660 / RTX 20xx and newer) with
  LLVM's NVPTX backend and loads it through the **CUDA driver API** (`nvcuda.dll` / `libcuda.so.1`) — no CUDA
  toolkit needed. `+=` reductions use GPU atomics.
- **Multi-GPU:** iterations are split across all GPUs (`APSI_GPUS` to choose); each GPU copies back only the rows it
  wrote. Without a GPU, or for kernels outside the GPU rules, the function runs on all CPU cores instead.
- `gpu_info()`; `use gpu`'s `gpu_info()` now works with only the NVIDIA driver (it required CuPy and crashed without it).

### Large models
- **`apsi plan` / `use scale`**: plans training of 1B–2T-parameter models, dense or mixture-of-experts, BF16 or
  FP8, on 13 GPU types: tensor × pipeline × data-parallel layout, recomputation, memory per GPU, GPU count, nodes,
  days, GPU-hours, dataset size and serving memory, with the standard published accounting
  ([`docs/SCALE.md`](docs/SCALE.md)).
- **`@` in native code** runs on the multi-threaded BLAS (the same kernels as NumPy), so matrix-heavy native
  functions get BLAS speed plus fused element-wise work around it.

### Science, engineering and medicine
- New standard modules, all verified against reference implementations (`tests/test_science.py`):
  - `use special`: gamma, digamma, beta, error functions and inverses, regularized incomplete gamma/beta and
    inverses, Bessel J and I, elliptic K and E, Legendre/associated Legendre, Hermite, Laguerre, Chebyshev,
    spherical harmonics, ζ, E1. Relative errors are 1e-16 to 1e-13 against SciPy.
  - `use stats`: 11 distributions (pdf/pmf, cdf, sf, ppf, sample), descriptive statistics, t-tests, ANOVA,
    χ², Fisher's exact test, Mann–Whitney, Wilcoxon, Kruskal–Wallis, exact two-sample KS, Jarque–Bera,
    Pearson/Spearman/Kendall, binomial test, Wilson and Clopper–Pearson intervals, OLS and logistic regression,
    power and sample size (exact noncentral t), bootstrap, permutation tests, Bonferroni/Holm/BH/BY. Results agree
    with SciPy and statsmodels to 1e-10 or better.
  - `use signal`: Butterworth design (second-order sections), zero-phase filtering with SciPy-identical edges,
    notch, windows, convolution, peak detection (height, distance, prominence), Welch, spectrogram,
    Savitzky–Golay, resampling, Hilbert envelope.
  - `use mechanics`: Verlet, 4th-order Yoshida and RK4 integration; Hamiltonian and Lagrangian dynamics whose
    equations of motion come from automatic differentiation; N-body gravity with energy and momentum
    tracking; Kepler's equation, orbital elements, Hohmann transfers; inertia tensors, quaternions and torqued
    rigid bodies; normal modes; Euler–Bernoulli beams and plane trusses by finite elements.
  - `use pde`: heat (Crank–Nicolson, ADI), wave, advection, Poisson by fast sine/Fourier transforms, 2-D
    incompressible Navier–Stokes (pseudo-spectral), conjugate gradients, BiCGSTAB, batched tridiagonal solver.
  - `use quantum`: Schrödinger bound states (fourth-order) and split-operator time evolution, expectation values
    and uncertainties, operators, spin matrices, qubit gates and circuits, measurement sampling, density matrices,
    partial trace, entropy, fidelity, concurrence, hydrogen.
  - `use physics` gains formulas with units: Planck, Wien, Stefan–Boltzmann, Lorentz transformations, Doppler,
    Snell/Fresnel/Brewster, thin lens, diffraction, Coulomb fields, exact Biot–Savart, Lorentz force, kinetic
    theory, decay.
  - `use medical`: one- and two-compartment pharmacokinetics with repeated dosing, non-compartmental analysis;
    SIR/SEIR, final size, incidence rates, RR/OR/RD/NNT, Mantel–Haenszel; diagnostic accuracy with confidence
    intervals, ROC with DeLong intervals and paired comparison, Cohen's kappa, Brier score; Kaplan–Meier, log-rank,
    Cox regression (Efron ties) and Harrell's C, matching lifelines; eGFR (CKD-EPI 2021), Cockcroft–Gault, QTc, BSA,
    BMI; ECG R-peak detection and HRV; DICOM reading (pixel-identical to pydicom, identifiers left out by default)
    with 3-D series and windowing. Marked as research and education software, not a medical device.
  - `use lab`: one seed for every generator, provenance records, JSON-lines experiment logs, measurement
    formatting, Markdown/LaTeX/CSV tables.

### PyTorch programs → native Apsi
- New module `use torch`: PyTorch's API implemented natively with PyTorch's parameter names and exact results —
  layers (convolution with groups/dilation/`"same"`, pooling, BatchNorm with running statistics, LayerNorm, RMSNorm,
  GroupNorm, Embedding, multi-layer and bidirectional LSTM/GRU/RNN, MultiheadAttention with every mask option,
  Transformer encoder/decoder), losses with all options (`weight`, `ignore_index`, `label_smoothing`, `pos_weight`,
  `reduction`), eight optimizers, fifteen learning-rate schedulers including `OneCycleLR` and `ReduceLROnPlateau`,
  `DataLoader` over tensors or your own `Dataset` classes, `state_dict` / `load_state_dict`, gradient clipping and
  gradient checkpointing.
- `apsi convert` now targets `use torch`, so schedulers, BatchNorm, attention, RNN states, custom datasets and loss
  options are converted exactly instead of being reported for review. A full training run (convolution, BatchNorm,
  LSTM, attention, AdamW, cosine annealing, clipping, label smoothing) ends with the same weights as PyTorch to 1e-9.
  `torch.save` / `safetensors` files keep PyTorch's names, so weights move between PyTorch and Apsi unchanged.
- Training state is computed once per step (`value_and_grad_aux`): values from the forward pass stay available
  without a second forward pass.
- New: `layers.custom_layer`, `layers.load_torch_weights`, `nn.loader`, more `nn` losses, the built-ins `swapaxes`,
  `permute`, `unsqueeze`, `squeeze`, a differentiable `erf` and `clip`, and the exact GELU.

### Reinforcement learning (`use rl`)
- Vectorised environments (CartPole identical to Gymnasium's, grid world, bandits), GAE, PPO (reaches the maximum
  CartPole return), double DQN, REINFORCE, DPO / IPO / Bradley–Terry / GRPO / KL-penalised RLHF objectives,
  Monte Carlo tree search, beam search and best-of-n. Guide: `docs/RL.md`.

### Memory
- Gradient checkpointing (`torch.checkpoint`, `layers.checkpoint`): about a third of the peak memory for twice the
  compute, with identical gradients (dropout masks are replayed).
- float32 training keeps parameters, gradients and optimizer state in float32 end to end (`fit(…, precision="fp32")`).

### Libraries (`.apsil`)
- New package format and registry client: `apsi install NAME` (newest version, semantic-version ranges, dependencies,
  lock file), `apsi uninstall`, `apsi pack` (checks the code, builds a reproducible `dist/NAME-VERSION.apsil`),
  `apsi publish` with API tokens, `apsi search`, `apsi lib list / info / registry / login`; the official registry is https://apsi.dev. Every download is
  verified by SHA-256 per file and scanned by Apsi Guard. Installed libraries are used with `use NAME`.
- `use NAME` now also makes the module available as `NAME.member` (unless the name is taken).
- `apsi new NAME --template lib` creates a publishable library.

### Benchmarks
- New heavy suite in `benchmarks/heavy/` against C, C++, Rust, Go and Mojo with checksum verification.

### Requirements
- Apsi now needs **Python 3.12 or newer** (tested with 3.12, 3.13 and 3.14), NumPy 1.26+ and llvmlite 0.43+. The
  Windows installer and the portable Linux build bring their own Python 3.12; `import apsi` on an older Python
  gives a clear message.

### Model files
- **Apsi checkpoints (`.apsim`)**: `save_model` / `load_model` / `model_info`. Standard safetensors files that
  also keep physical units, compressed tensors (INT8/INT4/FP8/FP6/FP4/BF16) at their real size, and the parameter
  tree. A SHA-256 over the tensor table and data is checked when loading. Optional HMAC-SHA256 signatures use a
  per-user key or a shared key. Sharding with a Hugging Face style index supports 70B–2T-parameter checkpoints.
  Writes are atomic, and loading is memory-mapped.
- **GGUF**: `save_gguf` / `load_gguf` / `gguf_info`, version 3, F32/F16/BF16/Q8_0/Q4_0 and integer tensors, all
  metadata types. Q8_0 and Q4_0 output is bit-identical to the reference `gguf` package (0 of 1,331,200 bytes
  differ over 200 random matrices).
- `apsi model info | verify | convert` converts between safetensors, GGUF and `.apsim`.
- Hardened parsers: header sizes, offsets and tensor sizes are checked before mapping, and fuzzed files produce
  clear errors instead of crashes. The model functions are covered by permission checks and `apsi audit`.

### Tools
- `apsi native info | explain FILE | ir FILE FN | asm FILE FN | clear-cache`; `apsi run --native` (every function
  that allows it becomes machine code); `apsi doctor` reports the native compiler and the GPUs.
- New built-ins `native`, `gpu`, `native_threads`, `native_info`, `gpu_info`; editor grammars and snippets
  (`native`, `pfor`, `gpu`) for `parallel`, `@native` and `@gpu` in every supported editor.
- Examples `31_native_speed.apsi` and `32_gpu.apsi`; `tests/test_native.py` runs every native construct against the
  bytecode engine and requires identical results and errors.
- [`docs/NATIVE.md`](docs/NATIVE.md) (native code reference and benchmarks) and
  [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) (`apsi` not recognized and PATH, VS Code colours and
  Ctrl+F5, `needs permission 'python'`).

### REPL
- The prompt shows the current folder, like a shell: `C:\Users\me\project apsi>` (`~/project apsi>` on Linux; long
  paths shortened in the middle; `APSI_PROMPT=short` for the plain `apsi>`). `cd` no longer prints the folder,
  since the prompt shows it.
- `ls` / `dir` print an aligned table: folders first, then files, with modification time and size, a header and a
  summary line (`1 folder, 2 files (2.9 KB), 1 hidden (ls -a)`); empty folders say so.
- A line ending in an operator (`let x = 1 +`) now continues correctly: the REPL already waited for the next line,
  but the two lines were then rejected as a syntax error.

### Fixes
- **Windows setup wizard build:** the wheel builder read `pyproject.toml` with a regular expression and wrote the
  dependency `pyreadline3>=3.4; sys_platform == \"win32\"` with the escaping left in, which pip rejects
  (`invalid metadata: Expected a marker variable or quoted string`). It now uses a real TOML parser (`tomllib`,
  with a correct fallback for Python 3.9/3.10) and refuses to write malformed requirements.
- The wheel builder, `apsi build`, the installation manifest (`apsi verify`) and the playground bundle now include
  subpackages (previously only top-level files were packaged and verified).
- Installers say clearly that terminals and editors opened before installation keep the old PATH, and the Linux
  installer prints the command that fixes the current shell.

### Packaging
- New dependency: `llvmlite` (prebuilt wheels for Windows and Linux; it contains LLVM). The Windows payload and the
  Linux portable build bundle it; the shell installer installs it into a private environment when the system lacks
  it; the `.deb` recommends `python3-llvmlite`. Without it Apsi still runs, on the bytecode engine.

## 1.10.0 — first production release with installers (8 November 2026)

Development started on **Sunday, 8 November 2026 at about 04:00** (Bangkok time, UTC+7); this release was completed the same morning at about **09:00** — about **5 hours** of development.

### Command line and REPL
- **New REPL**: shell words work as in a terminal (`cd C:\Users\me\project`, `cd -`, `pwd`, `ls`/`dir`, `clear`/`cls`,
  `run app.apsi`, `exit`), any operating-system command with `!cmd`, and 26 `:commands` (`:help`, `:vars`, `:type`,
  `:doc`, `:time`, `:bench`, `:load`, `:reload`, `:run`, `:check`, `:test`, `:fmt`, `:edit`, `:paste`, `:save`,
  `:history`, `:ast`, `:tokens`, `:reset`, `:perms`, `:env`, …)
  - automatic multi-line input (open brackets, trailing `:`, `\`, `=`, `,` or an operator), `_` holds the last result
  - persistent history, Tab completion of names, fields, `:commands` and paths (pyreadline3 on Windows),
    colours (respects `NO_COLOR`), Ctrl+C cancels the current input instead of quitting, helpful hints after errors
- **Shortcuts**: `apsi` alone starts the REPL (or runs a program piped on stdin), `apsi FILE.apsi ARGS` runs a file,
  `apsi -e "CODE"` evaluates code, `apsi NAME` runs an `apsi.toml` task, mistyped commands get "did you mean"
  - permission flags may come first and never swallow the file: `apsi --allow-python app.apsi`,
    `apsi run --allow-read app.apsi`, `apsi --allow-python -e "CODE"`
- **Projects**: `apsi new NAME --template app|lib|physics|ai|server`, `apsi init`, `apsi.toml`
  (`[project] main`, `[run] profile/allow`, `[tasks]`), `apsi task [NAME]`; `apsi run` / `apsi test` without
  arguments use the project
- **New commands**: `apsi eval`, `apsi watch` (re-run / re-test / re-check on save), `apsi bench`, `apsi doc`,
  `apsi completion bash|zsh|fish|powershell`, `apsi help COMMAND`
- `apsi run`: program arguments (read them with `args()`), `--time`, `--watch`, `-` for stdin;
  `apsi check` and `apsi fmt` accept folders and default to the current folder; coloured error messages
- New built-ins: `args()`, `input(prompt)`, `exit(code)`
- **`apsi build app.apsi`** makes an executable (`app.exe` on Windows, `app` on Linux) containing the program, its
  imports, the parsed program, Apsi as bytecode and the permissions chosen at build time; `--standalone` (Windows)
  adds its own Python runtime with the packages the program uses, `--windowed`, `--add` for data
- **compile cache**: every program file is parsed once per process (previously the Guard scan, the checker and the run
  each parsed it) and kept in a signed on-disk cache (`~/.apsi/cache/compiled`); `apsi compile PATHS` fills it ahead
  of time, `--runtime`, `--stats`, `--clear`, `APSI_NO_CACHE=1`

### Python to Apsi
- **`apsi convert`** translates Python files and whole folders: let/var inference, block-scope hoisting,
  functions, closures, lambdas, classes and dataclasses (with inheritance), generators, comprehensions, f-strings,
  nested and starred unpacking, `with open`, try/except, comments and docstrings
- library mapping for numpy, math, random, json, time, os.path, sys, statistics, PyTorch tensors, `torch.nn`
  layers (→ `use layers`) and `torch.nn.functional`; every other library keeps working through `pyimport`
- every place that is not an exact translation is reported with its line; the result is checked automatically;
  `--run` converts, checks and runs; **Apsi: Convert Python File to Apsi** in VS Code/Cursor

### AI
- **`use layers`**: any architecture as a graph of layers — `sequential`, `residual`, `parallel`, `dag` (any directed
  acyclic graph with several inputs/outputs), dense, convolution, pooling, LSTM/GRU, multi-head and cross attention,
  transformer blocks, normalisation, dropout, 12 activations
- **graph neural networks**: `graph()`, GCN, GraphSAGE, GAT, GIN, graph pooling and differentiable
  `segment_sum/mean/max/softmax` (gradients verified against finite differences)
- model zoo: `mlp`, `resnet_mlp`, `cnn`, `lstm_classifier`, `transformer_lm`, `transformer_classifier`, `autoencoder`,
  `vae`, `gcn_net`, `gat_net`, `gin_net`, `siamese`; `fit` (mini-batches, graph masks, validation), `predict`, `summary`
- **PyTorch backend**: every `use layers` model trains and runs on PyTorch with `fit(…, backend="torch")` — CPU,
  NVIDIA CUDA, AMD ROCm, Apple MPS, Intel XPU; precisions fp64/fp32/tf32 and bf16/fp16 mixed precision (automatic
  loss scaling), `torch.compile`, fused attention, cuDNN LSTMs; `backend="auto"`, `devices()`,
  `torch_module(model, params)` (a real `torch.nn.Module`) and `params_from_torch`; `check_torch()` compares every
  layer's outputs and gradients with NumPy. Needs `--allow-python`. Guide: `docs/BACKENDS.md`
- `apsi pkg install torch-auto` (also used by the `ai` group and the setup wizard) picks the PyTorch build from the
  NVIDIA driver version — CUDA 12.8 / 12.6 / 12.4 / 11.8 — ROCm for AMD on Linux, else CPU;
  explicit builds: `apsi pkg install torch-cuda` (CUDA 12.6) · `torch-cu128` (RTX 50) · `torch-cu124` · `torch-cu118` · `torch` (CPU):
  PyTorch is taken from PyTorch's own index (a CPU-only wheel from PyPI can no longer slip in), an installed build is
  kept, a different build replaces it
- `fit` mini-batches models with several inputs (`dag(inputs=[…])`, `siamese`); new `mean_pool(axis)` layer
- new complete reference `docs/LAYERS.md` (every layer with shapes and parameters, recipes, error messages)

### Physics, mathematics, data
- `sci.solve_ivp`: implicit, L-stable, adaptive TR-BDF2 for stiff systems and adaptive Dormand–Prince RK45, with units
  and dense output (`t_eval`); reproduces the Robertson kinetics reference values
- `fsum` (correctly rounded sums) and Python-compatible built-ins (`any`, `all`, `divmod`, `pow`, `ord`, `chr`, `dict`,
  `set`, `tuple`, `iter`, `next`, `format`, `getattr`, `hasattr`, …)
- data pipelines: `from_jsonl`, `from_parquet`, `interleave`, `window` (sliding windows), `shard`
- new examples `28_any_architecture` and `29_stiff_chemistry`

### Editors
- **one command for every editor**: `apsi editors [list|install|uninstall]`, also run by the setup wizard (component
  *Code editor support*, listed on the *System check* page) and undone by the uninstaller
- VS Code family — **VS Code, Cursor, Windsurf, VSCodium, VS Code Insiders, Trae** — get the same extension (through
  their command-line tool, or unpacked straight into their extension folder when the tool is missing)
- **Sublime Text** package: colours, build system with variants and clickable errors, command palette, comments,
  indentation, Goto Symbol, snippets
- **Neovim / Vim** package: filetype, colours, indentation, `:make`, mappings, help file; the language server starts
  automatically in Neovim
- **Notepad++** user-defined language with Apsi colours
- **JetBrains** TextMate bundle, **Helix**, **Zed** and **Emacs** (`apsi-mode`) configurations
- `tools/build_editors.py` keeps the Sublime/TextMate grammar and snippets identical to the VS Code ones

### Fixes
- `use hub` on the processor: a product could wait forever when several threads used the model and a worker
  thread went to sleep at the same moment (a lost wake-up, seen on ARM64). Publishing a job and counting the
  sleeping workers are now sequentially consistent atomics, and a worker leaves its sleep with an atomic count.

### Python conversion: libraries come along
- `apsi convert` converts the installed libraries a program imports that are written in Python (found in the
  program's `.venv` of any Python version, or in the converting Python) into `apsi_packages/` next to it, with their
  data files and `.dist-info`; the result runs on Apsi alone. Dynamic imports, `importlib.resources` and
  `importlib.metadata` find the converted libraries. Verified with `requests` + `urllib3` + `idna` + `certifi`
  (63 modules) against a local HTTP server with the real libraries blocked. `--no-packages` keeps Python imports.
- Libraries of compiled code (C, C++, Rust) stay Python imports; the notes say so and name Apsi's own tool for the
  job; libraries that are not installed are listed once
- `apsi run` no longer uses a `.venv` next to the program by itself (only `--venv DIR`)
- `from .pkg import submodule` inside a package converts to the submodule (relative level kept)

### VS Code extension
- `apsi run` does not mix in a `.venv` of another Python version or build (for example a free-threaded 3.14 venv
  under Apsi's 3.12): its compiled packages cannot load there ("No module named 'pyarrow.lib'"); it says so and
  how to install the packages for Apsi instead
- Apsi Guard: words in a docstring or long message are checked only when they are paths (a sentence that mentions
  `.env` or credentials is not code that reads them), `tdata` must be a folder name (no match inside `netdata` or
  `metadata`), and a line gives one finding per kind
- `apsi test examples` no longer downloads a model or opens ports: `31_ai_server` and `27_pretrained_models` load
  their models only when run, not under `apsi test` (the Windows build waited for a 1 GB download there)
- **finds the installed Apsi of every layout:** the side-by-side installs (`Programs\Apsi\<version>`, the version
  chosen with `apsi use` first) run through their own Python directly; a launcher found on PATH (`apsi.cmd`) is
  started through the shell, which current VS Code needs on Windows (before, the language server could not start and
  **Convert Python File to Apsi** failed with an empty output)
- **Convert Python File to Apsi** runs in the background, the progress note closes when it ends, and a failure says
  why (the command line and the error are in the Apsi output)
- richer grammar: definitions vs. calls, built-ins, parameters, types and generic types, units in their own colour,
  `as [unit]`, `±`, derivatives, f-string formats, triple-quoted strings, decorators, `TODO`, Apsi code blocks in Markdown
- **Apsi Dark** / **Apsi Light** themes, snippets, smarter indentation and folding
- commands: Run, Run with Arguments, Run Project, Test, Test Project, Check, Watch, Benchmark, Audit, REPL,
  Run Selection in REPL (`Shift+Enter`), Load File into REPL, Run Project Task, New Project, Show Documentation,
  Run Apsi Command, Restart Language Server; ▶ button, status-bar button, right-click menus, key bindings
- task provider for `apsi.toml` tasks and an `$apsi` problem matcher
- the file icon is now the Apsi logo letters only, on a transparent background

### Installers
- **`ApsiSetup-1.10.0-linux.sh`**: one-file installer for every Linux distribution (and WSL), no root, built offline
  by `packaging/make_linux_installer.py`: commands, `.apsi` file type and icons, *Run with Apsi*, templates, editor
  support, `uninstall.sh`; uses the system Python and installs numpy privately only when it is missing
- the wheel is built with the standard library only (`packaging/make_wheel.py`), so release builds work offline;
  the `.deb` now carries the release manifest (`apsi verify` passes)
- Windows: deep recursion works again — the program thread asked for a 256 MiB stack, which Windows refuses (the
  limit is below 256 MiB), so programs fell back to a 1,000-level limit; now 255 MiB with smaller fall-backs
- **Windows 10/11**: `ApsiSetup-1.10.0-x64.exe` contains Python 3.12 and numpy — nothing else to install, no administrator rights
  - adds `apsi` to PATH, creates Start-menu entries and registers `.apsi` files
  - `.apsi` files: logo icon (letters only, transparent, every standard size 16–256), *Run with Apsi*, *Edit*
    (VS Code when installed), *Test with Apsi*, and **New → Apsi source file**; Explorer refreshes the icon immediately
  - uninstall from Settings → Apps
- the setup wizard shows its *Preparing Apsi Setup…* window immediately and keeps it on top of other windows until the
  wizard is ready, then opens the wizard in its place; a balanced 900 × 610 window; software rendering, crash and
  load-failure recovery, and helpers that can never crash Setup (more stable on every GPU driver); English-only locales
  make the package smaller and faster to unpack
- Linux: `.apsi` file icons in every size, *New Document → Apsi file* when `~/Templates` exists, more terminals
  for *Open the REPL*
- **Portable Windows**: `apsi-1.10.0-windows-x64.zip` — unzip and use
- **Linux (Debian/Ubuntu)**: `sudo apt install ./apsi_1.10.0_all.deb`
- **Any system**: `pip install apsi-1.10.0-py3-none-any.whl`

### Security (see docs/SECURITY.md)
- **Apsi Guard** scans programs automatically before they run; `apsi scan` scans on demand (Microsoft Defender/ClamAV with
  `--av`, YARA rules and block lists)
  - detects Trojan Source, look-alike characters, password or key theft, data exfiltration, start-up persistence,
    crypto-miners, ransomware patterns, sandbox escapes and obfuscated code
- **Ransomware shield**: stops a program that rewrites or deletes many existing files in a short time, or touches a decoy file
- **Protected locations**: no program may overwrite `~/.apsi`, the runtime, system folders or start-up files; reading
  password stores needs an explicit, specific permission
- **Network guard**: blocks SSRF (cloud metadata, link-local), checks every redirect and limits response sizes
- `--allow-python` allows ordinary libraries only; subprocess, os and ctypes need `--allow-python=all`
  - `--allow-ffi=lib@sha256` pins native libraries by hash; libraries in temp folders are refused
- the installed runtime runs Python isolated (`-I`)
- **Setup**: checks the SHA-256 of every payload file before installing, validates the target folder, enables Electron fuses,
  checks the sender of every IPC message and whitelists every option

### Setup wizard (Electron + React)
- Express and Custom modes: system check, folder (validated while you type), components, optional packages
  (GPU/ONNX/PyTorch/Transformers), security profile and Apsi Guard, summary, progress with rollback
- maintenance mode: Modify, Repair, Uninstall; `--silent` for unattended installs
- the Apsi logo in every standard size (ICO 16–256, PNG 16–1024) for Setup, NSIS, `.apsi` files, VS Code and the playground

### Package commands
- `apsi pkg list | install | remove | lock | audit` installs library bundles (gpu, onnx, torch, transformers, sci, data, ai, all)
  - binary wheels only, locked with hashes
- `apsi scan`, `apsi audit`, `apsi verify`, `apsi config`
- `setup.ps1` / `setup.sh` install, test and build the installers in one command

### Language and libraries
- `use precision`: FP64, FP32, FP16, BF16, FP8 (E4M3/E5M2), FP6 (E3M2/E2M3), FP4 (E2M1), INT8, INT4, UINT8
  - bit-exact with the standards, bit-packing, per-channel quantization, quantization-aware training and `qmatmul`
- `use data`: streaming pipelines (parallel map, prefetch, shuffle, batch), memory-mapped tensors on SSD, signed disk cache
- `use parallel`: tasks, channels, locks, atomics, parallel_for/reduce, set_threads, bench
- `use web`: HTTP server for many concurrent requests (asyncio + worker pool + multi-process), secure by default
  - JWT, API keys, rate limits, CORS, static files, metrics, TLS
- `use gpu`: `device()`, `to_device()`, `gpu_info()`; autodiff works on CuPy tensors
- `use models`: memory-mapped safetensors read/write, compressed loading (INT8/FP8/FP4), `quantize_model`, `qlinear`,
  ONNX Runtime, Hugging Face, PyTorch
- a line starting with `|>` continues the previous line
- `from_tensors(..., batch=N, shuffle=true)` yields vectorized batches
- new examples: `23_web_api`, `24_low_precision`, `25_data_pipeline`, `26_model_serving`, `27_pretrained_models`

### Documentation
- everything is in English
- [`docs/LANGUAGE.md`](docs/LANGUAGE.md): one complete reference for the syntax, the type system, units, functions,
  structs, modules, every built-in and module member, the REPL, the command line and `apsi.toml`
  (replaces `SPEC.md` and `STDLIB.md`)

## 0.3.0
- script installers for Windows/Linux, CI, language server, VS Code extension, `apsi fmt`, web playground, `use sci`, LSTM/GRU

## 0.2.0
- Apsi → bytecode compiler, higher-order autodiff, `±` uncertainty, `rk45`

## 0.1.0
- the language, units, shape generics, autodiff, `evolve`, nn, permissions, interop
