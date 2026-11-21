## 2.6.2 — large graph and language models on one card, training that resumes exactly and learns from its own output, half the optimizer's memory, your own GPU kernels (21 November 2026)

**Install:** [apsi.dev/download](https://apsi.dev/download) — `ApsiSetup-2.6.2-x64.exe` (Windows 10/11),
`ApsiSetup-2.6.2-linux.sh` (any Linux, WSL), `apsi_2.6.2_all.deb` (Ubuntu / Debian), or from this source:
`pip install numpy`, then `./apsi.sh` / `apsi.bat`. What's new, with examples: [docs/WHATS-NEW.md](https://github.com/ATECTechnologies/apsi/blob/main/docs/WHATS-NEW.md).

A 155 M-parameter model of several graph encoders, a mixture-of-experts router and an 18-layer decoder, with graphs
of 6,144 nodes and sequences of 3,072 tokens, trains on a 12 GB RTX 3060 in 6.3 GB, its decoder also learning from
its own sampled answers during training. Everyday code runs as fast as 2.6.0 (the same programs on one GitHub runner:
every row within run-to-run noise; `apsi check` 4 % faster). Tested: 468 tests on every push (Linux, macOS, Windows;
Python 3.12–3.14). New example: [`examples/47_train_your_llm.apsi`](https://github.com/ATECTechnologies/apsi/blob/main/examples/47_train_your_llm.apsi).

### Training that resumes exactly, and batches made while it trains
- **Exact resume of everything:** a checkpoint now keeps the random generators, the step counts of parameter groups
  that step every batch, and your program's own state: `fit_batches(..., state=record(get=fn() => text,
  set=fn(text) => ...))` (a replay buffer, running averages, a curriculum position). Five steps + resume + five steps
  give the same loss as ten in one go (relative difference below 1e-10, on the card and on the processor). An epoch
  that has nothing left after resuming is no longer reported as a loss of 0.
- **Batches made when training reaches them:** an item `record(later=fn, end_step=false)` in the batch list is
  called at its turn and its batches are trained there; inside it `training_params()` reads the weights of that
  moment and `training_update(params, only=["part.sub"])` writes some back (a part trained by an optimizer of your
  own). Reinforcement learning from the model's own samples, data that depends on the model, curricula.
- **Steps of their own length:** `accumulate=0` ends a step at the batch whose targets say `end_step=true`; each
  batch divides its gradient by its `step_items`. Parameter groups with `every="batch"` take their own AdamW step
  after every batch (`hold=true` keeps the gradient for the next one): a decoder with its own optimizer inside a
  larger model.
- `card_loss(..., scale="field")` (a batch that counts more, less or with a negative sign) and labels `none` (the
  whole loss in `rest`, which may then read the logits).

### Less memory, the same updates
- **`adam(state="bf16")` / `adamw(state="bf16")`:** the optimizer's two moments in bfloat16 with stochastic rounding
  (random low bits added before cutting, so the rounding averages out): 4 bytes per weight less on the card, the
  update computed in float32 as before. Checkpoints keep the 16-bit moments; the processor path rounds the same way.
- A graph batch can say `eval_regions=[...]`: typed graph encoders of those regions run without dropout in that
  batch (a view read as in evaluation).
- `memory_slots` takes a memory made earlier (`graph.mem_given`), `brain_routing` can explore
  (`graph.route_sample`: Gumbel top-k choices kept per row) and gives the REINFORCE gradient of its choices
  (`rl_policy_grad`).

### The language
- **Raw strings** `r#"..."#`: no escapes, several lines, end at `"` followed by the same number of `#`
  (`r##"..."##` when the text holds `"#`). Regular expressions, Windows paths, other languages' source.
- **`pymodule(name, source)`:** Python source kept inside your program becomes a module (`--allow-python`), so a
  program ships as `.apsi` files only. Its code is the program's own: its `_private` names are reachable, a
  library's are not (`--allow-python=all` opens those too).
- `save_safetensors` keeps 0-d tensors as scalars (shape `[]`, as other programs write them).
- New example: `examples/47_train_your_llm.apsi` — a tokenizer, a LLaMA / Qwen3-style decoder, training and text
  written by the model, in 40 lines.

### Models that did not fit now train
- **Activation recompute** (`fit_batches(recompute="auto")`, the default on a card): the first step measures the
  memory; only the blocks whose activations do not fit are computed again in backward (residual / transformer
  blocks, hgt, global attention, graph transformer layers). `checkpoint(layer)` always recomputes. Same weights to
  the last bit as keeping every activation.
- **Global attention in pieces of whole rows** (`APSI_GA_PIECE`): nothing per edge is kept between forward and
  backward, so memory grows with the nodes, not nodes × keys.
- Weights reach the card in pieces; every parameter starts on a 32-byte boundary, so one odd-sized head no longer
  pushes the matrices after it off the fast 16-bit products.

### Your own kernels, your own losses
- **`gpu_layer(forward, backward, init, out)`**: your `@gpu` kernels as a layer of a model the GPU trainer trains,
  on the trainer's own buffers.
- **`card_loss(output, rest, labels, weights=)`**: a model with several outputs; the language model's cross-entropy
  stays on the card, the other outputs go to `rest` on the processor. `weights=` gives each label its own weight
  (`sum(w ce) / sum(w)`), exactly on the card.
- **Parameter groups:** `fit_batches(groups=[record(params=[...], lr=, b1=, b2=, eps=, weight_decay=, clip=,
  decay="matrices"|"all"|"none"|[parts], no_decay=[...]), ...])` — each part of a model with its own AdamW settings
  and its own gradient clipping, scaled by the schedule; the same update on the card and on the processor.
- **`grad_scale(k)`**: a chosen share of the gradient flows back (0 = stop, 0.08 = a soft bridge).

### Graph models of several encoders
- `typed_graph_encoder`: `scope="all"` (global attention, graph context and final norm over every node),
  `out_bias="type"`, `act=` (also the exact `gelu_exact` on the card), `dropout=`, `region=` (several encoders over
  one batch graph, each over its own region of nodes). `global_attention` `dropout=` / `region=`; `typed_pool`
  `others=`.
- New layers: `node_text_encoder` (token and hash means, attention pooling with positions, language / family / type /
  structure embeddings, plain-text rows), `memory_slots` (a decoder memory from the best nodes of two node sets, two
  summaries and a null slot), `brain_routing` (one mixture of experts with running statistics, balancing bias and
  communication between the chosen experts, and a fusion bridge with an anti-forgetting penalty), `mask_rows`,
  `gated_delta`; `pick_rows` with nothing picked gives zeros as on NumPy. `crypto.crc32`, `hash_int`.
- Stateful layers replay their running statistics in backward, and `model_state` / `set_model_state` save them;
  `fit_batches` checkpoints carry them.

### Model files of other programs
- **`models.named_tensors(params, rows, shapes)` / `from_named_tensors(like, named, rows)`**: a parameter tree under
  another program's tensor names (fused weights, transposes, scalars), so either program reads the other's
  checkpoints; `load_named_tensors(path)` reads a `.safetensors` file under its exact names and
  `safetensors_metadata(path)` its text metadata (a tokenizer, a configuration).
- GPU checkpoints record where each weight sits and refuse a checkpoint of another layout.

### Fixes
- The bfloat16 copy of a long array (the whole weight array, ~155 M values) is made in launches the kernels can
  cover; before, the tail stayed stale and the 16-bit products were refused by the self-check.
- `layers.fit_batches` passes `exact_sums=` to the GPU trainer; `predict` on the card takes several inputs,
  several outputs and the batch's graph.
