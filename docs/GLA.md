# GLA in Apsi

GLA 3.0 — the code-and-language model of `GLA.py` / `gla_brain_dual.py` — is a model of `use layers`. It trains with
`fit_batches` on NumPy, on Apsi's own GPU trainer, and split across as many cards as there are, with the same
training everywhere (the tests check it step by step).

```apsi
use layers

let model = gla("tiny")                        # "21b" (the GLA 3.0 release), "3b", "small", "tiny"
let params = model.init(seed=0)
print(gla_parts(model, params))                # every part's parameters: exactly gla_count(model.config)

let train = fn(epoch) => [gla_batch(gla_task("tiny", 8, seed=1000 * epoch + s)) for s in 0..10]
let r = fit_batches(model, params, train, gla_loss, opt=adam(lr=0.01), epochs=60)
print(gla_accuracy(model, r.params, gla_batch(gla_task("tiny", 64, seed=7))))
```

The full program is `examples/45_gla.apsi`.

## What the model is

| Part | In Apsi | GLA |
|---|---|---|
| node encoder | `node_encoder(d)`: hashed text, sub-tokens, node type, language slot, family, flags and five structural codes (depth, sibling, in / out degree, line) | `TokenEfficientEncoder` |
| Brain A (code), B (editing), C (text) | `hgt_brain(d, heads, layers, types, relations, global_every, max_keys, ratio)`: a typed input projection, `hgt` layers, `global_attention` with learned keys after every `global_every`-th layer, the graph context after every layer, a final norm | `HGTBrain` |
| pooling, heads | `typed_pool(types, init)` (a learned weight per node type), a semantic head, validity and issue heads per brain | `_pool`, `semantic_head`, `task_heads` |
| bridge | `gated_fusion(d)` | `AntiForgetBridge` |
| router | `expert_router(d, experts, top_k, tasks)`: FlyPrompt, the chosen experts attend to each other | `FlyPromptRouter`, `ExpertCommsBlock` |
| graph memory | `brain_memory(d_brain, d, k_a, k_c, types_a, types_c)`: the best-scored nodes of Brain A and Brain C, both summaries, a null slot | `GraphMemory` |
| writer | `embedding`, `multi_head_attention(…, causal, kv_heads, rope, qk_norm, bias=false)`, gated cross-attention into the memory (`gate=0.25, memory_mask=true`), `swiglu`, `rms_norm_layer`, `tied_output` | `Writer` |
| parts trained elsewhere | `param_bank` (PPO policy heads, math core): held so a checkpoint has every weight | `CodePolicyHead`, `MathPhysicsAdapter`, `MathPhysicsHead` |

`gla_count(cfg)` is GLA's own formula (`estimate_model_params`); the built model has exactly that many parameters,
part by part (`gla_parts`). GLA 3.0 (`gla("21b")`): 20,480,386,836.

## One graph per batch

GLA keeps a table per node type and a list of edges per relation. In Apsi a batch is one graph: `gla_batch(samples)`
puts the code graph and the text graph of every sample into it, `node_types` say what each node is (GLA's 15 node
types), the edge `types` are GLA's relations (every one also backwards, `rev_…`), and `batch` says which sample a node
belongs to. Each brain *owns* its node types and relations — `types=[…]` and `relations=[…]` on `hgt`,
`typed_linear`, `typed_norm`, `global_attention`, `hgt_brain`, `typed_pool` — and the nodes of other types pass
through it unchanged, so three brains over one graph compute what GLA's per-type dictionaries compute.

A sample (`gla_sample`) is a code graph and a text graph — each a list of `gla_node(type, token, depth, sibling,
line, …)` and edges `(from, relation, to)` — the answer to write (token ids), and the validity and issue labels.

## The loss

`gla_loss(outputs, targets)` is GLA 3.0's: the writer's next-token loss on the answer, `0.3` validity (BCE),
`0.3` issues (multi-label BCE over syntax, security, safety, undefined_var, unused_import, logic, other), `0.2`
alignment of the code vector with Brain C's summary (InfoNCE, both directions), the router's balance (load, z-loss,
importance) and `0.1` the value head. The model has seven outputs and the loss reads them all: a `dag` with several
outputs trains with a loss of your own on every backend, the targets may be a record.

## Many cards

```apsi
let r = fit_batches(gla("21b"), params, batches, gla_loss, device="cuda:all", accumulate=8)
```

The model is cut into stages between its nodes where the work balances and every stage's weights, gradients and
optimizer state fit its card; the values a later stage needs travel to it by name (several at once: the brains'
node states, their summaries, the memory, the outputs the loss reads). The writer's tied embedding may land on two
cards: both hold the table and their gradients are summed every step, so it stays one weight. `stages=[…]` sets
the cut yourself. Training the 21b model needs about 16 bytes a parameter (fp32 weights, gradients, Adam):
about 330 GB over the cards.

## On the card

The brains run on the GPU trainer's own kernels: typed products (rows grouped by type, one product per type),
the relations' key transforms in one batched product, one softmax over each node's in-edges, global attention with
learned keys (the scores made on the card; the choice above `max_keys` keys on the processor), the graph context
(the means on the card, the centring of graphs x d numbers on the processor), typed pooling likewise. The writer's
self-attention, gated and masked cross-attention, SwiGLU and tied output are kernels too. The node encoder, the
router, the bridge and the memory run on the processor inside the model on the card (their inputs are one row per
sample or the nodes' ids); see `docs/GPU.md`.
