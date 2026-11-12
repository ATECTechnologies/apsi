# Graphs that reach far

Graph neural networks pass messages along edges: one layer moves information one hop. Two problems follow, and
both are measurable:

* **Values do not travel far.** In a post-norm network that also counts each node as its own neighbour, every
  layer dilutes what arrived from the neighbours; in practice values make it about three hops, however many
  layers there are.
* **Everything is squeezed into one vector.** A node that has heard from eight leaves holds one summary of them
  and cannot say which leaf carried which value. A mean over the whole graph has the same problem.

`use layers` has the tools large graph models use against both:

| Tool | What it does | Reach |
|---|---|---|
| `graph_transformer(d, heads, relations)` | pre-norm residual block: attention over the neighbours only, separate key/value weights per relation, a learned bias per relation and per hop distance, then an MLP | one hop per layer, kept over many layers |
| `global_attention(d, heads, max_keys=4096)` | every node attends to every node of its own graph by content; graphs of a batch never see each other; above `max_keys` nodes the keys are an evenly strided subset | **any distance in one layer** |
| `k_hop(g, k)` | adds a shortcut edge to everything within `k` hops (`edge.hop` = distance) | k hops per layer |
| `mix_hop(n_in, n_out, hops=[0, 1, 2])` | several distances in parallel in one layer: `[x, Âx, Â²x, …]`, each with its own weights | parallel hops |
| `graph_memory(d, k)` | the `k` most salient node states of every graph, weighted by a learned score: `[graphs, k, d]` for a decoder to attend into | the whole graph, node by node |
| `graph_transformer_net(n_in, d, out, layers, heads, relations, global_every, task)` | a complete model: input projection, local blocks with a global block every `global_every` layers, final norm, node or graph output | all of the above |
| `hop_distances(g, node)`, `eccentricity(g)` | breadth-first hop counts; the longest shortest path | measuring how far information must go |
| `hgt(d, heads, types, relations)`, `typed_linear`, `typed_norm` | heterogeneous graphs (`graph(…, node_types=)`): weights per node type, keys through each relation; `types` / `relations` as counts or lists of the ids a layer owns (the rest pass through) | one hop per layer |
| `global_attention(…, ratio=0.5, learned_keys=true, bias=true, zero_init=true, types=[…])` | GLA's form: half the heads, values weighted by a learned score that also picks the keys above `max_keys`, starts as a no-op | any distance in one layer |
| `hgt_brain(d, heads, layers, types, relations, global_every, max_keys, ratio)`, `typed_pool(types, init)` | one of GLA's brains (see `docs/GLA.md`) and its pooling | all of the above |

## Relations and both directions

```apsi
let g = graph(edges=edges, n=n, types=kinds, reverse=true, batch=graph_of_node)
```

`types` gives every edge a relation (`0`, `1`, …), for example syntax, data flow and calls in a program graph.
`reverse=true` adds every edge backwards as a relation of its own, so information flows both ways with separate
weights (from a use back to its definition, from a callee back to its caller). `graph_transformer(…, relations=
g.relations)` learns one key and value projection per relation.

## A task that needs reach

Node 0 of each chain must report a class written at the other end, five hops away (the full program is
`examples/42_long_range_graphs.apsi`):

```apsi
let model = graph_transformer_net(4, 16, 2, layers=1, heads=2, relations=2, global_every=1)
let r = fit(model, model.init(seed=0), X, y, loss="cross_entropy", opt=adam(lr=0.01), epochs=150,
            graph=g, mask=mask, verbose=false)
```

Measured by the test suite (`tests/test_graphs.py`):

| Model | Accuracy at node 0 |
|---|---:|
| 1 `graph_transformer` + 1 `global_attention` | 100 % (one global step reaches the far end) |
| 6 `graph_transformer` blocks, no global attention | 100 % (values survive six layers) |
| 2 `graph_transformer` blocks | at chance (the far end is out of reach) |

## Cost

* `graph_transformer`: proportional to the number of edges, like `gat`.
* `global_attention`: nodes × keys per graph (keys = nodes up to `max_keys`, then `max_keys`). For large graphs
  put a global block after every second local block, as `global_every=2` does.
* `k_hop`: the number of edges grows with the k-hop neighbourhood; use it for sparse graphs and small `k`.
* All of these train with `fit(…)` and Apsi's own autodiff; `fit(…, workers=N)` spreads mini-batches over cores.
