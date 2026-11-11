# `use layers` — reference

`use layers` builds neural networks of **any architecture** out of small, composable layers: plain stacks,
residual networks, any directed acyclic graph (several inputs and outputs, shared weights), convolutional and
recurrent networks, transformers and graph neural networks. Models are ordinary Apsi values, so `grad`,
`value_and_grad`, the optimizers of `use nn`, `save_params` / `load_params` and `fit` all work on them. Every
model runs on NumPy (exact float64, always available) or on PyTorch (CPU, CUDA, ROCm, Apple MPS, Intel XPU) —
see [`BACKENDS.md`](BACKENDS.md).

- [1. The model contract](#1-the-model-contract)
- [2. Parameters](#2-parameters)
- [3. Layers](#3-layers)
- [4. Composition: sequential, residual, parallel, dag](#4-composition)
- [5. Graph neural networks](#5-graph-neural-networks)
- [6. The model zoo](#6-the-model-zoo)
- [7. Training: fit, predict](#7-training)
- [8. Inspection, saving and loading](#8-inspection-saving-and-loading)
- [9. Writing your own layer](#9-writing-your-own-layer)
- [10. Recipes](#10-recipes)
- [11. Errors and how to fix them](#11-errors-and-how-to-fix-them)

---

## 1. The model contract

Every layer and every model is a value with two operations:

| Operation | Meaning |
|---|---|
| `model.init(seed=0)` | creates the parameters — a tree of records and lists of tensors. The same seed gives the same parameters on every machine. |
| `model(params, x, train=false, graph=…, mask=…)` | applies the model. `train=true` turns on dropout and the VAE's random sampling. `graph` is needed by graph layers, `mask` is an additive attention mask. |

```apsi
use layers
use nn

let model = sequential([linear(4, 32), activation("gelu"), linear(32, 3)])
var params = model.init(seed=0)
let logits = model(params, randn(8, 4))            # shape (8, 3)
```

Because the parameters are kept outside the model, a model value can be shared freely (between threads, between
training runs, between a server and a trainer) and several parameter sets can be used with the same model.

## 2. Parameters

`init` returns a tree whose leaves are tensors:

- a layer with weights returns a `record`, e.g. `linear(4, 3).init()` → `record(W=Tensor[4, 3], b=Tensor[3])`
- `sequential` and `parallel` return a **list** with one entry per layer
- `dag` returns a **record** with one field per node name
- layers without weights (activations, pooling, dropout, …) return an empty `record()`

Read and change parameters like any other value: `params[0].W`, `params.encoder[2].b`. Every function that works
on trees — `tree_map`, `grad`, optimizers, `count_params`, `save_params` — works on these.

Initialisation: `linear` uses He initialisation (`init="glorot"` for Glorot/Xavier), convolutions He, attention and
GNN weights Glorot, embeddings N(0, 0.02²), LSTM/GRU uniform ±1/√hidden with the LSTM forget-gate bias set to 1.

## 3. Layers

Shapes: `B` batch, `T` sequence length, `D` features, `C` channels, `H × W` image size, `N` graph nodes.

### Dense and embeddings

| Constructor | Input → output | Parameters |
|---|---|---|
| `linear(n_in, n_out, bias=true, init="he")` | `[…, n_in]` → `[…, n_out]` | `W[n_in, n_out]`, `b[n_out]` |
| `dense(n_in, n_out, act=none)` | linear followed by an activation | as `linear` |
| `bilinear(n1, n2, n_out)` | `[a: [B, n1], b: [B, n2]]` → `[B, n_out]`, `aᵀ W_k b + b_k` | `W[n_out, n1, n2]`, `b` |
| `embedding(vocab, d)` | integer ids `[B, T]` → `[B, T, d]` | `E[vocab, d]` |
| `positional_encoding(max_len, d, learned=false)` | `[B, T, d]` → same, adds position information (sinusoidal, or a learned table) | `P[max_len, d]` when learned |

### Convolution and pooling (images `[B, C, H, W]`)

| Constructor | Meaning |
|---|---|
| `conv2d_layer(c_in, c_out, k=3, stride=1, padding="same")` | 2-D convolution (cross-correlation, as in PyTorch); `padding` is `"same"` (k // 2) or a number. Parameters `W[c_out, c_in, k, k]`, `b[c_out]`. |
| `max_pool(k=2)`, `avg_pool(k=2)` | k × k pooling with stride k (an incomplete border is dropped) |
| `global_avg_pool()` | `[B, C, H, W]` → `[B, C]` |
| `flatten_layer()` | `[B, …]` → `[B, everything else]` |
| `reshape_layer(d1, d2, …)` | `[B, …]` → `[B, d1, d2, …]` |

### Sequences

| Constructor | Meaning |
|---|---|
| `lstm_layer(n_in, hidden, return_sequences=true)` | `[B, T, n_in]` → `[B, T, hidden]` (or `[B, hidden]`, the last step, when `return_sequences=false`). Gates i, f, g, o in one fused weight `W[n_in + hidden, 4·hidden]`. |
| `gru_layer(n_in, hidden, return_sequences=true)` | GRU with weights `Wzr`, `bzr`, `Wh`, `bh` |
| `last_step()` | `[B, T, D]` → `[B, D]` |
| `mean_pool(axis=1)` | mean over an axis, e.g. over time `[B, T, D]` → `[B, D]` |

### Attention and transformers

| Constructor | Meaning |
|---|---|
| `multi_head_attention(d, heads=4, causal=false)` | self-attention on `x: [B, T, d]`, or cross-attention on `[query, memory]`. `causal=true` hides future positions; `model(p, x, mask=M)` adds any additive mask `M[T, S]` (use `-1e9` to hide). Parameters `Wq, Wk, Wv, Wo[d, d]`, `bo[d]`. |
| `transformer_block(d, heads=4, ff=4·d, causal=false, dropout=0.0, activation="gelu")` | pre-norm block: `x + attention(norm(x))`, then `x + MLP(norm(x))` |
| `transformer_encoder(d, heads=4, layers=2, ff, causal, dropout)` | `layers` blocks followed by a final layer norm |

### Normalisation, regularisation, scaling

| Constructor | Meaning |
|---|---|
| `layer_norm_layer(d, eps=1e-5)` | normalises the last axis; `gamma`, `beta` |
| `rms_norm_layer(d, eps=1e-6)` | `x / rms(x) · gamma` (as in LLaMA-style models) |
| `batch_norm_layer(d, eps=1e-5)` | normalises over the batch (and spatial) axes with the current batch's statistics; channel axis 1 |
| `dropout_layer(p=0.1)` | active only with `train=true`; keeps the expected value (inverted dropout) |
| `scale(d, shift=true)` | learnable per-feature `x · s + t` |

### Activations

`activation(name)` with `name` one of `relu`, `gelu` (tanh approximation), `silu` / `swish`, `mish`, `elu`,
`leaky_relu` (slope 0.01), `tanh`, `sigmoid`, `softplus`, `softmax` (last axis), `linear` / `identity` — or
`activation(any_function)`, e.g. `activation(fn(x) => x * sigmoid(1.702 * x))`. Inside `sequential([...])` a name can be
written directly: `sequential([linear(2, 8), "relu", linear(8, 1)])`.

### Small helpers

| Constructor | Meaning |
|---|---|
| `identity()` | returns its input |
| `select(i)` | element `i` of a list input (e.g. one branch of `parallel(..., combine="list")`) |
| `lambda_layer(fn, name="lambda")` | any function `fn(x)` as a layer without parameters |
| `add()`, `multiply()`, `concatenate()` | combine the list of inputs of a `dag` node |

## 4. Composition

### `sequential(layers)`

Applies the layers in order. Parameters: a list.

```apsi
let net = sequential([conv2d_layer(1, 16), "relu", max_pool(2), flatten_layer(), linear(16 * 14 * 14, 10)])
```

### `residual(inner, pre_norm=none)`

`x + inner(x)`; with `pre_norm=layer_norm_layer(d)` it is `x + inner(norm(x))`. Parameters `record(inner, norm)`.

### `parallel(branches, combine="concat")`

Runs every branch on the same input and combines the results: `"concat"` (last axis), `"add"`, `"mul"`, `"mean"`
or `"list"` (keep them separate). Parameters: a list, one entry per branch.

### `dag(nodes, inputs=["x"], output=…)` — any directed acyclic graph

Each node is `[name, layer, input]` or `[name, layer, [input1, input2, …]]`; a node with several inputs
receives them as a list. Nodes may only use inputs or nodes defined before them, and every name is unique (both are
checked when the model is created, with a clear message). `output` is a name or a list of names — the model then
returns a list.

```apsi
# two inputs, a shared trunk, two heads
let net = dag([
    ["img",   cnn(1, 64, image=28), "image"],
    ["tab",   mlp([12, 64, 64]), "table"],
    ["both",  concatenate(), ["img", "tab"]],
    ["trunk", mlp([128, 128, 64]), "both"],
    ["cls",   linear(64, 10), "trunk"],
    ["reg",   linear(64, 1), "trunk"],
], inputs=["image", "table"], output=["cls", "reg"])

let p = net.init(seed=0)
let outs = net(p, [images, tables])          # [class logits, regression output]
```

Weight sharing: `siamese(encoder, combine, head)` (section 6) applies one encoder — one parameter set — to two
inputs `[a, b]`; use it as a node of a larger `dag` when you need shared weights inside your own graph.

## 5. Graph neural networks

### The graph

```apsi
let g = graph(edges=[[0, 1], [1, 2], [2, 0]], n=3, undirected=true)      # or graph(src=…, dst=…, n=…)
```

| Option | Meaning |
|---|---|
| `edges` / `src`, `dst` | directed edges `from → to` |
| `n` | number of nodes (default: largest id + 1) |
| `undirected=true` | adds every edge in both directions |
| `self_loops=true` | adds `i → i` for every node (the default; GCN/GAT expect it) |
| `batch=[…]` | the graph id of every node, for graph-level tasks with many small graphs in one batch |

A graph has `.n`, `.edges`, `.src`, `.dst`, `.in_degree`, `.out_degree` and `.adjacency()`.

### Graph layers (node features `[N, D]`; apply with `model(p, x, graph=g)`)

| Constructor | Update |
|---|---|
| `gcn(n_in, n_out, bias=true)` | `h_i = Σ_j x_j W / √(deg_i · deg_j)` (Kipf & Welling) |
| `sage(n_in, n_out, aggregate="mean")` | `h_i = x_i W_self + agg_{j→i} x_j W_neigh` with `mean`, `sum` or `max` |
| `gat(n_in, n_out, heads=4, concat=true, slope=0.2)` | graph attention; heads concatenated (`[N, heads·n_out]`) or averaged |
| `gin(n_in, n_out, hidden=n_out)` | `MLP((1 + ε) x_i + Σ_j x_j)` with a learnable ε |
| `graph_pool(how="mean")` | `[N, D]` → `[graphs, D]` using `batch` (`mean`, `sum`, `max`) |

### Differentiable graph operations

`segment_sum(data, ids, n)`, `segment_mean`, `segment_max`, `segment_softmax(scores, ids, n)` (softmax within each
segment, e.g. over the incoming edges of every node), `segment_count(ids, n)`, `gather(x, ids)`. With these you can
write any message-passing layer yourself (see section 9).

### Node classification with a training mask

```apsi
let r = fit(gat_net(8, 16, classes, heads=4), …, features, labels, loss="cross_entropy",
            graph=g, mask=train_mask, val=…, epochs=200)
```

The whole graph is used in every step; only nodes where `mask` is true count in the loss.

## 6. The model zoo

| Model | Input → output |
|---|---|
| `mlp(sizes, act="relu", dropout=0.0, norm=false, out_act=none)` | e.g. `mlp([2, 64, 64, 3])` |
| `resnet_mlp(d_in, d_out, width=64, blocks=3, act="gelu")` | deep MLP with pre-norm residual blocks — trains stably at any depth |
| `cnn(in_channels, classes, image=28, channels=[16, 32], dense_width=64)` | `[B, C, image, image]` → `[B, classes]` |
| `lstm_classifier(n_in, hidden, classes, layers=1)` | `[B, T, n_in]` → `[B, classes]` |
| `transformer_lm(vocab, d=64, heads=4, layers=2, context=128, dropout=0.0)` | GPT-style: token ids `[B, T]` → next-token logits `[B, T, vocab]` |
| `transformer_classifier(vocab, classes, d=64, heads=4, layers=2, context=128)` | token ids `[B, T]` → `[B, classes]` |
| `autoencoder(d_in, latent, hidden=[64], act="relu")` | returns `[reconstruction, code]` |
| `vae(d_in, latent, hidden=[64], act="relu")` | returns `[reconstruction, mu, logvar]`; train with `loss="vae"` (or `vae_loss(out, x, beta)`) |
| `gcn_net(n_in, hidden, classes, layers=2, dropout=0.5)` | node classification |
| `gat_net(n_in, hidden, classes, heads=4, dropout=0.3)` | node classification with attention |
| `gin_net(n_in, hidden, classes, layers=3, pool="sum")` | graph classification (uses the graph's `batch`) |
| `siamese(encoder, combine="concat", head=none)` | one encoder with shared weights applied to inputs `[a, b]` |

Every zoo model is an ordinary composition of the layers above, so `summary(model)` shows exactly what is inside and
you can copy and change it.

## 7. Training

### `fit`

```apsi
let r = fit(model, params, X, y, loss="cross_entropy", opt=adam(lr=1e-3), epochs=20, batch_size=64)
print(r.history[-1])            # loss of the last epoch
let trained = r.params
```

| Argument | Default | Meaning |
|---|---|---|
| `model`, `params` | | the model and its starting parameters |
| `X`, `y` | | inputs and targets (a list of inputs for models with several inputs). Without `y` the target is `X` (autoencoders, VAEs). |
| `loss` | `"mse"` | `"mse"`, `"mae"`, `"cross_entropy"` (integer labels), `"bce"` (logits), `"vae"`, `"huber"` (torch backend), or a function `loss(prediction, target)` |
| `opt` | `adam(1e-3)` | any optimizer of `use nn`: `sgd(lr, momentum)`, `adam(lr, b1, b2, eps)`, `adamw(lr, …, weight_decay)` |
| `epochs` | `10` | passes over the data |
| `batch_size` | all | mini-batch size; batches are reshuffled every epoch (reproducibly from `seed`) |
| `seed` | `0` | shuffling, dropout and sampling |
| `clip` | none | clip the global gradient norm to this value |
| `schedule` | none | a function `step → learning rate`, e.g. `fn(s) => cosine_lr(s, 1000, base=3e-4, warmup=100)` |
| `graph`, `mask` | none | graph tasks (section 5) |
| `val` | none | `[X_val, y_val]`; the validation loss is recorded every epoch |
| `verbose` | `true` | print the loss about ten times during training |
| `workers` | `1` | data-parallel training on several processes (Linux, macOS): every batch is split into shards, their gradients are averaged by shard size — the same step as one process — then one optimizer step is taken. `"auto"` uses every core. It pays off when the model's matrix products are too small to keep all cores busy on their own |
| `backend` | `"numpy"` | `"numpy"`, `"torch"` or `"auto"` — see [`BACKENDS.md`](BACKENDS.md) |
| `device`, `compile` | `"auto"`, `false` | torch backend only |
| `precision` | `"auto"` | torch backend: bf16 mixed precision on GPUs that have bf16 (NVIDIA RTX 30xx and newer, AMD MI200+, Intel XPU), fp16 with loss scaling on older GPUs and Apple MPS, fp32 on the CPU. Choose yourself with `"fp32"`, `"tf32"`, `"bf16"`, `"fp16"`, `"fp64"`, or any pair of torch dtypes as `{"store": "float32", "compute": "bfloat16"}`; `APSI_PRECISION=fp32` sets it for every program. The numpy backend trains in fp32 |

`fit` returns `record(params, history, val_history, device)` — `device` is `"numpy"` or the torch device used.

### Your own training loop

`fit` is a convenience. Any loop works, because the model is a function of its parameters:

```apsi
let opt = adamw(lr=3e-4, weight_decay=0.01)
var p = model.init(seed=0)
for step in 0..1000:
    let xb, yb = next_batch()
    let l, g = value_and_grad(fn(q) => cross_entropy(model(q, xb, train=true), yb))(p)
    p = opt.step(p, clip_grad_norm(g, 1.0))
```

### `predict`

`predict(model, params, X, graph=…, backend="numpy")` applies the model with `train=false`.

## 8. Inspection, saving and loading

- `summary(model, params)` prints a table of every layer and its parameter count, and returns it as a string
- `count_params(params)` — the number of trainable numbers
- `save_params("model.npz", params)` / `load_params("model.npz", model.init())` — portable NumPy archives
- `quantize_model(params, int8)` and `qlinear` shrink a trained model (see [`AI.md`](AI.md#3-use-less-ram--vram))
- on the torch backend, `torch_module(model, params)` gives a real `torch.nn.Module` (`state_dict`, ONNX export, …)

## 9. Writing your own layer

Any function `fn(params, x)` already is a model you can differentiate. To make it a reusable layer with its own
parameters, put it in a `dag`/`sequential` through `lambda_layer` (no parameters), or keep the parameters next to
the model in a record:

```apsi
use layers
use nn

# a gated message-passing layer written with the segment operations
fn gated_mp(p, x, g):
    let msg = gather(x, g.src) @ p.W
    let gate = sigmoid(gather(x, g.dst) @ p.G)
    return x + segment_mean(msg * gate, g.dst, g.n)

let p = record(W=randn(8, 8, seed=0) * 0.1, G=randn(8, 8, seed=1) * 0.1)
let l, grads = value_and_grad(fn(q) => mean(gated_mp(q, features, g) ** 2))(p)
```

Mix both styles freely: `record(backbone=net.init(seed=0), head=dense(64, 1))` is a valid parameter tree, and
`fn model(p, x) = net(p.backbone, x) @ p.head.W + p.head.b` is a valid model.

## 10. Recipes

**Image classifier**

```apsi
let net = cnn(1, 10, image=28, channels=[32, 64], dense_width=128)
let r = fit(net, net.init(seed=0), images, labels, loss="cross_entropy", opt=adam(lr=1e-3), epochs=5, batch_size=128)
```

**Character-level GPT**

```apsi
let lm = transformer_lm(vocab_size, d=128, heads=4, layers=4, context=128, dropout=0.1)
let r = fit(lm, lm.init(seed=0), tokens[:, 0:-1], tokens[:, 1:], loss="cross_entropy", epochs=10, batch_size=32,
            schedule=fn(s) => cosine_lr(s, 5000, base=3e-4, warmup=200), clip=1.0)
```

**Sequence regression with an LSTM**

```apsi
let net = sequential([lstm_layer(3, 64), lstm_layer(64, 64, return_sequences=false), linear(64, 1)])
```

**Wide & deep**

```apsi
let net = dag([["wide", linear(20, 1), "x"], ["deep", mlp([20, 128, 64, 1]), "x"], ["out", add(), ["wide", "deep"]]])
```

**Variational autoencoder**

```apsi
let v = vae(784, 16, hidden=[256])
let r = fit(v, v.init(seed=0), images_flat, loss="vae", epochs=20, batch_size=128)
let recon = predict(v, r.params, images_flat)[0]
```

**Graph classification** — many small graphs in one batch:

```apsi
let g = graph(src=all_src, dst=all_dst, n=total_nodes, batch=graph_id_of_node)
let net = gin_net(n_features, 64, classes, layers=3)
let r = fit(net, net.init(seed=0), node_features, graph_labels, loss="cross_entropy", graph=g, epochs=100)
```

## 11. Errors and how to fix them

| Message | Cause and fix |
|---|---|
| `sequential: 3 layers but 2 parameter entries` | the parameters belong to another model — create them with `model.init()` |
| `dag(): node 'h2' uses 'h3', which is not defined before it` | order the nodes so every input comes first |
| `graph layers need the graph: model(params, X, graph=g)` | pass `graph=g` to the model, `fit` and `predict` |
| `multi_head_attention: d=30 must be divisible by heads=4` | choose `d` as a multiple of `heads` |
| `unknown activation 'reLU'` | names are lower case; the message lists every valid name |
| `fit(): unknown loss` | use one of the listed names or pass a function |
| `fit(backend="torch") needs permission 'python'` | the torch backend runs PyTorch: `apsi run --allow-python app.apsi` |
| `backend="torch" needs PyTorch` | `apsi pkg install torch-cuda` (NVIDIA) or `apsi pkg install torch` (CPU) — see [`BACKENDS.md`](BACKENDS.md#1-setup) |
