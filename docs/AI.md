# Building AI with Apsi

Apsi does not force a ready-made architecture on you. Every network is an ordinary function that takes a
"parameter tree" (record / list / struct), and `grad` differentiates with respect to every parameter in it.
Whatever architecture you invent, you can write it down directly.


## 1. Build any architecture with `use layers`

Every layer is a pair of pure functions — `init(seed)` creates its parameters, `layer(params, x)` applies it — and
layers compose into graphs. Models are ordinary values: `grad`, optimizers, `save_params` and `fit` work on them.
This section is an overview; **[`LAYERS.md`](LAYERS.md) is the complete reference** (every layer with its shapes and
parameters, error messages, recipes).

```apsi
use layers
use nn

let model = sequential([linear(2, 64), activation("gelu"), dropout_layer(0.1), linear(64, 3)])
var params = model.init(seed=0)
let logits = model(params, X)                       # model(params, X, train=true) turns dropout on
let r = fit(model, params, X, y, loss="cross_entropy", opt=adam(lr=0.01), epochs=50, batch_size=64)
summary(model, r.params)                            # layer table with parameter counts
```

| Kind | Layers |
|---|---|
| dense | `linear(in, out)`, `dense(in, out, act)`, `bilinear`, `embedding(vocab, d)`, `positional_encoding(len, d, learned)` |
| convolution | `conv2d_layer(c_in, c_out, k, stride, padding="same")`, `max_pool(k)`, `avg_pool(k)`, `global_avg_pool()`, `flatten_layer()` |
| recurrent | `lstm_layer(in, hidden, return_sequences)`, `gru_layer(…)`, `last_step()` |
| attention | `multi_head_attention(d, heads, causal)` (self- or cross-attention), `transformer_block(d, heads, ff, causal)`, `transformer_encoder(d, heads, layers)` |
| **graph neural networks** | `gcn(in, out)`, `sage(in, out, aggregate)`, `gat(in, out, heads)`, `gin(in, out)`, `graph_pool("mean"\|"sum"\|"max")` + the graph `graph(edges=…, n=…, undirected, batch)` and the differentiable ops `segment_sum/mean/max/softmax`, `gather` |
| normalisation | `layer_norm_layer(d)`, `batch_norm_layer(d)`, `rms_norm_layer(d)`, `dropout_layer(p)`, `scale(d)` |
| activations | `activation("relu" \| "gelu" \| "silu" \| "swish" \| "mish" \| "elu" \| "leaky_relu" \| "tanh" \| "sigmoid" \| "softplus" \| "softmax")` or `activation(any_function)` |
| composition | `sequential([...])`, `residual(inner, pre_norm)`, `parallel(branches, combine="concat"\|"add"\|"mul"\|"mean"\|"list")`, **`dag(nodes, inputs, output)`** — any directed acyclic graph, several inputs and outputs, `add()`, `multiply()`, `concatenate()`, `lambda_layer(fn)`, `select(i)`, `identity()`, `reshape_layer(...)` |

**Any graph of layers** — each node is `[name, layer, input(s)]`:

```apsi
let net = dag([
    ["wide", linear(2, 3), "x"],
    ["deep", mlp([2, 32, 32, 3]), "x"],
    ["both", add(), ["wide", "deep"]],
], inputs=["x"], output="both")
```

**Graph neural networks** — pass the graph when applying or fitting; `mask` limits the loss to labelled nodes:

```apsi
let g = graph(edges=[[0, 1], [1, 2], [2, 0]], n=3, undirected=true)
let gnn = gat_net(8, 16, classes, heads=4)
let r = fit(gnn, gnn.init(seed=0), features, labels, loss="cross_entropy", graph=g, mask=train_mask, epochs=200)
```

**Model zoo** — `mlp(sizes)`, `resnet_mlp(in, out, width, blocks)`, `cnn(channels, classes, image)`,
`lstm_classifier(in, hidden, classes)`, `transformer_lm(vocab, d, heads, layers, context)` (GPT-style),
`transformer_classifier(vocab, classes, …)`, `autoencoder(in, latent)`, `vae(in, latent)` with `vae_loss`,
`gcn_net`, `gat_net`, `gin_net` (graph classification), `siamese(encoder, combine, head)`.

**Training** — `fit(model, params, X, y, loss="mse"|"cross_entropy"|"mae"|"bce"|"vae"|fn, opt, epochs, batch_size,
clip, schedule, graph, mask, val=(Xv, yv))` returns `record(params, history, val_history)`; `predict`, `summary`,
`count_params`. Full example: `examples/28_any_architecture.apsi`.

**On GPUs with PyTorch** — the same model, trained by PyTorch on CUDA / ROCm / Apple MPS / Intel XPU with mixed
precision and `torch.compile`; the result is a normal Apsi parameter tree again (details: [`BACKENDS.md`](BACKENDS.md)):

```apsi
let r = fit(model, params, X, y, loss="cross_entropy", epochs=20, batch_size=256,
            backend="torch", device="cuda", precision="bf16", compile=true)     # run with --allow-python
let m = torch_module(model, r.params)        # a real torch.nn.Module: state_dict, DDP, ONNX export
check_torch()                                # every layer compared with NumPy (outputs and gradients)
```

## 2. Design your own, function by function

```apsi
use nn

# a home-made architecture: gated residual blocks
fn block(p, x):
    let gate = sigmoid(x @ p.g.W + p.g.b)
    let h = gelu(layer_norm(x) @ p.f.W + p.f.b)
    return x + gate * h

fn model(p, x):
    var h = x @ p.inp.W
    for layer in p.blocks:
        h = block(layer, h)
    return h @ p.out.W

let params = record(inp=dense(16, 64), blocks=[record(g=dense(64, 64), f=dense(64, 64)) for _ in 0..4],
                    out=dense(64, 1))
let loss, grads = value_and_grad(fn(p) => mse(model(p, X), y))(params)
```

Building blocks:

| Area | Available |
|---|---|
| Ops | `@`, `einsum`, conv2d, max-pool, attention, layer/batch/RMS norm, dropout, embeddings, causal mask, LSTM, GRU |
| Activations | relu, gelu, sigmoid, tanh, softplus, softmax, log-softmax — and any function you write |
| Losses | mse, cross-entropy, or your own (including physics losses that use higher-order autodiff) |
| Optimizers | SGD (momentum), Adam, AdamW, RMSprop, Adagrad, L-BFGS (`minimize`), gradient clipping, LR schedules |
| Training | the ready-made `train()` or your own loop (`value_and_grad` + `opt.step`) |
| Data | `use data` (streaming, parallel, prefetch, SSD cache) |
| Precision | f64/f32/f16/bf16/fp8/fp6/fp4/int8/int4, quantization-aware training (`round_to`) |

## 3. Use less RAM / VRAM

| Technique | How | Effect |
|---|---|---|
| Memory-mapped loading | `load_safetensors(path)` | any file size opens instantly; only the parts you use are read |
| Load in a small format | `load_safetensors(path, dtype=int8)` / `fp8_e4m3` / `int4` / `fp4_e2m1` | 4–8× smaller than f32 (16× smaller than f64) |
| Compress an existing model | `quantize_model(params, int8)`, then compute with `qlinear(x, W, b)` | example 26: 67.5 KB → 18.6 KB with the same accuracy (99.3 %) |
| Train to tolerate compression | `round_to(W, fp8_e4m3)` inside the forward pass | gradients pass through (straight-through estimator) |
| Data larger than RAM | `open_tensor` / `load_tensor(mmap=true)` + `from_tensors(batch=…)` | stream from SSD |

## 4. Use pretrained models from other ecosystems

Install the library with pip as usual and call it from Apsi (run with `--allow-python`):

```apsi
use models
let classify = hf_pipeline("sentiment-analysis")          # pip install transformers torch
print(classify("Apsi is great"))

let net = onnx_model("resnet50.onnx")                       # pip install onnxruntime(-gpu)
let logits = net(image_batch)                               # picks TensorRT / CUDA / DirectML / CPU for you

let weights = torch_load("checkpoint.pt")                   # pip install torch (safe weights_only mode)
let hf_weights = load_safetensors("model.safetensors", dtype=int8)   # no extra library needed
```

Existing PyTorch/numpy code can be translated with `apsi convert model.py` (see [`PYTHON.md`](PYTHON.md)).
Any other Python library: `pyimport("module")`, then `to_torch` / `from_torch` to move tensors back and forth
(after `use layers`, `torch_module(model, params)` turns a whole Apsi model into a `torch.nn.Module`).
`apsi pkg install ai` installs ONNX Runtime, PyTorch and Transformers into Apsi's own runtime.

## 5. Serve it as an API

```apsi
use web
let app = server()
@app.post("/predict")
fn predict(req) = {"y": model(params, tensor(req.json()["x"]))}
app.run(port=8080)
```

Complete examples: `examples/23_web_api.apsi`, `26_model_serving.apsi` and `27_pretrained_models.apsi`.
Server security is covered in [`SERVER.md`](SERVER.md).

## Model file safety

- `load_safetensors` never runs code from the file (unlike pickle) and checks that every offset stays inside the file
- `torch_load` uses `weights_only=True`, so code embedded in a `.pt` file is not executed
- the on-disk cache is signed with HMAC
