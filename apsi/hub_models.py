"""Running Hugging Face models natively (NumPy, no PyTorch, no transformers) for `use hub`.

Architectures (from config.json "model_type" / "architectures"):

* encoders — bert, roberta, xlm-roberta, distilbert (and sentence-transformers models built on them):
  embeddings, sentence similarity, text classification, token classification, fill-mask;
* decoders — gpt2, llama, mistral, qwen2, qwen3, gemma: text generation with a key/value cache, greedy or sampled.

Weights come from safetensors (also sharded, BF16/F16/F32); they are converted to float32 once and kept.
"""
import json
import math
import os

import numpy as np

from .errors import RuntimeErr


class ModelError(RuntimeErr):
    kind = "HubError"


# ======================================================================================== weights
def read_safetensors(path):
    """{name: float32 array} keeping the original tensor names."""
    from .models import _read_header, _ST_TO_NP, _bf16_decode
    header, start = _read_header(path)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        if not (0 <= a <= b and start + b <= len(mm)):
            raise ModelError(f"safetensors: tensor '{name}' points outside the file")
        raw = mm[start + a:start + b]
        st, shape = info["dtype"], tuple(info["shape"])
        if st == "BF16":
            arr = _bf16_decode(raw.view(np.uint16)).reshape(shape)
        elif st in _ST_TO_NP:
            arr = raw.view(_ST_TO_NP[st]).reshape(shape)
        else:
            raise ModelError(f"safetensors: tensor '{name}' has dtype {st}, which the native runner does not read")
        out[name] = np.asarray(arr, np.float32) if np.asarray(arr).dtype.kind == "f" else np.asarray(arr)
    return out


def _strip(weights, prefixes):
    """Remove one leading model prefix (bert., roberta., model., transformer. …) when all body weights carry it."""
    for p in prefixes:
        if any(k.startswith(p) for k in weights):
            return {(k[len(p):] if k.startswith(p) else k): v for k, v in weights.items()}, True
    return weights, False


def _ln(x, w, b, eps):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def _rms(x, w, eps, offset=0.0):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * (w + offset)


def _gelu(x):                         # exact (erf) GELU, BERT's "gelu"
    from math import erf
    v = np.vectorize(erf, otypes=[np.float64])
    return (0.5 * x * (1.0 + v(x / math.sqrt(2.0)))).astype(np.float32) if x.size < 4096 else _gelu_fast(x)


def _gelu_fast(x):
    # erf through the Abramowitz–Stegun 7.1.26 rational form is not exact enough; use scipy when present
    try:
        from scipy.special import erf
        return (0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))).astype(np.float32)
    except ImportError:
        import math as _m
        f = np.frompyfunc(_m.erf, 1, 1)
        return (0.5 * x * (1.0 + f(x / _m.sqrt(2.0)).astype(np.float64))).astype(np.float32)


def _gelu_tanh(x):                    # GPT-2 / Gemma "gelu_new" / "gelu_pytorch_tanh"
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _act(name):
    name = (name or "gelu").lower()
    if name in ("gelu", "gelu_python"):
        return _gelu
    if name in ("gelu_new", "gelu_pytorch_tanh", "gelu_fast", "gelu_tanh"):
        return _gelu_tanh
    if name in ("silu", "swish"):
        return lambda x: x / (1.0 + np.exp(-x))
    if name == "relu":
        return lambda x: np.maximum(x, 0)
    if name == "tanh":
        return np.tanh
    raise ModelError(f"activation '{name}' is not supported natively")


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _lin(x, w, b=None):
    y = x @ w.T
    return y + b if b is not None else y


# ======================================================================================== encoders
class Encoder:
    """BERT / RoBERTa / XLM-RoBERTa / DistilBERT encoder with the heads that came with the checkpoint."""

    def __init__(self, cfg, W):
        self.cfg = cfg
        self.type = cfg.get("model_type", "bert")
        self.distil = self.type == "distilbert"
        prefixes = ("distilbert.",) if self.distil else ("bert.", "roberta.", "xlm_roberta.", "model.")
        self.W, _ = _strip(W, prefixes)
        self.eps = cfg.get("layer_norm_eps", 1e-12)
        self.heads = cfg.get("num_attention_heads") or cfg.get("n_heads")
        self.layers = cfg.get("num_hidden_layers") or cfg.get("n_layers")
        self.act = _act(cfg.get("hidden_act") or cfg.get("activation") or "gelu")
        self.pad = cfg.get("pad_token_id", 1 if self.type in ("roberta", "xlm-roberta") else 0)

    def g(self, name, required=True):
        v = self.W.get(name)
        if v is None and required:
            raise ModelError(f"the checkpoint has no weight '{name}' (model type {self.type})")
        return v

    def hidden(self, ids, mask=None, token_types=None):
        ids = np.atleast_2d(np.asarray(ids, np.int64))
        B, T = ids.shape
        mask = np.ones((B, T), np.float32) if mask is None else np.atleast_2d(np.asarray(mask, np.float32))
        if self.distil:
            e = self.g("embeddings.word_embeddings.weight")[ids] + self.g("embeddings.position_embeddings.weight")[:T]
            x = _ln(e, self.g("embeddings.LayerNorm.weight"), self.g("embeddings.LayerNorm.bias"), 1e-12)
        else:
            if self.type in ("roberta", "xlm-roberta", "camembert"):
                m = (ids != self.pad).astype(np.int64)
                pos = np.cumsum(m, axis=1) * m + self.pad
            else:
                pos = np.broadcast_to(np.arange(T), (B, T))
            tt = np.zeros((B, T), np.int64) if token_types is None else np.atleast_2d(np.asarray(token_types))
            e = self.g("embeddings.word_embeddings.weight")[ids] + self.g("embeddings.position_embeddings.weight")[pos]
            te = self.g("embeddings.token_type_embeddings.weight", False)
            if te is not None:
                e = e + te[tt]
            x = _ln(e, self.g("embeddings.LayerNorm.weight"), self.g("embeddings.LayerNorm.bias"), self.eps)
        neg = (1.0 - mask)[:, None, None, :] * np.float32(-1e9)
        H = x.shape[-1]
        nh = self.heads
        dh = H // nh
        for i in range(self.layers):
            if self.distil:
                p = f"transformer.layer.{i}."
                q = _lin(x, self.g(p + "attention.q_lin.weight"), self.g(p + "attention.q_lin.bias"))
                k = _lin(x, self.g(p + "attention.k_lin.weight"), self.g(p + "attention.k_lin.bias"))
                v = _lin(x, self.g(p + "attention.v_lin.weight"), self.g(p + "attention.v_lin.bias"))
            else:
                p = f"encoder.layer.{i}."
                q = _lin(x, self.g(p + "attention.self.query.weight"), self.g(p + "attention.self.query.bias"))
                k = _lin(x, self.g(p + "attention.self.key.weight"), self.g(p + "attention.self.key.bias"))
                v = _lin(x, self.g(p + "attention.self.value.weight"), self.g(p + "attention.self.value.bias"))
            q, k, v = (t.reshape(B, T, nh, dh).transpose(0, 2, 1, 3) for t in (q, k, v))
            att = _softmax(q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh) + neg)
            ctx = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, H)
            if self.distil:
                a = _lin(ctx, self.g(p + "attention.out_lin.weight"), self.g(p + "attention.out_lin.bias"))
                x = _ln(a + x, self.g(p + "sa_layer_norm.weight"), self.g(p + "sa_layer_norm.bias"), 1e-12)
                f = _lin(self.act(_lin(x, self.g(p + "ffn.lin1.weight"), self.g(p + "ffn.lin1.bias"))),
                         self.g(p + "ffn.lin2.weight"), self.g(p + "ffn.lin2.bias"))
                x = _ln(f + x, self.g(p + "output_layer_norm.weight"), self.g(p + "output_layer_norm.bias"), 1e-12)
            else:
                a = _lin(ctx, self.g(p + "attention.output.dense.weight"), self.g(p + "attention.output.dense.bias"))
                x = _ln(a + x, self.g(p + "attention.output.LayerNorm.weight"),
                        self.g(p + "attention.output.LayerNorm.bias"), self.eps)
                f = self.act(_lin(x, self.g(p + "intermediate.dense.weight"), self.g(p + "intermediate.dense.bias")))
                f = _lin(f, self.g(p + "output.dense.weight"), self.g(p + "output.dense.bias"))
                x = _ln(f + x, self.g(p + "output.LayerNorm.weight"), self.g(p + "output.LayerNorm.bias"), self.eps)
        return x

    def pooled(self, x):
        w = self.g("pooler.dense.weight", False)
        if w is None:
            return x[:, 0]
        return np.tanh(_lin(x[:, 0], w, self.g("pooler.dense.bias")))

    def classify(self, x):
        """Sequence-classification logits with the head stored in the checkpoint."""
        W = self.W
        if "classifier.out_proj.weight" in W:                     # RoBERTa head on <s>
            h = np.tanh(_lin(x[:, 0], W["classifier.dense.weight"], W["classifier.dense.bias"]))
            return _lin(h, W["classifier.out_proj.weight"], W["classifier.out_proj.bias"])
        if "pre_classifier.weight" in W:                           # DistilBERT head
            h = np.maximum(_lin(x[:, 0], W["pre_classifier.weight"], W["pre_classifier.bias"]), 0)
            return _lin(h, W["classifier.weight"], W["classifier.bias"])
        if "classifier.weight" in W:                               # BERT head on the pooled output
            return _lin(self.pooled(x), W["classifier.weight"], W["classifier.bias"])
        raise ModelError("this checkpoint has no classification head (it is a base model): use embed() instead")

    def token_logits(self, x):
        if "classifier.weight" not in self.W:
            raise ModelError("this checkpoint has no token-classification head")
        return _lin(x, self.W["classifier.weight"], self.W["classifier.bias"])

    def mlm_logits(self, x):
        W = self.W
        if "cls.predictions.transform.dense.weight" in W:          # BERT
            h = self.act(_lin(x, W["cls.predictions.transform.dense.weight"], W["cls.predictions.transform.dense.bias"]))
            h = _ln(h, W["cls.predictions.transform.LayerNorm.weight"], W["cls.predictions.transform.LayerNorm.bias"],
                    self.eps)
            dec = W.get("cls.predictions.decoder.weight", self.g("embeddings.word_embeddings.weight"))
            return h @ dec.T + W.get("cls.predictions.bias", W.get("cls.predictions.decoder.bias", 0))
        if "lm_head.dense.weight" in W:                            # RoBERTa
            h = _gelu(_lin(x, W["lm_head.dense.weight"], W["lm_head.dense.bias"]))
            h = _ln(h, W["lm_head.layer_norm.weight"], W["lm_head.layer_norm.bias"], self.eps)
            dec = W.get("lm_head.decoder.weight", self.g("embeddings.word_embeddings.weight"))
            return h @ dec.T + W.get("lm_head.bias", W.get("lm_head.decoder.bias", 0))
        if "vocab_transform.weight" in W:                          # DistilBERT
            h = _gelu(_lin(x, W["vocab_transform.weight"], W["vocab_transform.bias"]))
            h = _ln(h, W["vocab_layer_norm.weight"], W["vocab_layer_norm.bias"], 1e-12)
            dec = W.get("vocab_projector.weight", self.g("embeddings.word_embeddings.weight"))
            return h @ dec.T + W["vocab_projector.bias"]
        raise ModelError("this checkpoint has no masked-language-model head")


# ======================================================================================== decoders
class Decoder:
    """GPT-2 and the Llama family (Llama, Mistral, Qwen2/3, Gemma) with a key/value cache."""

    def __init__(self, cfg, W):
        self.cfg = cfg
        t = cfg.get("model_type", "")
        self.type = t
        self.gpt2 = t == "gpt2"
        if self.gpt2:
            self.W, _ = _strip(W, ("transformer.",))
            self.L = cfg["n_layer"]
            self.nh = cfg["n_head"]
            self.kvh = self.nh
            self.H = cfg["n_embd"]
            self.dh = self.H // self.nh
            self.eps = cfg.get("layer_norm_epsilon", 1e-5)
            self.ctx = cfg.get("n_positions", 1024)
        else:
            self.W = W
            self.L = cfg["num_hidden_layers"]
            self.nh = cfg["num_attention_heads"]
            self.kvh = cfg.get("num_key_value_heads") or self.nh
            self.H = cfg["hidden_size"]
            self.dh = cfg.get("head_dim") or self.H // self.nh
            self.eps = cfg.get("rms_norm_eps", 1e-6)
            self.ctx = cfg.get("max_position_embeddings", 4096)
            self.gemma = t.startswith("gemma")
            self.act = _act(cfg.get("hidden_activation") or cfg.get("hidden_act") or "silu")
            self._rope_init(cfg)
        self.window = cfg.get("sliding_window") if cfg.get("use_sliding_window", True) else None

    def g(self, name):
        v = self.W.get(name)
        if v is None:
            raise ModelError(f"the checkpoint has no weight '{name}' (model type {self.type})")
        return v

    def _rope_init(self, cfg):
        base = cfg.get("rope_theta", 10000.0)
        rot = int(self.dh * cfg.get("partial_rotary_factor", 1.0))
        inv = 1.0 / base ** (np.arange(0, rot, 2, dtype=np.float64) / rot)
        sc = cfg.get("rope_scaling") or {}
        kind = sc.get("rope_type") or sc.get("type")
        if kind == "llama3":
            factor, lo, hi = sc["factor"], sc.get("low_freq_factor", 1.0), sc.get("high_freq_factor", 4.0)
            old = sc.get("original_max_position_embeddings", 8192)
            wave = 2 * math.pi / inv
            smooth = (old / wave - lo) / (hi - lo)
            scaled = np.where(wave > old / lo, inv / factor, inv)
            mid = (1 - smooth) * inv / factor + smooth * inv
            inv = np.where((wave <= old / lo) & (wave >= old / hi), mid, scaled)
        elif kind == "linear":
            inv = inv / sc["factor"]
        elif kind not in (None, "default"):
            raise ModelError(f"rope scaling '{kind}' is not supported natively")
        self.inv_freq = inv
        self.rot = rot

    def _rope(self, x, pos):
        f = np.outer(pos, self.inv_freq)                           # [T, rot/2]
        emb = np.concatenate([f, f], -1)
        cos, sin = np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)
        r, rest = x[..., :self.rot], x[..., self.rot:]
        half = self.rot // 2
        rh = np.concatenate([-r[..., half:], r[..., :half]], -1)
        out = r * cos + rh * sin
        return np.concatenate([out, rest], -1) if rest.shape[-1] else out

    def new_cache(self):
        return [[None, None] for _ in range(self.L)]

    def forward(self, ids, cache, start):
        """Logits for ids [1, T] given `start` tokens already in the cache (which is updated)."""
        ids = np.atleast_2d(np.asarray(ids, np.int64))
        T = ids.shape[1]
        pos = np.arange(start, start + T)
        W = self.W
        if self.gpt2:
            x = self.g("wte.weight")[ids] + self.g("wpe.weight")[pos]
        else:
            x = self.g("model.embed_tokens.weight")[ids]
            if self.gemma:
                x = x * np.float32(math.sqrt(self.H))
        total = start + T
        causal = np.triu(np.full((T, total), -1e9, np.float32), k=start + 1)
        if self.window and total > self.window:
            qpos = pos[:, None]
            kpos = np.arange(total)[None, :]
            causal = causal + np.where(qpos - kpos >= self.window, np.float32(-1e9), 0)
        for i in range(self.L):
            if self.gpt2:
                p = f"h.{i}."
                h = _ln(x, self.g(p + "ln_1.weight"), self.g(p + "ln_1.bias"), self.eps)
                qkv = h @ self.g(p + "attn.c_attn.weight") + self.g(p + "attn.c_attn.bias")
                q, k, v = np.split(qkv, 3, axis=-1)
            else:
                p = f"model.layers.{i}."
                h = _rms(x, self.g(p + "input_layernorm.weight"), self.eps, 1.0 if self.gemma else 0.0)
                q = _lin(h, self.g(p + "self_attn.q_proj.weight"), W.get(p + "self_attn.q_proj.bias"))
                k = _lin(h, self.g(p + "self_attn.k_proj.weight"), W.get(p + "self_attn.k_proj.bias"))
                v = _lin(h, self.g(p + "self_attn.v_proj.weight"), W.get(p + "self_attn.v_proj.bias"))
            q = q.reshape(1, T, self.nh, self.dh).transpose(0, 2, 1, 3)
            k = k.reshape(1, T, self.kvh, self.dh).transpose(0, 2, 1, 3)
            v = v.reshape(1, T, self.kvh, self.dh).transpose(0, 2, 1, 3)
            if not self.gpt2:
                if p + "self_attn.q_norm.weight" in W:             # Qwen3: per-head RMSNorm before RoPE
                    q = _rms(q, W[p + "self_attn.q_norm.weight"], self.eps)
                    k = _rms(k, W[p + "self_attn.k_norm.weight"], self.eps)
                q = self._rope(q, pos)
                k = self._rope(k, pos)
            ck, cv = cache[i]
            k = k if ck is None else np.concatenate([ck, k], 2)
            v = v if cv is None else np.concatenate([cv, v], 2)
            cache[i] = [k, v]
            rep = self.nh // self.kvh
            kk = np.repeat(k, rep, axis=1) if rep > 1 else k
            vv = np.repeat(v, rep, axis=1) if rep > 1 else v
            scale = self.cfg.get("query_pre_attn_scalar")
            scale = 1.0 / math.sqrt(scale) if scale else 1.0 / math.sqrt(self.dh)
            att = _softmax(q @ kk.transpose(0, 1, 3, 2) * scale + causal)
            o = (att @ vv).transpose(0, 2, 1, 3).reshape(1, T, self.nh * self.dh)
            if self.gpt2:
                x = x + (o @ self.g(p + "attn.c_proj.weight") + self.g(p + "attn.c_proj.bias"))
                h = _ln(x, self.g(p + "ln_2.weight"), self.g(p + "ln_2.bias"), self.eps)
                m = _gelu_tanh(h @ self.g(p + "mlp.c_fc.weight") + self.g(p + "mlp.c_fc.bias"))
                x = x + (m @ self.g(p + "mlp.c_proj.weight") + self.g(p + "mlp.c_proj.bias"))
            else:
                x = x + _lin(o, self.g(p + "self_attn.o_proj.weight"), W.get(p + "self_attn.o_proj.bias"))
                h = _rms(x, self.g(p + "post_attention_layernorm.weight"), self.eps, 1.0 if self.gemma else 0.0)
                m = self.act(_lin(h, self.g(p + "mlp.gate_proj.weight"))) * _lin(h, self.g(p + "mlp.up_proj.weight"))
                x = x + _lin(m, self.g(p + "mlp.down_proj.weight"))
        if self.gpt2:
            x = _ln(x, self.g("ln_f.weight"), self.g("ln_f.bias"), self.eps)
            head = W.get("lm_head.weight", self.g("wte.weight"))
        else:
            x = _rms(x, self.g("model.norm.weight"), self.eps, 1.0 if self.gemma else 0.0)
            head = W.get("lm_head.weight")
            if head is None:
                head = self.g("model.embed_tokens.weight")
        logits = x @ head.T
        cap = self.cfg.get("final_logit_softcapping")
        if cap:
            logits = np.tanh(logits / cap) * cap
        return logits

    def generate(self, ids, max_new_tokens=64, temperature=0.0, top_k=0, top_p=1.0, eos=None, seed=None,
                 repetition_penalty=1.0, stream=None):
        ids = list(np.asarray(ids).reshape(-1))
        if len(ids) + max_new_tokens > self.ctx:
            max_new_tokens = max(0, self.ctx - len(ids))
        rng = np.random.default_rng(seed)
        cache = self.new_cache()
        logits = self.forward(np.array([ids]), cache, 0)[0, -1]
        out = []
        eos = set([eos] if isinstance(eos, int) else (eos or []))
        for _ in range(int(max_new_tokens)):
            lg = logits.astype(np.float64)
            if repetition_penalty != 1.0:
                for t in set(ids + out):
                    lg[t] = lg[t] / repetition_penalty if lg[t] > 0 else lg[t] * repetition_penalty
            if not temperature:
                nxt = int(np.argmax(lg))
            else:
                lg = lg / temperature
                if top_k:
                    kth = np.partition(lg, -int(top_k))[-int(top_k)]
                    lg = np.where(lg < kth, -np.inf, lg)
                pr = _softmax(lg)
                if top_p < 1.0:
                    order = np.argsort(-pr)
                    c = np.cumsum(pr[order])
                    keep = order[:max(1, int(np.searchsorted(c, top_p)) + 1)]
                    mask = np.zeros_like(pr)
                    mask[keep] = 1
                    pr = pr * mask
                    pr = pr / pr.sum()
                nxt = int(rng.choice(len(pr), p=pr))
            if nxt in eos:
                break
            out.append(nxt)
            if stream is not None:
                stream(nxt)
            logits = self.forward(np.array([[nxt]]), cache, len(ids) + len(out) - 1)[0, -1]
        return out


# ======================================================================================== loading
ENCODERS = {"bert", "roberta", "xlm-roberta", "distilbert", "camembert"}
DECODERS = {"gpt2", "llama", "mistral", "qwen2", "qwen3", "gemma"}


def build(cfg, weights):
    t = cfg.get("model_type", "")
    if t in ENCODERS:
        return Encoder(cfg, weights)
    if t in DECODERS:
        return Decoder(cfg, weights)
    raise ModelError(f"model type '{t}' does not run natively yet (natively: {', '.join(sorted(ENCODERS | DECODERS))}); "
                     "use hf_pipeline with --allow-python")


def load_folder(folder):
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    W = {}
    idx = os.path.join(folder, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx, encoding="utf-8") as fh:
            for shard in sorted(set(json.load(fh)["weight_map"].values())):
                W.update(read_safetensors(os.path.join(folder, shard)))
    else:
        W.update(read_safetensors(os.path.join(folder, "model.safetensors")))
    return cfg, W
