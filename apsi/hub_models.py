"""Running Hugging Face models natively (NumPy, no PyTorch, no transformers) for `use hub`.

Architectures (from config.json "model_type" / "architectures"):

* encoders — bert, roberta, xlm-roberta, distilbert (and sentence-transformers models built on them):
  embeddings, sentence similarity, text classification, token classification, fill-mask;
* decoders — gpt2, llama, mistral, qwen2, qwen3, gemma: text generation with a key/value cache, greedy or sampled.

Weights come from safetensors (also sharded, BF16/F16/F32). The file is read once, in its own format; a tensor is
decoded to float32 the first time the processor needs it. On a GPU the bfloat16 weights go to the card as they are,
so a float32 copy of the model is never made.
"""
import json
import math
import os
import shutil
import sys

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
    raw = arr = None
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
        # copies: the weights are read in full by every forward pass anyway, and no open mapping is left behind
        # (on Windows an open mapping keeps the file locked, so a cached model could not be replaced or deleted)
        out[name] = np.array(arr, np.float32 if arr.dtype.kind == "f" else arr.dtype)
    del raw, arr
    try:
        mm._mmap.close()
    except (AttributeError, BufferError, ValueError):     # still referenced: closed when collected
        pass
    del mm
    return out


class _Ref:
    """A tensor still in its file: read when needed (no RAM is held for it until then)."""
    __slots__ = ("path", "offset", "shape", "dtype")

    def __init__(self, path, offset, shape, dtype):
        self.path, self.offset, self.shape, self.dtype = path, offset, tuple(shape), np.dtype(dtype)

    @property
    def nbytes(self):
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    def load(self):
        n = int(np.prod(self.shape, dtype=np.int64))
        a = np.fromfile(self.path, dtype=self.dtype, count=n, offset=self.offset)
        if a.size != n:
            raise ModelError(f"{self.path} is shorter than its header says (changed or damaged?)")
        return a.reshape(self.shape)

    def rows(self, ids):
        """Some rows of a matrix, read one by one (embedding lookups)."""
        ids = np.asarray(ids, np.int64).reshape(-1)
        cols = int(np.prod(self.shape[1:], dtype=np.int64))
        out = np.empty((ids.size, cols), self.dtype)
        rb = cols * self.dtype.itemsize
        with open(self.path, "rb") as fh:
            for k, i in enumerate(ids):
                if not 0 <= i < self.shape[0]:
                    raise IndexError(f"token id {int(i)} is outside the vocabulary ({self.shape[0]})")
                fh.seek(self.offset + int(i) * rb)
                out[k] = np.frombuffer(fh.read(rb), self.dtype)
        return out



def _progress(msg):
    """A line on stderr about long work (making the int8 copy of a large model …), so a wait is never silent.
    APSI_PROGRESS=0 turns it off."""
    if os.environ.get("APSI_PROGRESS", "1").strip() in ("0", "false", "no"):
        return
    try:
        sys.stderr.write(f"apsi hub: {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _free_of(r):
    try:
        return int(r.free_memory())
    except Exception:
        return None


def _ram_available():
    try:
        from .memory import _meminfo
        total, avail, _, _ = _meminfo()
        return int(total), int(avail)
    except Exception:
        return 0, 0


class _Q8Store:
    """int8 copies of the matrices for the processor, made once and kept on disk next to the GPU kernel cache
    (memory-mapped afterwards: the next start reads them at disk speed and the OS shares the pages)."""

    def __init__(self, net, groups):
        import hashlib
        from .security import config_dir
        self.net = net
        h = hashlib.sha256(b"apsi-q8-1")
        for ns in groups.values():
            for n in ns:
                nm = n if n else net._head_name()
                kind, ref = net.W._raw.get(nm, (None, None))
                src = getattr(net.W, "src", {}).get(nm)
                if isinstance(ref, _Ref):
                    src = (ref.path, ref.offset, ref.shape)
                if src is not None:                         # read lazily or memory-mapped: keyed by the file
                    st = os.stat(src[0])
                    h.update(f"{os.path.abspath(src[0])}|{st.st_size}|{int(st.st_mtime)}|{src[1]}|{src[2]}".encode())
                else:
                    h = None
                    break
            if h is None:
                break
        self.dir = os.path.join(config_dir(), "cache", "cpu-int8", h.hexdigest()[:24]) if h is not None else None
        self.ok = self.dir is not None and os.path.exists(os.path.join(self.dir, "complete"))
        self.made = 0
        self.total = sum(int(np.prod(net._wshape(n if n else net._head_name()))) for ns in groups.values() for n in ns)
        self.done_values = 0
        self.t0 = None
        self.last = 0.0
        if not self.ok and self.dir is not None:
            need = self.total * 1.13                         # int8 values and one float32 scale per 32
            try:
                os.makedirs(os.path.dirname(self.dir), exist_ok=True)
                free = shutil.disk_usage(os.path.dirname(self.dir)).free
            except OSError:
                free = need * 2
            if free < need * 1.1:                            # no room for the copy: kept in RAM instead
                _progress(f"only {free / 2 ** 30:.1f} GB free on the disk for the int8 copy "
                          f"({need / 2 ** 30:.1f} GB needed): it stays in RAM")
                self.dir = None
            elif self.total * 2 > 2 ** 30:
                _progress(f"making the int8 copy of the processor's weights once ({need / 2 ** 30:.1f} GB in "
                          f"{os.path.dirname(self.dir)}); later starts read it directly")

    def _paths(self, key):
        safe = key.replace("/", "_")
        return os.path.join(self.dir, safe + ".codes.npy"), os.path.join(self.dir, safe + ".scales.npy")

    def get(self, key, names):
        from . import hub_cpu
        if self.ok:
            try:
                c, sc = self._paths(key)
                return hub_cpu.Q8(np.load(c, mmap_mode="r"), np.load(sc, mmap_mode="r"))
            except (OSError, ValueError):
                self.ok = False
        import time as _t
        if self.t0 is None:
            self.t0 = _t.perf_counter()
        q = hub_cpu.Q8.from_bf16(self.net._bf16(names))
        self.made += 1
        self.done_values += q.codes.size
        now = _t.perf_counter()
        if self.total * 2 > 2 ** 30 and (now - self.last > 5 or self.done_values >= self.total):
            self.last = now
            frac = self.done_values / max(self.total, 1)
            left = (now - self.t0) / max(frac, 1e-9) * (1 - frac)
            _progress(f"int8 copy {frac * 100:.0f} % ({self.done_values * 1.13 / 2 ** 30:.1f} of "
                      f"{self.total * 1.13 / 2 ** 30:.1f} GB)" + (f", about {left:.0f} s left" if frac < 1 else ", done"))
        if self.dir is not None:
            try:
                os.makedirs(self.dir, exist_ok=True)
                c, sc = self._paths(key)
                for path, arr in ((c, q.codes), (sc, q.scales)):
                    with open(path + ".tmp", "wb") as fh:
                        np.save(fh, arr)
                    os.replace(path + ".tmp", path)
                # memory-mapped from now on: a model larger than the free RAM pages in and out instead of
                # filling the memory (a 14B model's processor half is 10 GB in int8)
                q = hub_cpu.Q8(np.load(c, mmap_mode="r"), np.load(sc, mmap_mode="r"))
            except OSError:
                self.dir = None                             # read-only home: keep it in RAM only
        return q

    def done(self):
        if self.made and self.dir is not None:
            try:
                open(os.path.join(self.dir, "complete"), "w").close()
            except OSError:
                pass

class Weights:
    """name -> float32 array, decoded from the checkpoint's own format the first time it is used.

    The file is not read when the model is opened: each tensor is read when first needed (on a GPU, straight to the
    card, so a whole copy of the model is never held in RAM). raw(name) gives the stored form, ("bf16", uint16
    array) or ("f32", float32 array); rows(name, ids) reads only some rows (embedding lookups); shape(name) reads
    nothing."""

    KEEP = 4 << 20                                          # stored tensors up to this size stay in memory once read

    def __init__(self):
        self._raw = {}
        self._f32 = {}
        self.mapped = False                                 # memory-mapped: no float32 cache of big matrices

    def add(self, name, kind, arr):
        self._raw[name] = (kind, arr)
        self._f32.pop(name, None)

    def __getitem__(self, name):
        v = self._f32.get(name)
        if v is None:
            kind, arr = self.raw(name)                      # KeyError for a missing weight, like a dict
            v = _decode(kind, arr)
            if not self.mapped or v.ndim < 2:               # mapped: big matrices are decoded each time
                self._f32[name] = v
        return v

    def get(self, name, default=None):
        return self[name] if name in self._raw else default

    def __contains__(self, name):
        return name in self._raw

    def __iter__(self):
        return iter(self._raw)

    def __len__(self):
        return len(self._raw)

    def keys(self):
        return self._raw.keys()

    def items(self):
        return ((k, self[k]) for k in self._raw)

    def raw_kind(self, name):
        return self._raw[name][0]

    def raw(self, name):
        kind, arr = self._raw[name]
        if isinstance(arr, _Ref):
            ref = arr
            arr = ref.load()
            if ref.nbytes <= self.KEEP:
                self._raw[name] = (kind, arr)
        return kind, arr

    def shape(self, name):
        return tuple(self._raw[name][1].shape)

    def rows(self, name, ids):
        kind, arr = self._raw[name]
        if name in self._f32:
            return self._f32[name][ids]
        if isinstance(arr, _Ref):
            shape = np.shape(ids)
            r = arr.rows(ids)
            return _decode(kind, r).reshape(*shape, -1)
        return _decode(kind, arr[ids])

    def renamed(self, fn):
        w = Weights()
        w.mapped = self.mapped
        w._raw = {fn(k): v for k, v in self._raw.items()}
        w._f32 = {fn(k): v for k, v in self._f32.items()}
        return w

    def decode_all(self, drop_raw=False):
        """Everything as float32 (the processor path); drop_raw frees the stored form afterwards."""
        for k in self._raw:
            self[k]
        if drop_raw:
            self._raw = {k: ("f32", v) for k, v in self._f32.items()}


def _decode(kind, arr):
    if kind == "bf16":
        a = np.ascontiguousarray(arr)
        return (a.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return np.ascontiguousarray(arr, np.float32) if arr.dtype.kind == "f" else np.asarray(arr)


def read_safetensors_lazy(path, into=None, mmap=False):
    """The tensors of a safetensors file in their stored format, read only when used (mmap=True maps the file
    instead: the operating system pages the weights in from the disk). Returns a Weights."""
    from .models import _read_header, _ST_TO_NP
    header, start = _read_header(path)
    size = os.path.getsize(path)
    buf = np.memmap(path, dtype=np.uint8, mode="r") if mmap else None
    W = into if into is not None else Weights()
    W.mapped = W.mapped or bool(mmap)
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        if not (0 <= a <= b and start + b <= size):
            raise ModelError(f"safetensors: tensor '{name}' points outside the file")
        st, shape = info["dtype"], tuple(info["shape"])
        if st == "BF16":
            kind, dt = "bf16", np.uint16
        elif st in _ST_TO_NP:
            kind, dt = ("f32" if st == "F32" else st.lower()), _ST_TO_NP[st]
        else:
            raise ModelError(f"safetensors: tensor '{name}' has dtype {st}, which the native runner does not read")
        if int(np.prod(shape, dtype=np.int64)) * np.dtype(dt).itemsize != b - a:
            raise ModelError(f"safetensors: tensor '{name}' has the wrong size for its shape")
        if buf is not None:
            W.add(name, kind, buf[start + a:start + b].view(dt).reshape(shape))
            W.__dict__.setdefault("src", {})[name] = (path, start + a, shape)     # for caches keyed by the file
        else:
            W.add(name, kind, _Ref(path, start + a, shape, dt))
    return W


def _strip(weights, prefixes):
    """Remove one leading model prefix (bert., roberta., model., transformer. …) when all body weights carry it."""
    for p in prefixes:
        if any(k.startswith(p) for k in weights):
            fn = lambda k: k[len(p):] if k.startswith(p) else k      # noqa: E731
            if isinstance(weights, Weights):
                return weights.renamed(fn), True
            return {fn(k): v for k, v in weights.items()}, True
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
        self.kv_cache = os.environ.get("APSI_KV_CACHE", "fp16").lower()   # GPU cache: "fp16" or "fp32"
        self.gpu = None                 # hub_gpu.Runner while the matrices live on a GPU
        self._gm = {}                   # key -> (GPU matrix, split points): the per-product path
        self._gl = None                 # per-layer GPU weights: the whole-network path

    # ---------------------------------------------------------------- devices
    def _groups(self):
        """key -> weight names multiplied with the same input (fused into one matrix on the GPU)."""
        out = {}
        for i in range(self.L):
            if self.gpt2:
                p = f"h.{i}."
                for k in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
                    out[p + k] = [p + k + ".weight"]
            else:
                p = f"model.layers.{i}."
                out[p + "qkv"] = [p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"]
                out[p + "o"] = [p + "self_attn.o_proj.weight"]
                out[p + "gate_up"] = [p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"]
                out[p + "down"] = [p + "mlp.down_proj.weight"]
        out["head"] = [None]
        return out

    def _head(self):
        if self.gpt2:
            return self.W.get("lm_head.weight", self.g("wte.weight"))
        head = self.W.get("lm_head.weight")
        return head if head is not None else self.g("model.embed_tokens.weight")

    def _wshape(self, name):
        if name is None:
            name = self._head_name()
        if isinstance(self.W, Weights):
            return self.W.shape(name)
        return self.W[name].shape

    def _head_name(self):
        if self.gpt2:
            return "lm_head.weight" if "lm_head.weight" in self.W else "wte.weight"
        return "lm_head.weight" if "lm_head.weight" in self.W else "model.embed_tokens.weight"

    def _bf16(self, names):
        """[rows, cols] bfloat16 bits (uint16) of these weights stacked by rows, without a float32 copy when the
        checkpoint is already bfloat16."""
        from . import hub_gpu
        parts = []
        for n in names:
            kind, arr = self.W.raw(n) if isinstance(self.W, Weights) else ("f32", self.W[n])
            if kind != "bf16":
                arr = hub_gpu.to_bf16_pairs(np.asarray(arr, np.float32)).view(np.uint16).reshape(arr.shape)
            parts.append(arr)
        return parts[0] if len(parts) == 1 else np.concatenate(parts, 0)

    def _weight(self, name):
        """[out, in] for x @ W.T (GPT-2 stores [in, out])."""
        if name is None:
            return self._head()
        w = self.g(name)
        return w.T if self.gpt2 else w

    def to_device(self, device="auto", pending=None, gpu_layers=None):
        """Put the model on a GPU when one is usable. Returns (device name, reason when it is the processor).

        device="auto": the GPU when there is one; a model larger than the card's free memory runs its first layers
        on the GPU and the rest on the processor (from RAM, or from the SSD when the weights are memory-mapped).
        device="cuda": all of it on the GPU, or an error. gpu_layers=N: exactly N layers on the GPU.
        pending: a hub_gpu.prepare(device) started earlier (the GPU starts up while the weights are read)."""
        self.release()
        from . import hub_gpu
        strict = str(device).lower().startswith(("cuda", "gpu"))
        r, why = pending.result() if pending is not None else hub_gpu.runner(device)
        if r is None:
            if strict:
                raise ModelError(f"no usable GPU: {why}")
            return "cpu", why
        groups = self._groups()
        need = sum(int(np.prod(self._wshape(n))) * 2 for names in groups.values() for n in names)
        full_ok = getattr(r, "full", False) and self._full_capable()
        if not full_ok and need > 0.9 * r.dev.memory:
            msg = f"the model needs {need / 2 ** 30:.1f} GB on the GPU, which has {r.dev.memory / 2 ** 30:.1f} GB"
            r.close()
            if strict:
                raise ModelError(msg)
            return "cpu", msg
        try:
            placed = self._upload_full(r, strict, gpu_layers) if full_ok else 0
            if placed == 0 and full_ok:
                r.close()
                msg = "not even one layer of the model fits in the GPU's free memory"
                if strict:
                    raise ModelError(msg)
                return "cpu", msg
            if not full_ok:
                for key, names in groups.items():
                    ws = [self._weight(n) for n in names]
                    m = r.upload(ws[0] if len(ws) == 1 else np.concatenate(ws, 0))
                    if m is None:
                        continue
                    self._gm[key] = (m, list(np.cumsum([w.shape[0] for w in ws])[:-1]))
        except Exception as e:
            r.close()
            self._gm = {}
            self._gl = None
            if str(device).lower().startswith(("cuda", "gpu")):
                raise ModelError(f"could not put the model on the GPU: {e}")
            return "cpu", f"{type(e).__name__}: {e}"
        self.gpu = r
        name = f"cuda:{r.dev.index} ({r.name})"
        ng = self._ng if self._gl is not None else (self.L if self._gm else 0)
        self.plan = {"gpu": {"name": getattr(r, "name", "gpu"), "layers": ng, "of": self.L,
                             "bytes": int(getattr(r, "used", 0) or 0),
                             "free_after": _free_of(r),
                             "tensor_cores": bool(getattr(r, "tc", None)), "kv_cache": self.kv_cache,
                             "weights": getattr(self, "gpu_weights_used", "bf16")}}
        if self._gl is not None and self._ng < self.L:
            _progress(f"{self._ng} of {self.L} layers on {getattr(r, 'name', 'the GPU')} "
                      f"({int(getattr(r, 'used', 0) or 0) / 2 ** 30:.1f} GB), {self.L - self._ng} on the processor")
            return f"{name} + cpu ({self._ng} of {self.L} layers on the GPU)", \
                f"the model is larger than the GPU's free memory: {self.L - self._ng} layers run on the processor"
        return name, ""

    def release(self):
        self.__dict__.pop("_last", None)
        if self.gpu is not None:
            self.gpu.close()
        self.gpu = None
        self._gm = {}
        self._gl = None

    @property
    def full(self):
        """True while the whole network (weights, cache, attention) runs on the GPU."""
        return self.gpu is not None and self._gl is not None

    def _act_kind(self):
        name = (self.cfg.get("hidden_activation") or self.cfg.get("hidden_act") or "silu").lower()
        if name in ("silu", "swish"):
            return 0
        if name in ("gelu_new", "gelu_pytorch_tanh", "gelu_fast", "gelu_tanh"):
            return 1
        return None

    def _full_capable(self):
        from .hub_gpu import MAX_DH
        if self.gpt2 or self._act_kind() is None or self.nh % self.kvh or self.dh > MAX_DH or self.dh % 4:
            return False
        names = [self._head_name()]
        for i in range(self.L):
            p = f"model.layers.{i}."
            names += [p + f"self_attn.{n}_proj.weight" for n in "qkv"] + [
                p + "self_attn.o_proj.weight", p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight",
                p + "mlp.down_proj.weight"]
        return all(n in self.W and len(self._wshape(n)) == 2 and self._wshape(n)[1] % 2 == 0 for n in names)

    def _layer_bytes(self, i):
        p = f"model.layers.{i}."
        return 2 * sum(int(np.prod(self._wshape(p + n))) for n in (
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"))

    def _layer_values(self, i):
        return self._layer_bytes(i) // 2

    def _head_tied(self):
        return self._head_name() in ("model.embed_tokens.weight", "wte.weight")

    def _upload_full(self, r, strict=False, gpu_layers=None):
        """The weights on the GPU for the whole-network path: all layers when they fit, else the first ones that do
        (the rest run on the processor). Returns the number of layers placed.

        The weight format is chosen to keep the whole model on the card (APSI_GPU_WEIGHTS, or gpu_weights=):
        "auto" takes the first of bfloat16, int8 and int4 in which every layer fits, and int4 with as many
        layers as fit when none does (a 14B model on a 12 GB card: all 48 layers in int4 instead of 16 in bfloat16,
        which makes it about 20x faster). "bf16", "int8", "int4" force one."""
        from .hub_gpu import QBYTES
        W = self.W
        mode = str(getattr(self, "gpu_weights", None) or os.environ.get("APSI_GPU_WEIGHTS", "auto")).strip().lower()
        mode = {"q8": "int8", "q4": "int4", "fp16": "bf16", "bfloat16": "bf16", "e4m3": "fp8", "fp8_e4m3": "fp8",
                "e2m1": "fp4", "mxfp4": "fp4"}.get(mode, mode)
        if mode not in ("auto", "bf16", "int8", "int4", "fp8", "fp4"):
            raise ModelError(f"gpu_weights must be auto, bf16, int8, int4, fp8 or fp4 (got '{mode}')")
        head_vals = int(np.prod(self._wshape(None)))
        free = r.free_memory()

        def fit(kind, kv_tokens):
            kv_layer = 2 * (2 if self.kv_cache == "fp16" else 4) * self.kvh * self.dh * kv_tokens
            hk = kind if not self._head_tied() else "bf16"      # a tied head is also the embedding table
            head = head_vals * QBYTES[hk]
            gu = 2 * int(np.prod(self._wshape("model.layers.0.mlp.gate_proj.weight")))
            scratch = 0 if kind == "bf16" else 2 * gu               # the largest matrix, dequantized for prompts
            budget = free - head - scratch - (384 << 20)        # the cache of a layer counts with that layer
            n, used = 0, 0
            for i in range(self.L):
                b = self._layer_values(i) * QBYTES[kind] + kv_layer
                if used + b > budget:
                    break
                used += b
                n += 1
            return n, kv_layer
        kinds = [mode] if mode != "auto" else ["bf16", "int8", "int4"]
        choice = None
        for kind in kinds:
            for kv_tokens in sorted({min(self.ctx, 8192), min(self.ctx, 4096)}, reverse=True):
                n, kv_layer = fit(kind, kv_tokens)
                if n == self.L:
                    choice = (kind, n, kv_layer)
                    break
            if choice:
                break
        if choice is None:                                     # nothing fits whole: the most layers on the card
            kind = kinds[-1]
            n, kv_layer = fit(kind, min(self.ctx, 8192))
            choice = (kind, n, kv_layer)
        kind, n, kv_layer = choice
        if any(self._wshape(f"model.layers.0.{w}")[1] % 32 for w in ("self_attn.q_proj.weight", "mlp.down_proj.weight")):
            kind = "bf16"                                      # widths the int formats cannot take
            n, kv_layer = fit("bf16", min(self.ctx, 8192))
        kinds_of = [kind] * n
        if mode == "auto" and kind == "int4" and n == self.L:
            # the room left over turns layers back into int8, from the first (the most sensitive) on
            hk = "int4" if not self._head_tied() else "bf16"
            gu = 2 * int(np.prod(self._wshape("model.layers.0.mlp.gate_proj.weight")))
            spare = free - head_vals * QBYTES[hk] - 2 * gu - (384 << 20) - \
                sum(self._layer_values(i) * QBYTES["int4"] + kv_layer for i in range(n))
            for i in range(n):
                extra = self._layer_values(i) * (QBYTES["int8"] - QBYTES["int4"])
                if spare < extra:
                    break
                spare -= extra
                kinds_of[i] = "int8"
        self.gpu_weights_used = kind if len(set(kinds_of)) <= 1 else \
            f"int8 ({kinds_of.count('int8')} layers) + int4 ({kinds_of.count('int4')} layers)"
        kv_bytes = kv_layer * self.L
        if gpu_layers is not None:
            n = max(0, min(int(gpu_layers), self.L))
        elif strict and n < self.L:
            need = (sum(self._layer_values(i) for i in range(self.L)) * QBYTES[kind] + head_vals * 2 + kv_bytes) \
                / 2 ** 30
            raise ModelError(f"the model needs about {need:.1f} GB on the GPU, which has {free / 2 ** 30:.1f} GB free"
                             " (device=\"auto\" puts what fits on the GPU and the rest on the processor)")
        if n == 0:
            return 0
        if kind != "bf16":
            _progress(f"weights on the GPU in {self.gpu_weights_used} (to fit the card; APSI_GPU_WEIGHTS=bf16 keeps "
                      f"bfloat16): {n} of {self.L} layers")
        def up_for(k_):
            return (lambda ws: r.qmatrix(ws, k_)) if k_ != "bf16" else r.matrix
        up = up_for(kind)
        layers = []
        for i in range(n):
            p = f"model.layers.{i}."
            qkv = [p + f"self_attn.{x}_proj.weight" for x in "qkv"]
            up = up_for(kinds_of[i])
            L = {"qkv": up(self._bf16(qkv)), "o": up(self._bf16([p + "self_attn.o_proj.weight"])),
                 "gu": up(self._bf16([p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"])),
                 "down": up(self._bf16([p + "mlp.down_proj.weight"]))}
            bias = [W.get(p + f"self_attn.{x}_proj.bias") for x in "qkv"]
            L["qkv_b"] = r.vec(np.concatenate([b if b is not None else np.zeros(self._wshape(w)[0], np.float32)
                                               for b, w in zip(bias, qkv)])) if any(b is not None for b in bias) else None
            ob = W.get(p + "self_attn.o_proj.bias")
            L["o_b"] = r.vec(ob) if ob is not None else None
            L["ln1"] = r.vec(self.g(p + "input_layernorm.weight"))
            L["ln2"] = r.vec(self.g(p + "post_attention_layernorm.weight"))
            qn = W.get(p + "self_attn.q_norm.weight")
            L["qn"] = r.vec(qn) if qn is not None else None
            L["kn"] = r.vec(W[p + "self_attn.k_norm.weight"]) if qn is not None else None
            L["I"] = self._wshape(p + "mlp.down_proj.weight")[1]
            layers.append(L)
        self._gnorm = r.vec(self.g("model.norm.weight"))
        self._ghead = up_for(kind)(self._bf16([self._head_name()])) if not self._head_tied() else \
            r.matrix(self._bf16([self._head_name()]))
        self._gl = layers
        self._ng = n
        return n

    def _prepare_cpu(self):
        """bfloat16 matrices stacked like on the GPU, for the native product on the processor (half the memory
        and time of float32). Only for bfloat16 checkpoints; otherwise the float32 path stays."""
        self._cm = {}
        self._cm_ready = True
        if self.gpt2 or not isinstance(self.W, Weights):
            return False
        from . import hub_cpu
        groups = self._groups()
        if self.gpu is not None:                            # split: only the layers that run on the processor
            on_gpu = set(self._gm)
            if self._gl is not None:
                on_gpu |= {k for k in groups if k == "head" or (k.startswith("model.layers.") and
                                                               int(k.split(".")[2]) < self._ng)}
            groups = {k: v for k, v in groups.items() if k not in on_gpu}
        names = [n for ns in groups.values() for n in ns]
        if not all(self.W.raw_kind(n if n else self._head_name()) == "bf16" for n in names):
            return False
        if not hub_cpu.available():
            return False
        mode = os.environ.get("APSI_CPU_WEIGHTS", "int8").strip().lower()
        q8 = mode in ("int8", "q8", "auto") and hub_cpu.q8_available() and \
            all(self._wshape(n if n else self._head_name())[1] % hub_cpu.QB == 0 for n in names)
        self.cpu_weights = "int8" if q8 else "bf16"
        store = _Q8Store(self, groups) if q8 else None
        for key, ns in groups.items():
            ns = [n if n else self._head_name() for n in ns]
            cuts = list(np.cumsum([self._wshape(n)[0] for n in ns])[:-1])
            self._cm[key] = (store.get(key, ns) if q8 else self._bf16(ns), cuts)
        if store is not None:
            store.done()
        self._plan_cpu(groups, q8)
        return True

    def _plan_cpu(self, groups, q8):
        """Where the processor's share of the weights lives, and why (self.plan["cpu"])."""
        values = sum(int(np.prod(self._wshape(n if n else self._head_name()))) for ns in groups.values() for n in ns)
        size = values * (1.13 if q8 else 2)
        total, avail = _ram_available()
        mapped = q8 and all(isinstance(w[0].codes, np.memmap) for w in self._cm.values()
                            if hasattr(w[0], "codes"))
        where = "RAM"
        note = ""
        if avail and size > 0.8 * avail:
            where = "SSD, memory-mapped" if mapped else "RAM (more than is free: the system may swap)"
            note = (f"the processor's {size / 2 ** 30:.1f} GB of weights are more than 80 % of the free RAM "
                    f"({avail / 2 ** 30:.1f} GB): tokens are limited by the SSD's read speed")
            _progress(note)
        if self.gpu is not None or size > 2 ** 30:
            _progress(f"processor: {len(groups)} weight groups, {'int8' if q8 else 'bfloat16'}, "
                      f"{size / 2 ** 30:.1f} GB in {where}")
        self.plan = dict(getattr(self, "plan", {}) or {})
        self.plan["cpu"] = {"layers": sorted({int(k.split(".")[2]) for k in groups if k.startswith("model.layers.")}),
                            "weights": "int8" if q8 else "bf16", "bytes": int(size), "in": where,
                            "ram_total": total, "ram_available": avail, "note": note}

    def _mm(self, x, key, names):
        """[x @ W.T for each named weight]; one fused product on the GPU."""
        if self.gpu is not None and key not in self._gm and self.__dict__.get("_gl") is not None:
            y = self._mm_streamed(x, key, names)            # a long prompt in a split model: on the GPU after all
            if y is not None:
                return y
        if self.gpu is None or key not in self._gm:         # the processor: also the layers of a split model
            if self.__dict__.get("_cm_for") is not self.gpu or not self.__dict__.get("_cm_ready"):
                self._cm_for = self.gpu
                self._prepare_cpu()
            if key in self._cm:
                from . import hub_cpu
                w, cuts = self._cm[key]
                y = hub_cpu.matmul(x, w)
                return np.split(y, cuts, -1) if cuts else [y]
        if self.gpu is not None and key in self._gm:
            m, cuts = self._gm[key]
            try:
                y = self.gpu.matmul(x, m)
                return np.split(y, cuts, -1) if cuts else [y]
            except Exception as e:                          # a driver error: carry on on the processor
                _gpu_gave_up(e)
                self.release()
        return [x @ self._weight(n).T for n in names]

    PROMPT_ON_GPU = 64

    def _mm_streamed(self, x, key, names):
        """A prompt of many tokens through a layer that lives on the processor: its bfloat16 weights go to the GPU
        for this product (about 25 GB/s over PCIe 4.0) and the tensor cores do the rest — a 512-token prompt
        through 32 processor layers of a 14B model takes seconds instead of a minute. Decoding (one token at a
        time) stays on the processor, where reading the weights once is cheaper than sending them."""
        rows = int(np.prod(x.shape[:-1]))
        if rows < self.PROMPT_ON_GPU or os.environ.get("APSI_PROMPT_ON_GPU", "1").strip() in ("0", "false", "no"):
            return None
        if self.__dict__.get("_stream_off"):
            return None
        try:
            ns = [n if n else self._head_name() for n in names]
            if self.gpt2:
                return None
            bits = self._bf16(ns)
            out = sum(int(self._wshape(n)[0]) for n in ns)
            need = bits.nbytes + rows * (bits.shape[1] + out) * 4 + (128 << 20)
            free = _free_of(self.gpu)
            if free is not None and free < need:
                return None
            m = self.gpu.matrix(bits)
            if m is None:
                return None
            try:
                y = self.gpu.matmul_rows(x, m)
            finally:
                self.gpu.free(m.buf)
            cuts = list(np.cumsum([self._wshape(n)[0] for n in ns])[:-1])
            return np.split(y, cuts, -1) if cuts else [y]
        except Exception as e:                              # anything odd: the processor path, from now on
            self._stream_off = True
            _progress(f"long prompts stay on the processor ({type(e).__name__}: {e})")
            return None

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
        elif kind == "yarn":                          # long contexts (Qwen2.5 to 128k, DeepSeek, …)
            factor = float(sc["factor"])
            old = sc.get("original_max_position_embeddings") or cfg.get("max_position_embeddings", 4096) / factor
            fast, slow = float(sc.get("beta_fast", 32)), float(sc.get("beta_slow", 1))

            def corr(n_rot):
                return rot * math.log(old / (n_rot * 2 * math.pi)) / (2 * math.log(base))
            lo_, hi_ = max(math.floor(corr(fast)), 0), min(math.ceil(corr(slow)), rot - 1)
            if lo_ == hi_:
                hi_ += 0.001
            ramp = np.clip((np.arange(rot // 2, dtype=np.float64) - lo_) / (hi_ - lo_), 0, 1)
            keep = 1 - ramp                                 # 1: extrapolate (high frequencies), 0: interpolate
            inv = inv / factor * (1 - keep) + inv * keep
            att = sc.get("attention_factor")
            self.rope_scale = float(att) if att else (0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0)
        elif kind not in (None, "default"):
            raise ModelError(f"rope scaling '{kind}' is not supported natively (supported: llama3, linear, yarn)")
        self.inv_freq = inv
        self.rot = rot

    def _rope(self, x, pos):
        f = np.outer(pos, self.inv_freq)                           # [T, rot/2]
        emb = np.concatenate([f, f], -1)
        m = getattr(self, "rope_scale", 1.0)
        cos, sin = (np.cos(emb) * m).astype(np.float32), (np.sin(emb) * m).astype(np.float32)
        r, rest = x[..., :self.rot], x[..., self.rot:]
        half = self.rot // 2
        rh = np.concatenate([-r[..., half:], r[..., :half]], -1)
        out = r * cos + rh * sin
        return np.concatenate([out, rest], -1) if rest.shape[-1] else out

    def new_cache(self):
        if self.full:
            return _GpuCache(self)
        return [[None, None] for _ in range(self.L)]

    CHUNK = 512                       # prompt positions per pass on the GPU (most; fewer when the context is long)
    SCRATCH = 256 << 20               # bytes of attention scores per pass at most

    def _forward_gpu(self, ids, cache, start, last, fetch=True):
        r = self.gpu
        ids = np.atleast_2d(np.asarray(ids, np.int64))
        Tall = ids.shape[1]
        cache.ensure(start + Tall)
        r.begin()
        H, nh, kvh, dh = self.H, self.nh, self.kvh, self.dh
        width = (nh + 2 * kvh) * dh
        half = self.rot // 2
        eps, off = self.eps, (1.0 if self.gemma else 0.0)
        sc = self.cfg.get("query_pre_attn_scalar")
        scale = 1.0 / math.sqrt(sc) if sc else 1.0 / math.sqrt(dh)
        kind = self._act_kind()
        V = self._ghead.rows
        outs = []
        c0 = 0
        while c0 < Tall:
            T = min(self.CHUNK, Tall - c0)
            while T > 64 and 4 * self.nh * T * (start + c0 + T) > self.SCRATCH:
                T //= 2
            c = ids[0, c0:c0 + T]
            s0 = start + c0
            c0 += T
            if c.size and (int(c.min()) < 0 or int(c.max()) >= V):
                raise ModelError(f"token id {int(c.max()) if c.max() >= V else int(c.min())} is outside the "
                                 f"vocabulary of {V} tokens")
            X = r.ws("x", T * H)
            if self._head_name() == "model.embed_tokens.weight" and hasattr(r, "embed"):
                r.embed(self._ghead, c, X, math.sqrt(H) if self.gemma else 1.0)   # tied: the head is the table
            else:
                x = self.W.rows("model.embed_tokens.weight", c) if isinstance(self.W, Weights) \
                    else self.g("model.embed_tokens.weight")[c]
                if self.gemma:
                    x = x * np.float32(math.sqrt(H))
                r.put(X, np.asarray(x, np.float32).reshape(-1))
            COS, SIN = cache.cos, cache.sin
            QKV = r.ws("qkv", T * width)
            Q, O = r.ws("q", T * nh * dh), r.ws("o", T * nh * dh)
            S = r.ws("s", nh * T * (s0 + T))
            if self._ng < self.L:                      # the layers that did not fit run on the processor
                pos = np.arange(s0, s0 + T)
                causal = np.triu(np.full((T, s0 + T), -1e9, np.float32), k=s0 + 1)
                if self.window and s0 + T > self.window:
                    causal = causal + np.where(pos[:, None] - np.arange(s0 + T)[None, :] >= self.window,
                                               np.float32(-1e9), 0)
            for i, Ld in enumerate(self._gl):
                I = Ld["I"]
                GU, M = r.ws("gu", T * 2 * I), r.ws("m", T * I)
                r.linear(Ld["qkv"], X, QKV, T, bias=Ld["qkv_b"], ln=Ld["ln1"], eps=eps, off=off)
                r.qkv_prep(QKV, Q, cache.K[i], cache.V[i], COS, SIN, Ld["qn"], Ld["kn"], T, nh, kvh, dh, self.rot,
                           s0, cache.cap, eps, half=cache.half)
                r.attention(Q, cache.K[i], cache.V[i], O, S, T, nh, kvh, dh, s0, cache.cap, self.window, scale,
                            half=cache.half)
                r.linear(Ld["o"], O, X, T, bias=Ld["o_b"], acc=True)
                r.linear(Ld["gu"], X, GU, T, ln=Ld["ln2"], eps=eps, off=off)
                r.act_mul(GU, M, T, I, kind)
                r.linear(Ld["down"], M, X, T, acc=True)
            if self._ng < self.L:
                x = r.get(X, T * H).reshape(1, T, H)
                for i in range(self._ng, self.L):
                    x = self._layer(i, x, cache.cpu, pos, causal, T)
                r.put(X, np.ascontiguousarray(x, np.float32).reshape(-1))
            if last and c0 < Tall:
                continue
            rows, first = (1, T - 1) if last else (T, 0)
            LOG = r.ws("log", rows * V)
            r.linear(self._ghead, X, LOG, rows, ln=self._gnorm, eps=eps, off=off, xoff=first * H)
            if not fetch and last:
                return None                         # the scores stay on the card (sampling there)
            outs.append(r.get(LOG, rows * V).reshape(rows, V))
        logits = np.concatenate(outs)[None]
        cap = self.cfg.get("final_logit_softcapping")
        if cap:
            logits = np.tanh(logits / cap) * cap
        return logits

    def _layer(self, i, x, cache, pos, causal, T):
        """One transformer block on the processor (cache[i] holds its keys and values)."""
        W = self.W
        if self.gpt2:
            p = f"h.{i}."
            h = _ln(x, self.g(p + "ln_1.weight"), self.g(p + "ln_1.bias"), self.eps)
            qkv = self._mm(h, p + "attn.c_attn", [p + "attn.c_attn.weight"])[0] + self.g(p + "attn.c_attn.bias")
            q, k, v = np.split(qkv, 3, axis=-1)
        else:
            p = f"model.layers.{i}."
            h = _rms(x, self.g(p + "input_layernorm.weight"), self.eps, 1.0 if self.gemma else 0.0)
            q, k, v = self._mm(h, p + "qkv", [p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                                              p + "self_attn.v_proj.weight"])
            for name in ("q", "k", "v"):
                bias = W.get(p + f"self_attn.{name}_proj.bias")
                if bias is not None:
                    if name == "q":
                        q = q + bias
                    elif name == "k":
                        k = k + bias
                    else:
                        v = v + bias
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
            x = x + (self._mm(o, p + "attn.c_proj", [p + "attn.c_proj.weight"])[0] + self.g(p + "attn.c_proj.bias"))
            h = _ln(x, self.g(p + "ln_2.weight"), self.g(p + "ln_2.bias"), self.eps)
            m = _gelu_tanh(self._mm(h, p + "mlp.c_fc", [p + "mlp.c_fc.weight"])[0] + self.g(p + "mlp.c_fc.bias"))
            x = x + (self._mm(m, p + "mlp.c_proj", [p + "mlp.c_proj.weight"])[0] + self.g(p + "mlp.c_proj.bias"))
        else:
            o = self._mm(o, p + "o", [p + "self_attn.o_proj.weight"])[0]
            ob = W.get(p + "self_attn.o_proj.bias")
            x = x + (o + ob if ob is not None else o)
            h = _rms(x, self.g(p + "post_attention_layernorm.weight"), self.eps, 1.0 if self.gemma else 0.0)
            gate, up = self._mm(h, p + "gate_up", [p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"])
            x = x + self._mm(self.act(gate) * up, p + "down", [p + "mlp.down_proj.weight"])[0]
        return x

    def forward(self, ids, cache, start, last=False, fetch=True):
        """Logits for ids [1, T] given `start` tokens already in the cache (which is updated).
        last=True: only for the last position (all that generation needs)."""
        if isinstance(cache, _GpuCache):
            with self._lock:
                return self._forward_gpu(ids, cache, start, last, fetch)
        ids = np.atleast_2d(np.asarray(ids, np.int64))
        T = ids.shape[1]
        pos = np.arange(start, start + T)
        W = self.W
        if self.gpt2:
            x = self.g("wte.weight")[ids] + self.g("wpe.weight")[pos]
        else:
            x = self.W.rows("model.embed_tokens.weight", ids) if isinstance(self.W, Weights) \
                else self.g("model.embed_tokens.weight")[ids]
            if self.gemma:
                x = x * np.float32(math.sqrt(self.H))
        total = start + T
        causal = np.triu(np.full((T, total), -1e9, np.float32), k=start + 1)
        if self.window and total > self.window:
            qpos = pos[:, None]
            kpos = np.arange(total)[None, :]
            causal = causal + np.where(qpos - kpos >= self.window, np.float32(-1e9), 0)
        for i in range(self.L):
            x = self._layer(i, x, cache, pos, causal, T)
        if last:
            x = x[:, -1:]
        if self.gpt2:
            x = _ln(x, self.g("ln_f.weight"), self.g("ln_f.bias"), self.eps)
        else:
            x = _rms(x, self.g("model.norm.weight"), self.eps, 1.0 if self.gemma else 0.0)
        logits = self._mm(x, "head", [None])[0]
        cap = self.cfg.get("final_logit_softcapping")
        if cap:
            logits = np.tanh(logits / cap) * cap
        return logits

    def generate(self, ids, max_new_tokens=64, temperature=0.0, top_k=0, top_p=1.0, eos=None, seed=None,
                 repetition_penalty=1.0, stream=None):
        """Token ids after `ids`. On a GPU, a failure part way is recovered from: out of memory frees the scratch
        space and goes on in smaller steps; any other driver error moves the model to the processor, which carries on
        from the tokens already produced (stream sees every token once)."""
        ids = list(np.asarray(ids).reshape(-1))
        out = []
        with self._lock:
            tries = 0
            while True:
                more = []
                try:
                    self._generate(ids + out, more, max_new_tokens - len(out), temperature, top_k, top_p, eos, seed,
                                   repetition_penalty, stream)
                    out += more
                    return out
                except Exception as e:
                    out += more
                    if self.gpu is None or isinstance(e, ModelError):
                        raise
                    tries += 1
                    if _out_of_memory(e) and tries == 1:
                        self.gpu.trim()                    # scratch buffers go; retry on the GPU in small passes
                        self.CHUNK = 64
                        continue
                    _gpu_gave_up(e)
                    self.release()

    @property
    def _lock(self):
        lk = self.__dict__.get("_lk")
        if lk is None:
            import threading
            lk = self.__dict__.setdefault("_lk", threading.RLock())
        return lk

    def _generate(self, ids, out, max_new_tokens, temperature, top_k, top_p, eos, seed, repetition_penalty, stream):
        ids = list(ids)
        if len(ids) + max_new_tokens > self.ctx:
            max_new_tokens = max(0, self.ctx - len(ids))
        rng = np.random.default_rng(seed)
        cache, done = self._reuse(ids)                      # a conversation's earlier turns are already computed
        on_gpu = isinstance(cache, _GpuCache) and bool(temperature) and getattr(self.gpu, "sample_prep", None)
        fed = list(ids)
        self._last = None
        logits = self.forward(np.array([ids[done:]]), cache, done, last=True, fetch=not on_gpu)
        self._last = (cache, fed)
        V = self._ghead.rows if on_gpu else len(logits[0, -1])
        seen = np.zeros(V, bool)                            # tokens the repetition penalty applies to
        seen[[t for t in ids if 0 <= t < len(seen)]] = True
        seen_ids, n_seen = None, [int(seen.sum())]
        if on_gpu:                                          # sampling on the card: only candidates come back
            SEEN = self.gpu.ws("seen", V)
            self.gpu.put(SEEN, seen.astype(np.float32) if repetition_penalty != 1.0 else np.zeros(V, np.float32))
        eos = set([eos] if isinstance(eos, int) else (eos or []))
        for step in range(int(max_new_tokens)):
            nxt = self._sample_gpu(SEEN, V, temperature, top_k, top_p, repetition_penalty, rng) if on_gpu else None
            if nxt is None:
                lg = np.array(self.gpu.get(self.gpu.ws("log", V), V) if on_gpu else logits[0, -1], np.float32)
                if repetition_penalty != 1.0:
                    if seen_ids is None or len(seen_ids) != n_seen[0]:
                        seen_ids = np.flatnonzero(seen)
                    s = lg[seen_ids]
                    lg[seen_ids] = np.where(s > 0, s / repetition_penalty, s * repetition_penalty)
                if not temperature:
                    nxt = int(np.argmax(lg))
                else:
                    nxt = _sample(lg, temperature, top_k, top_p, rng)
            if nxt in eos:
                break
            out.append(nxt)
            if nxt < len(seen) and not seen[nxt]:
                seen[nxt] = True
                n_seen[0] += 1
                if on_gpu and repetition_penalty != 1.0:
                    self.gpu.put(SEEN, np.ones(1, np.float32), nxt)
            if stream is not None:
                stream(nxt)
            if step + 1 == int(max_new_tokens):
                break                                       # the last token needs no forward pass
            self._last = None
            logits = self.forward(np.array([[nxt]]), cache, len(fed), last=True, fetch=not on_gpu)
            fed.append(nxt)
            self._last = (cache, fed)

    def _reuse(self, ids):
        """(cache, positions in it to keep): the cache of the last generation when it starts with the same tokens
        (a conversation re-sends its earlier turns), else a new one. At least one token is always recomputed."""
        last = self.__dict__.get("_last")
        if last is not None:
            cache, fed = last
            if isinstance(cache, _GpuCache) == self.full and (not isinstance(cache, _GpuCache) or cache.r is self.gpu):
                n = 0
                lim = min(len(fed), len(ids) - 1)
                while n < lim and fed[n] == ids[n]:
                    n += 1
                if n >= 16:
                    if isinstance(cache, _GpuCache):
                        cache.cpu = [[k[:, :, :n] if k is not None else None, v[:, :, :n] if v is not None else None]
                                     for k, v in cache.cpu]
                    else:
                        cache = [[k[:, :, :n] if k is not None else None, v[:, :, :n] if v is not None else None]
                                 for k, v in cache]
                    return cache, n
        return self.new_cache(), 0

    DELTA = 16.0                     # candidates: within e^16 of the most likely token (after the temperature)
    CANDIDATES = 4096

    def _sample_gpu(self, SEEN, V, temperature, top_k, top_p, rp, rng):
        """The next token from the logits left on the GPU, or None when the processor must look at all of them
        (a very flat distribution). Same choice rule as _sample."""
        g, Z, ids, vals, count = self.gpu.sample_prep(self.gpu.ws("log", V), SEEN, V, float(rp), 1.0 / temperature,
                                                      self.DELTA, self.CANDIDATES)
        if count > self.CANDIDATES or count == 0:
            return None
        e = np.exp((vals.astype(np.float64) - g))
        order = np.lexsort((ids, -e))                       # most likely first; ties by id (reproducible)
        ids, e = ids[order], e[order]
        if top_k and int(top_k) < V:
            if count < int(top_k):
                return None
            ids, e = ids[:int(top_k)], e[:int(top_k)]
            Z = float(e.sum())
        if top_p < 1.0:
            c = np.cumsum(e) / Z
            if c[-1] < top_p:
                return None
            keep = max(1, int(np.searchsorted(c, top_p)) + 1)
            ids, e = ids[:keep], e[:keep]
        elif not top_k and e.sum() < Z * (1 - 1e-6):        # plain temperature sampling needs every token
            return None
        w = np.cumsum(e)
        return int(ids[min(np.searchsorted(w, rng.random() * w[-1], side="right"), ids.size - 1)])


def _out_of_memory(e):
    t = str(e).upper()
    return "OUT_OF_MEMORY" in t or "OUT OF MEMORY" in t or "CUDA ERROR 2:" in t or "CUDA ERROR 2 " in t


_WARNED = set()


def _gpu_gave_up(e):
    """Say once per kind of failure that the model moved to the processor (APSI_QUIET=1 says nothing)."""
    import sys
    key = type(e).__name__ + str(e)[:80]
    if key in _WARNED or os.environ.get("APSI_QUIET", "").strip() in ("1", "true", "yes"):
        return
    _WARNED.add(key)
    print(f"apsi hub: the graphics card stopped working ({e}); the model carries on on the processor.",
          file=sys.stderr)


def _sample(lg, temperature, top_k, top_p, rng):
    """One token from logits with temperature, top-k and nucleus (top-p) filtering. Only the few hundred most likely
    tokens are sorted (a full sort of a 150,000-token vocabulary costs more than a GPU forward pass); when they do
    not reach top_p, everything is. float32 throughout except the running sums."""
    lg = np.asarray(lg, np.float32)
    V = lg.size
    inv = np.float32(1.0 / temperature)
    pool = None
    if top_k and int(top_k) < V:
        kk = int(top_k)
        pool = np.argpartition(lg, V - kk)[V - kk:]
        kth = lg[pool].min()
        pool = np.flatnonzero(lg >= kth)                       # ties at the edge stay in, as before
    mx = lg.max()
    if top_p < 1.0 or pool is not None:
        Z = None if pool is not None else float(np.exp((lg - mx) * inv).sum(dtype=np.float64))
        n = 256
        while True:
            cand = pool
            if cand is None:
                cand = np.argpartition(lg, V - n)[V - n:] if n < V else np.arange(V)
            e = np.exp((lg[cand] - mx) * inv)
            if Z is None:
                Z = float(e.sum(dtype=np.float64))
            order = np.argsort(-e, kind="stable")
            cand, e = cand[order], e[order]
            c = np.cumsum(e, dtype=np.float64) / Z
            if top_p >= 1.0 or c[-1] >= top_p or pool is not None or cand.size >= V:
                break
            n = min(V, n * 8)                          # a flat distribution: look at more of the tail
        if top_p < 1.0:
            keep = max(1, int(np.searchsorted(c, top_p)) + 1)
            cand, e = cand[:keep], e[:keep]
        w = np.cumsum(e, dtype=np.float64)
        return int(cand[min(np.searchsorted(w, rng.random() * w[-1], side="right"), cand.size - 1)])
    e = np.exp((lg - mx) * inv)
    w = np.cumsum(e, dtype=np.float64)
    return int(min(np.searchsorted(w, rng.random() * w[-1], side="right"), V - 1))


class _GpuCache:
    """Keys and values of every layer in GPU memory, [kv heads, capacity, head dim], growing as needed."""

    def __init__(self, dec):
        self.dec = dec
        self.r = dec.gpu
        self.cap = 0
        self.K, self.V = [], []
        self.cpu = [[None, None] for _ in range(dec.L)]       # layers that run on the processor
        self.cos = self.sin = None
        self.rope_n = 0
        self.half = dec.kv_cache == "fp16"                    # float16 keys and values: half the memory

    def _rope(self, need):
        """cos / sin of every position up to the cache's capacity, on the card once (not per token)."""
        d = self.dec
        n = min(d.ctx, max(need, 2 * self.cap, 256))
        if n <= self.rope_n:
            return
        half = max(d.rot // 2, 1)
        f = np.outer(np.arange(n), d.inv_freq) if d.rot // 2 else np.zeros((n, 1))
        for name in ("cos", "sin"):
            old = getattr(self, name)
            if old is not None:
                self.r.free(old)
            buf = self.r.alloc(n * half)
            m = getattr(d, "rope_scale", 1.0)
            self.r.put(buf, ((np.cos(f) if name == "cos" else np.sin(f)) * m).astype(np.float32).reshape(-1))
            setattr(self, name, buf)
        self.rope_n = n

    def ensure(self, need):
        if need <= self.cap:
            return
        self._rope(need)
        d = self.dec
        if need > d.ctx:
            raise ModelError(f"the text is longer than the model's window ({d.ctx} tokens)")
        new = min(d.ctx, max(need, 2 * self.cap, 256))
        r, kvh, dh = self.r, d.kvh, d.dh
        K, V = [], []
        for i in range(d._ng):
            k, v = r.alloc_kv(kvh * new * dh, self.half), r.alloc_kv(kvh * new * dh, self.half)
            if self.cap:
                for h in range(kvh):
                    r.copy_kv(k, h * new * dh, self.K[i], h * self.cap * dh, self.cap * dh, self.half)
                    r.copy_kv(v, h * new * dh, self.V[i], h * self.cap * dh, self.cap * dh, self.half)
                r.free(self.K[i])
                r.free(self.V[i])
            K.append(k)
            V.append(v)
        self.K, self.V, self.cap = K, V, new

    def free(self):
        for b in self.K + self.V + [x for x in (self.cos, self.sin) if x is not None]:
            try:
                self.r.free(b)
            except Exception:
                pass
        self.K, self.V, self.cap = [], [], 0

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


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


def load_folder(folder, mmap=False):
    """(config, Weights): the checkpoint read in its stored format (see Weights)."""
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    W = Weights()
    idx = os.path.join(folder, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx, encoding="utf-8") as fh:
            for shard in sorted(set(json.load(fh)["weight_map"].values())):
                read_safetensors_lazy(os.path.join(folder, shard), W, mmap)
    else:
        read_safetensors_lazy(os.path.join(folder, "model.safetensors"), W, mmap)
    return cfg, W
