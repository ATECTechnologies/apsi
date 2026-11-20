"""`use models` — load, save, compress and run AI models.

Native (no extra libraries):
  * safetensors read/write (the format used by Hugging Face, PyTorch, JAX …), memory-mapped:
    opening a 10 GB checkpoint costs almost no RAM until tensors are touched.
  * load straight into a compact format: load_safetensors(path, dtype=int8 / fp8 / int4 / bf16)
    stores weights as QTensors (2–16× smaller than f32) and qlinear() runs them.
  * quantize_model / model_size / dequantize_model for any parameter tree.
  * Apsi checkpoints (.apsim: units, compressed tensors, SHA-256, signatures, sharding) and GGUF
    (llama.cpp / Ollama, bit-exact Q8_0 / Q4_0): save_model / load_model / model_info,
    save_gguf / load_gguf / gguf_info (see modelio.py and docs/MODEL-FILES.md).

Ecosystem bridges (need --allow-python; install the library with pip as usual):
  * onnx_model(path)                     -> onnxruntime (CPU / CUDA / DirectML / TensorRT)
  * hf_pipeline(task, model)             -> Hugging Face transformers pipelines
  * torch_load(path) / torch_module(...) -> PyTorch
"""
import json
import os
import struct

import numpy as np

from . import modelio

from . import ops
from .errors import RuntimeErr
from .precision import QTensor, quantize, dequantize, _bf16_decode, _minifloat_decode, FORMATS
from .units import Quantity
from .values import Record, tree_map, tree_leaves

_ST_TO_NP = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32,
             "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_, "U16": np.uint16,
             "U32": np.uint32, "U64": np.uint64}
_NP_TO_ST = {np.dtype(v): k for k, v in _ST_TO_NP.items()}
_SPECIAL = {"BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1}


# ------------------------------------------------------------------ safetensors
def _read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        if n > 100 * 1024 * 1024:
            raise RuntimeErr("safetensors header is too large (corrupt or malicious file)")
        header = json.loads(fh.read(n).decode("utf-8"))
    return header, 8 + n


def safetensors_info(path):
    """List tensors in a .safetensors file without loading them: {name: (dtype, shape)}."""
    header, _ = _read_header(path)
    return {k: (v["dtype"], tuple(v["shape"])) for k, v in header.items() if k != "__metadata__"}


def safetensors_metadata(path):
    """The text metadata stored in a .safetensors file's header (a dict of strings; empty when there is none)."""
    header, _ = _read_header(path)
    return dict(header.get("__metadata__") or {})


def load_safetensors(path, dtype=None, names=None, nested=True):
    """Load a .safetensors checkpoint (memory-mapped).

    dtype=none keeps the stored precision (BF16/FP8 are decoded to f32 on load);
    dtype=f32/f16 converts; dtype=bf16/fp8_e4m3/int8/int4/fp4_e2m1 … keeps weights as
    compact QTensors (matrices are quantised per output channel).
    nested=true turns "layer.0.weight" into record(layer=[record(weight=...)]) style trees.
    """
    header, start = _read_header(path)
    size = os.path.getsize(path)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    want = None if dtype is None else getattr(dtype, "name", dtype)
    out = {}
    for name, info in header.items():
        if name == "__metadata__" or (names is not None and name not in names):
            continue
        st, shape = info["dtype"], tuple(info["shape"])
        a, b = info["data_offsets"]
        if not (0 <= a <= b and start + b <= size):
            raise RuntimeErr(f"safetensors: tensor '{name}' points outside the file (corrupt or malicious)")
        raw = mm[start + a:start + b]
        if st in _ST_TO_NP:
            arr = raw.view(_ST_TO_NP[st]).reshape(shape)
        elif st == "BF16":
            arr = _bf16_decode(raw.view(np.uint16)).reshape(shape)
        elif st in ("F8_E4M3", "F8_E5M2"):
            fmt = FORMATS["fp8_e4m3" if st == "F8_E4M3" else "fp8_e5m2"]
            arr = _minifloat_decode(fmt, np.asarray(raw)).reshape(shape)
        else:
            raise RuntimeErr(f"safetensors: unsupported dtype {st} for '{name}'")
        out[name] = _convert(arr, want)
    if nested is None:                                   # the flat dict under the exact names
        return out
    return _nest(out) if nested else Record(**{k.replace(".", "_"): v for k, v in out.items()})


def _convert(arr, want):
    if want is None:
        return arr
    if want in ("f64", "fp64"):
        return np.asarray(arr, np.float64)
    if want in ("f32", "fp32"):
        return np.asarray(arr, np.float32)
    if want in ("f16", "fp16"):
        return np.asarray(arr, np.float16)
    if np.asarray(arr).dtype.kind not in "f" or np.ndim(arr) < 2:
        return np.asarray(arr, np.float32)              # keep biases / norms / ints exact
    return quantize(np.asarray(arr, np.float32), want, axis=0 if np.ndim(arr) == 2 else None)


def _nest(flat):
    root = {}
    for name, v in flat.items():
        parts = name.split(".")
        node = root
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v

    def build(d):
        if not isinstance(d, dict):
            return d
        if d and all(k.isdigit() for k in d):
            return [build(d[k]) for k in sorted(d, key=int)]
        return Record(**{k: build(v) for k, v in d.items()})
    return build(root)


def _flatten(tree, prefix=""):
    out = {}
    if isinstance(tree, Record):
        items = tree.fields.items()
    elif isinstance(tree, dict):
        items = tree.items()
    elif isinstance(tree, (list, tuple)):
        items = ((str(i), v) for i, v in enumerate(tree))
    elif hasattr(tree, "fields") and hasattr(tree, "type"):
        items = tree.fields.items()
    else:
        return {prefix.rstrip("."): tree}
    for k, v in items:
        out.update(_flatten(v, f"{prefix}{k}."))
    return out


def _rows_of(rows):
    out = []
    for r in rows:
        r = list(r.fields.values()) if isinstance(r, Record) else list(r)
        name, path = str(r[0]), str(r[1])
        idx = int(r[2]) if len(r) > 2 and r[2] is not None else -1
        tr = bool(r[3]) if len(r) > 3 else False
        r0 = int(r[4]) if len(r) > 4 and r[4] is not None else 0
        r1 = int(r[5]) if len(r) > 5 and r[5] is not None else -1
        out.append((name, path, idx, tr, r0, r1))
    return out


def _at(tree, path):
    node = tree
    for k in path.split("."):
        if isinstance(node, (list, tuple)):
            node = node[int(k)]
        elif isinstance(node, Record):
            node = node.fields[k]
        else:
            node = node[k]
    return node


def named_tensors(params, rows, shapes=None):
    """Another program's names for a parameter tree: rows [name, path, index, transpose, start, end] say that
    tensor `name` (rows start:end of its first axis; end -1 = all) is leaf `path` of the tree (its entry `index`
    of the first axis when index >= 0), transposed when transpose is true. Several rows may fill one tensor (a fused
    weight). shapes (name -> shape) gives a tensor its own shape (e.g. a scalar for a [1] leaf). Returns a dict."""
    shapes = dict(shapes.fields if isinstance(shapes, Record) else (shapes or {}))
    parts = {}
    for name, path, idx, tr, r0, r1 in _rows_of(rows):
        a = np.asarray(ops.raw(_at(params, path)), np.float64)
        if idx >= 0:
            a = a[idx]
        if tr:
            a = a.T
        parts.setdefault(name, []).append((r0, r1, a))
    out = {}
    for name, ps in parts.items():
        if len(ps) == 1 and ps[0][0] == 0 and ps[0][1] < 0:
            a = ps[0][2]
        else:
            ps.sort(key=lambda t: t[0])
            a = np.concatenate([t[2] for t in ps], 0)
        if name in shapes:
            a = a.reshape(tuple(int(v) for v in shapes[name]))
        out[name] = np.array(a, np.float32, order="C")        # (ascontiguousarray would make a scalar 1-d)
    return out


def load_named_tensors(path):
    """Every tensor of a .safetensors file under its exact name (a dict name -> array, memory-mapped)."""
    return load_safetensors(path, nested=None)


def from_named_tensors(like, named, rows):
    """The inverse of named_tensors: a parameter tree shaped like `like`, its leaves taken from `named` (a dict or
    record of name -> array, e.g. load_safetensors(path, nested=false)). A leaf no row names keeps its value from
    `like`; a missing name or a wrong shape is an error that names it."""
    named = dict(named.fields if isinstance(named, Record) else named)
    flat = _flatten(like)
    new = {k: np.array(np.asarray(ops.raw(v), np.float64), copy=True) for k, v in flat.items()}
    for name, path, idx, tr, r0, r1 in _rows_of(rows):
        if name not in named:
            raise RuntimeErr(f"from_named_tensors: '{name}' is not in the file")
        if path not in new:
            raise RuntimeErr(f"from_named_tensors: the model has no parameter '{path}'")
        src = np.asarray(ops.raw(named[name]), np.float64)
        if r0 != 0 or r1 >= 0:
            src = src[r0:(r1 if r1 >= 0 else None)]
        if tr:
            src = src.T
        dst = new[path]
        want = dst[idx].shape if idx >= 0 else dst.shape
        if src.size != int(np.prod(want)) or (src.ndim == len(want) and tuple(src.shape) != tuple(want)):
            raise RuntimeErr(f"from_named_tensors: '{name}' has shape {tuple(src.shape)}, '{path}' needs {want}")
        if idx >= 0:
            dst[idx] = src.reshape(want)
        else:
            new[path] = src.reshape(want)
    def build(t, pre):
        if isinstance(t, Record):
            return Record(**{k: build(v, f"{pre}{k}.") for k, v in t.fields.items()})
        if isinstance(t, dict):
            return {k: build(v, f"{pre}{k}.") for k, v in t.items()}
        if isinstance(t, (list, tuple)):
            return type(t)(build(v, f"{pre}{i}.") for i, v in enumerate(t))
        return new.get(pre.rstrip("."), t)
    return build(like, "")


def save_safetensors(path, params, metadata=None):
    """Save any parameter tree (records, lists, structs) as a .safetensors file."""
    flat = _flatten(params)
    header, chunks, offset = {}, [], 0
    for name, v in flat.items():
        if isinstance(v, QTensor):
            v = dequantize(v)
        if isinstance(v, Quantity):
            v = v.v
        arr = np.ascontiguousarray(np.asarray(v))
        if arr.dtype not in _NP_TO_ST:
            arr = arr.astype(np.float32)
        data = arr.tobytes()
        header[name] = {"dtype": _NP_TO_ST[arr.dtype], "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(data)]}
        chunks.append(data)
        offset += len(data)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in dict(metadata).items()}
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((-len(hb)) % 8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for c in chunks:
            fh.write(c)
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------ compression
def quantize_model(params, dtype=None, min_size=1024):
    """Store every weight matrix (>= min_size elements) in a compact format; others stay as-is."""
    if dtype is None:
        dtype = "int8"
    fmt = getattr(dtype, "name", dtype)

    def q(x):
        if isinstance(x, np.ndarray) and x.dtype.kind == "f" and x.ndim >= 2 and x.size >= min_size:
            return quantize(x, fmt, axis=0 if x.ndim == 2 else None)
        return x
    return tree_map(q, params)


def dequantize_model(params):
    return tree_map(lambda x: dequantize(x) if isinstance(x, QTensor) else x, params)


def model_size(params):
    """Bytes used by all tensors in a parameter tree (counts compact formats at their real size)."""
    total = 0
    for x in tree_leaves(params):
        if isinstance(x, QTensor):
            total += x.nbytes
        elif isinstance(x, Quantity):
            total += np.asarray(x.v).nbytes
        elif isinstance(x, (np.ndarray, np.generic)):
            total += x.nbytes
    return total


def qlinear(x, W, b=None):
    """x @ W (+ b) where W may be a QTensor — decodes, multiplies and accumulates in FP32."""
    Wv = dequantize(W) if isinstance(W, QTensor) else W
    if isinstance(Wv, np.ndarray) and Wv.dtype != np.float32 and isinstance(ops.raw(x), np.ndarray):
        Wv = Wv.astype(np.float32)
    y = x @ Wv
    return y + b if b is not None else y


# ------------------------------------------------------------------ ecosystem bridges
def install(interp):
    from .interop import _need
    from .security import check_path

    def _path(p):
        return check_path(interp, p, "read", "model files")

    def _wpath(p):
        return check_path(interp, p, "write", "saving model files")

    def onnx_model(path, providers=None):
        """Run an ONNX model with onnxruntime: m = onnx_model("net.onnx"); m({"input": x})."""
        _need(interp, "python", "onnx_model()")
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeErr("onnxruntime is not installed (pip install onnxruntime, or onnxruntime-gpu)")
        avail = ort.get_available_providers()
        prov = list(providers) if providers else [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider",
                                                               "DmlExecutionProvider", "CPUExecutionProvider")
                                                  if p in avail]
        sess = ort.InferenceSession(_path(path), providers=prov)
        return OnnxModel(sess)

    def hf_pipeline(task, model=None, device=None, **kw):
        """A Hugging Face transformers pipeline: classify = hf_pipeline("sentiment-analysis")."""
        _need(interp, "python", "hf_pipeline()")
        try:
            from transformers import pipeline
        except ImportError:
            raise RuntimeErr("transformers is not installed (pip install transformers torch)")
        p = pipeline(task, model=model, device=device, **kw)
        from .interop import from_py, to_py

        def call(*args, **kwargs):
            return from_py(_jsonable(p(*[to_py(a) for a in args], **kwargs)))
        call.__name__ = f"hf_pipeline({task})"
        return call

    def torch_load(path, map_location="cpu"):
        _need(interp, "python", "torch_load()")
        try:
            import torch
        except ImportError:
            raise RuntimeErr("PyTorch is not installed (pip install torch)")
        obj = torch.load(_path(path), map_location=map_location, weights_only=True)
        return tree_map(lambda t: t.detach().cpu().numpy() if hasattr(t, "detach") else t,
                        _from_torch_tree(obj))

    interp.modules["models"] = {
        "load_safetensors": lambda path, dtype=None, names=None, nested=True:
            load_safetensors(_path(path), dtype, names, nested),
        "save_safetensors": lambda path, params, metadata=None: save_safetensors(_wpath(path), params, metadata),
        "safetensors_metadata": lambda path: safetensors_metadata(_path(path)),
        "load_named_tensors": lambda path: load_named_tensors(_path(path)),
        "named_tensors": named_tensors, "from_named_tensors": from_named_tensors,
        "safetensors_info": lambda path: Record(**{k.replace(".", "_"): v for k, v in
                                                   safetensors_info(_path(path)).items()}),
        "quantize_model": quantize_model, "dequantize_model": dequantize_model,
        "model_size": model_size, "qlinear": qlinear,
        "onnx_model": onnx_model, "hf_pipeline": hf_pipeline, "torch_load": torch_load,
        # GGUF (llama.cpp / Ollama / LM Studio) and Apsi checkpoints (.apsim) - every file goes through the
        # permission checks, shards and index files included
        "save_gguf": lambda path, params, metadata=None, dtype="f16", name=None, architecture="apsi":
            modelio.save_gguf(_wpath(path), params, metadata, dtype, name, architecture),
        "load_gguf": lambda path, nested=True: modelio.load_gguf(_path(path), nested),
        "gguf_info": lambda path: modelio.gguf_info(_path(path)),
        "save_model": lambda path, params, metadata=None, shard="4GB", sign=False:
            modelio.save_model(path, params, metadata, shard, sign, check=_wpath),
        "load_model": lambda path, verify=True, key=None: modelio.load_model(path, verify, key, check=_path),
        "model_info": lambda path: modelio.model_info(_path(path)),
        "save_checkpoint": lambda path, params, opt=None, step=None, config=None, files=None, metadata=None,
            shard="4GB", sign=False: modelio.save_checkpoint(path, params, opt, step, config, files, metadata, shard,
                                                             sign, check=_wpath, read=_path),
        "load_checkpoint": lambda path, verify=True, key=None: modelio.load_checkpoint(path, verify, key, check=_path),
    }


class OnnxModel:
    def __init__(self, sess):
        self.sess = sess
        self.inputs = [i.name for i in sess.get_inputs()]
        self.outputs = [o.name for o in sess.get_outputs()]

    def __call__(self, *args, **kwargs):
        feed = {}
        if len(args) == 1 and isinstance(args[0], dict):
            feed = dict(args[0])
        else:
            feed = dict(zip(self.inputs, args))
        feed.update(kwargs)
        types = {i.name: i.type for i in self.sess.get_inputs()}
        conv = {}
        for k, v in feed.items():
            a = np.asarray(ops.raw(v))
            t = types.get(k, "")
            if "float16" in t:
                a = a.astype(np.float16)
            elif "float" in t and "double" not in t:
                a = a.astype(np.float32)
            elif "int64" in t:
                a = a.astype(np.int64)
            elif "int32" in t:
                a = a.astype(np.int32)
            conv[k] = a
        outs = self.sess.run(None, conv)
        return outs[0] if len(outs) == 1 else tuple(outs)

    def __repr__(self):
        return f"<onnx model inputs={self.inputs} outputs={self.outputs}>"


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def _from_torch_tree(obj):
    if isinstance(obj, dict):
        return _nest({str(k): v for k, v in obj.items()})
    return obj
