"""Saving and loading models: the standard formats and Apsi's own checkpoint format.

* **GGUF** (llama.cpp, Ollama, LM Studio, …): `save_gguf` / `load_gguf` / `gguf_info`, GGUF version 3, with
  F32, F16, BF16, Q8_0 and Q4_0 tensors (the block formats of ggml, bit-exact), metadata of every GGUF type and
  32-byte data alignment. Files open in any GGUF reader; checked against the reference `gguf` package.
* **Apsi checkpoints (`.apsim`)**: `save_model` / `load_model` / `model_info`. Every `.apsim` file is a valid
  safetensors file, so PyTorch, JAX, Hugging Face and any safetensors reader can load the weights. Apsi adds in the
  safetensors metadata:
    - the parameter tree (records and lists come back as they were saved),
    - physical units of tensors (`Quantity` comes back with its unit),
    - compressed tensors (FP8 / FP6 / FP4 / INT8 / INT4 / BF16 QTensors) stored at their real size with their scales,
    - a SHA-256 of the tensor data, checked when loading (a damaged or modified file is refused),
    - an optional HMAC-SHA256 signature (your per-user key, or a shared key) proving who wrote the file,
    - sharding: large models are split into files of `shard` bytes (default 4 GB) plus a
      `NAME.apsim.index.json` in the Hugging Face layout, so 70B–2T-parameter checkpoints work like small ones.
  Loading is memory-mapped: opening a checkpoint costs almost no RAM until tensors are used.
"""
import hashlib
import hmac
import json
import os
import struct

import numpy as np

from .errors import RuntimeErr
from .units import Quantity, Dim, unit_string
from .values import Record

FORMAT = "apsi-model"
VERSION = "1"

# ============================================================================= GGUF
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGN = 32
# metadata value types
U8, I8, U16, I16, U32, I32, F32V, BOOLV, STR, ARR, U64, I64, F64V = range(13)
# tensor types (ggml)
T_F32, T_F16, T_Q4_0, T_Q8_0, T_I8, T_I16, T_I32, T_I64, T_F64, T_BF16 = 0, 1, 2, 8, 24, 25, 26, 27, 28, 30
_GG_NP = {T_F32: np.float32, T_F16: np.float16, T_F64: np.float64, T_I8: np.int8, T_I16: np.int16,
          T_I32: np.int32, T_I64: np.int64}
_GG_NAME = {T_F32: "F32", T_F16: "F16", T_Q4_0: "Q4_0", T_Q8_0: "Q8_0", T_BF16: "BF16", T_F64: "F64",
            T_I8: "I8", T_I16: "I16", T_I32: "I32", T_I64: "I64"}
QK = 32                                                     # values per Q4_0 / Q8_0 block


def _bf16_bits(a):
    """float32 -> bfloat16 bits, round to nearest even (NaN stays NaN)."""
    u = np.ascontiguousarray(a, np.float32).view(np.uint32)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(a)
    return np.where(nan, ((u >> 16) | 0x40).astype(np.uint16), rounded)


def _q8_0(a):
    """ggml Q8_0: blocks of 32 values = f16 scale d (=amax/127) + 32 int8."""
    x = a.reshape(-1, QK).astype(np.float32)
    amax = np.abs(x).max(axis=1)
    d = amax / 127.0
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d != 0)
    y = x * inv[:, None]
    q = (np.sign(y) * np.floor(np.abs(y) + np.float32(0.5))).astype(np.int8)   # C roundf: halves away from 0
    out = np.empty((x.shape[0], 2 + QK), np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def _q4_0(a):
    """ggml Q4_0: blocks of 32 values = f16 scale d (=max/-8) + 16 bytes of 4-bit codes (x[j] low, x[j+16] high)."""
    x = a.reshape(-1, QK).astype(np.float32)
    idx = np.abs(x).argmax(axis=1)
    mx = x[np.arange(x.shape[0]), idx]
    d = mx / -8.0
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d != 0)
    q = np.minimum(15, (x * inv[:, None] + 8.5).astype(np.int8)).astype(np.uint8)
    out = np.empty((x.shape[0], 2 + QK // 2), np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q[:, :QK // 2] | (q[:, QK // 2:] << 4)
    return out.tobytes()


def _dq8_0(buf, n):
    b = np.frombuffer(buf, np.uint8).reshape(-1, 2 + QK)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    return (b[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)[:n]


def _dq4_0(buf, n):
    b = np.frombuffer(buf, np.uint8).reshape(-1, 2 + QK // 2)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    lo = (b[:, 2:] & 0x0F).astype(np.int8) - 8
    hi = (b[:, 2:] >> 4).astype(np.int8) - 8
    return (np.concatenate([lo, hi], axis=1).astype(np.float32) * d).reshape(-1)[:n]


def _gstr(s):
    b = str(s).encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _gval(v):
    """(type, bytes) for a metadata value."""
    if isinstance(v, (bool, np.bool_)):
        return BOOLV, struct.pack("<?", bool(v))
    if isinstance(v, (int, np.integer)):
        v = int(v)
        if -(1 << 31) <= v < (1 << 31):
            return I32, struct.pack("<i", v)
        return I64, struct.pack("<q", v)
    if isinstance(v, (float, np.floating)):
        return F32V, struct.pack("<f", float(v))
    if isinstance(v, str):
        return STR, _gstr(v)
    if isinstance(v, (list, tuple)):
        items = list(v)
        if not items:
            return ARR, struct.pack("<IQ", I32, 0)
        t0, _ = _gval(items[0])
        parts = []
        for x in items:
            t, b = _gval(x)
            if t != t0:
                raise RuntimeErr("GGUF metadata arrays must hold values of one type")
            parts.append(b)
        return ARR, struct.pack("<IQ", t0, len(items)) + b"".join(parts)
    raise RuntimeErr(f"GGUF metadata cannot hold a {type(v).__name__}")


def _flat_tensors(params):
    from .models import _flatten
    from .precision import QTensor, dequantize
    out = {}
    units = {}
    for name, v in _flatten(params).items():
        if isinstance(v, QTensor):
            v = dequantize(v)
        if isinstance(v, Quantity):
            units[name] = unit_string(v.dim)
            v = v.v
        out[name] = np.asarray(v)
    return out, units


def save_gguf(path, params, metadata=None, dtype="f16", name=None, architecture="apsi"):
    """Save a parameter tree as a GGUF file (version 3).

    dtype: "f32", "f16" (default), "bf16", "q8_0" or "q4_0" for weight matrices; 1-D tensors (biases, norms) and
    matrices whose rows are not a multiple of 32 are kept in f32. metadata: a record/dict of strings, numbers,
    bools or lists, stored as GGUF key/values (use dotted keys such as "llama.context_length")."""
    want = {"f32": T_F32, "fp32": T_F32, "f16": T_F16, "fp16": T_F16, "bf16": T_BF16, "q8_0": T_Q8_0,
            "q4_0": T_Q4_0}.get(str(dtype).lower())
    if want is None:
        raise RuntimeErr("save_gguf dtype must be f32, f16, bf16, q8_0 or q4_0")
    tensors, units = _flat_tensors(params)
    kv = {"general.architecture": architecture, "general.name": name or os.path.splitext(os.path.basename(
        str(path)))[0], "general.alignment": ALIGN, "general.file_type": {T_F32: 0, T_F16: 1, T_Q4_0: 2,
                                                                           T_Q8_0: 7, T_BF16: 32}[want]}
    from . import __version__
    kv["apsi.version"] = __version__
    for k, u in units.items():
        kv[f"apsi.units.{k}"] = u
    md = metadata.fields if isinstance(metadata, Record) else dict(metadata or {})
    for k, v in md.items():
        kv[str(k)] = v
    infos, blobs = [], []
    offset = 0
    for tname, a in tensors.items():
        if a.dtype.kind == "f" or a.dtype == np.bool_:
            a = np.asarray(a, np.float32)
            t = want if (a.ndim >= 2 or want in (T_F16, T_BF16)) else T_F32
            if t in (T_Q8_0, T_Q4_0) and (a.shape[-1] % QK):
                t = T_F16 if want != T_F32 else T_F32
            if t == T_F32:
                data = np.ascontiguousarray(a, np.float32).tobytes()
            elif t == T_F16:
                data = np.ascontiguousarray(a, np.float16).tobytes()
            elif t == T_BF16:
                data = _bf16_bits(a).tobytes()
            elif t == T_Q8_0:
                data = _q8_0(np.ascontiguousarray(a))
            else:
                data = _q4_0(np.ascontiguousarray(a))
        else:
            npt = {np.dtype(np.int8): T_I8, np.dtype(np.int16): T_I16, np.dtype(np.int32): T_I32,
                   np.dtype(np.int64): T_I64}.get(a.dtype)
            if npt is None:
                a = a.astype(np.int64)
                npt = T_I64
            t, data = npt, np.ascontiguousarray(a).tobytes()
        dims = list(reversed(a.shape)) if a.ndim else [1]
        infos.append(_gstr(tname) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) +
                     struct.pack("<IQ", t, offset))
        pad = (-len(data)) % ALIGN
        blobs.append(data + b"\0" * pad)
        offset += len(data) + pad
    head = GGUF_MAGIC + struct.pack("<IQQ", GGUF_VERSION, len(infos), len(kv))
    for k, v in kv.items():
        t, b = (U32, struct.pack("<I", v)) if k == "general.alignment" or k == "general.file_type" else _gval(v)
        head += _gstr(k) + struct.pack("<I", t) + b
    head += b"".join(infos)
    head += b"\0" * ((-len(head)) % ALIGN)
    os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(head)
        for b in blobs:
            fh.write(b)
    os.replace(tmp, str(path))
    return str(path)


class _Reader:
    def __init__(self, buf):
        self.b, self.p = buf, 0

    def take(self, fmt):
        if self.p + struct.calcsize(fmt) > len(self.b):
            raise RuntimeErr("GGUF file is truncated or damaged")
        v = struct.unpack_from(fmt, self.b, self.p)
        self.p += struct.calcsize(fmt)
        return v[0] if len(v) == 1 else v

    def string(self):
        n = self.take("<Q")
        if n > len(self.b) - self.p:
            raise RuntimeErr("GGUF file is truncated or damaged")
        s = bytes(self.b[self.p:self.p + n]).decode("utf-8", "replace")
        self.p += n
        return s

    def value(self, t):
        simple = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", F32V: "<f", BOOLV: "<?",
                  U64: "<Q", I64: "<q", F64V: "<d"}
        if t in simple:
            return self.take(simple[t])
        if t == STR:
            return self.string()
        if t == ARR:
            et, n = self.take("<I"), self.take("<Q")
            if n > len(self.b) - self.p:
                raise RuntimeErr("GGUF file is truncated or damaged")
            return [self.value(et) for _ in range(n)]
        raise RuntimeErr(f"GGUF: unknown metadata type {t}")


def _gguf_parse(path):
    if os.path.getsize(str(path)) < 24:
        raise RuntimeErr(f"{path} is not a GGUF file")
    mm = np.memmap(str(path), dtype=np.uint8, mode="r")
    r = _Reader(mm)
    if bytes(mm[:4]) != GGUF_MAGIC:
        raise RuntimeErr(f"{path} is not a GGUF file")
    r.p = 4
    version = r.take("<I")
    if version not in (2, 3):
        raise RuntimeErr(f"GGUF version {version} is not supported (2 and 3 are)")
    n_t, n_kv = r.take("<Q"), r.take("<Q")
    kv = {}
    for _ in range(n_kv):
        k = r.string()
        kv[k] = r.value(r.take("<I"))
    align = kv.get("general.alignment", ALIGN)
    if not isinstance(align, int) or not 1 <= align <= 1 << 20:
        raise RuntimeErr("GGUF file has an invalid general.alignment")
    infos = []
    for _ in range(n_t):
        name = r.string()
        nd = r.take("<I")
        if nd > 8:
            raise RuntimeErr(f"GGUF tensor {name!r} has {nd} dimensions (ggml allows at most 4)")
        dims = [r.take("<Q") for _ in range(nd)]
        t, off = r.take("<I"), r.take("<Q")
        infos.append((name, tuple(reversed(dims)), t, off))
    start = r.p + ((-r.p) % align)
    return mm, kv, infos, start, version


def gguf_info(path):
    """Metadata and tensor list of a GGUF file (nothing is loaded)."""
    mm, kv, infos, start, version = _gguf_parse(path)
    return Record(version=version, metadata=Record(**{k.replace(".", "_"): v for k, v in kv.items()
                                                       if not isinstance(v, list) or len(v) < 64}),
                  keys=list(kv), tensors=Record(**{n.replace(".", "_"): Record(shape=list(s), type=_GG_NAME.get(t, str(t)))
                                                    for n, s, t, _ in infos}), count=len(infos))


def load_gguf(path, nested=True):
    """Load a GGUF file: F32/F16/F64/integer tensors are memory-mapped; BF16, Q8_0 and Q4_0 are decoded to f32.
    Returns record(params=…, metadata=…). Units saved by Apsi come back as quantities."""
    mm, kv, infos, start, _ = _gguf_parse(path)
    flat = {}
    for name, shape, t, off in infos:
        n = 1
        for d in shape:
            n *= int(d)
        base = start + off
        if t in _GG_NP:
            nbytes = n * np.dtype(_GG_NP[t]).itemsize
        elif t == T_BF16:
            nbytes = 2 * n
        elif t in (T_Q8_0, T_Q4_0):
            if n % QK:
                raise RuntimeErr(f"GGUF tensor {name!r}: {n} values is not a multiple of the block size {QK}")
            nbytes = n // QK * ((2 + QK) if t == T_Q8_0 else (2 + QK // 2))
        else:
            nbytes = 0
        if n < 0 or off < 0 or base + nbytes > len(mm):
            raise RuntimeErr(f"GGUF tensor {name!r} lies outside the file (truncated or damaged)")
        if t in _GG_NP:
            dt = np.dtype(_GG_NP[t])
            arr = np.frombuffer(mm, dt, count=n, offset=base).reshape(shape)
        elif t == T_BF16:
            u = np.frombuffer(mm, np.uint16, count=n, offset=base).astype(np.uint32) << 16
            arr = u.view(np.float32).reshape(shape)
        elif t == T_Q8_0:
            arr = _dq8_0(mm[base:base + n // QK * (2 + QK)], n).reshape(shape)
        elif t == T_Q4_0:
            arr = _dq4_0(mm[base:base + n // QK * (2 + QK // 2)], n).reshape(shape)
        else:
            raise RuntimeErr(f"GGUF tensor {name!r} has type {t} ({_GG_NAME.get(t, '?')}); Apsi reads "
                             f"F32, F16, BF16, F64, integer, Q8_0 and Q4_0")
        u = kv.get(f"apsi.units.{name}")
        if u:
            from .units import parse_unit
            scale, dim = parse_unit(u)
            arr = Quantity(np.asarray(arr) * scale, dim)
        flat[name] = arr
    from .models import _nest
    params = _nest(flat) if nested else Record(**{k.replace(".", "_"): v for k, v in flat.items()})
    meta = Record(**{k.replace(".", "_"): v for k, v in kv.items() if not k.startswith("apsi.units.")})
    return Record(params=params, metadata=meta)


# ============================================================================= Apsi checkpoints (.apsim)
_DT = {np.dtype(np.float64): "F64", np.dtype(np.float32): "F32", np.dtype(np.float16): "F16",
       np.dtype(np.int64): "I64", np.dtype(np.int32): "I32", np.dtype(np.int16): "I16", np.dtype(np.int8): "I8",
       np.dtype(np.uint8): "U8", np.dtype(np.uint16): "U16", np.dtype(np.uint32): "U32",
       np.dtype(np.uint64): "U64", np.dtype(np.bool_): "BOOL"}
_DT_R = {v: k for k, v in _DT.items()}


def _size(x):
    if isinstance(x, (int, float)):
        return int(x)
    from .scale import number
    t = str(x).strip().upper().rstrip("B")
    return int(number(t[:-1] + {"K": "K", "M": "M", "G": "B", "T": "T"}[t[-1]]) if t and t[-1] in "KMGT"
               else number(t))


def _key(sign):
    if sign is True:
        from .data import _cache_key
        return _cache_key()
    return str(sign).encode("utf-8")


def _entries(params):
    """name -> (array, extra metadata) with QTensors split into codes + scale and units recorded."""
    from .models import _flatten
    from .precision import QTensor
    out, meta = {}, {"units": {}, "qtensors": {}}
    for name, v in _flatten(params).items():
        if isinstance(v, QTensor):
            out[name] = np.ascontiguousarray(v.codes)
            sc = np.ascontiguousarray(np.asarray(v.scale, np.float32))
            out[name + "__qscale"] = sc.reshape(-1) if sc.ndim else sc.reshape(1)
            meta["qtensors"][name] = {"fmt": getattr(v.fmt, "name", v.fmt), "shape": list(v.shape), "axis": v.axis, "packed": bool(v.packed),
                                      "scale_shape": list(np.shape(v.scale)),
                                      "dim": [str(x) for x in v.dim] if v.dim is not None else None}
            continue
        if isinstance(v, Quantity):
            meta["units"][name] = [str(x) for x in v.dim]
            v = v.v
        a = np.asarray(v)
        if a.dtype not in _DT:
            a = a.astype(np.float32)
        out[name] = np.ascontiguousarray(a)
    return out, meta


def _layout(header):
    """The tensor table in a fixed form: hashed with the data, so names, shapes and dtypes are protected too."""
    t = {k: [v["dtype"], [int(d) for d in v["shape"]], [int(o) for o in v["data_offsets"]]]
         for k, v in header.items() if k != "__metadata__"}
    return json.dumps(t, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_shard(path, tensors, meta_strings):
    header, offset = {}, 0
    for name, a in tensors.items():
        n = a.nbytes
        header[name] = {"dtype": _DT[a.dtype], "shape": list(a.shape), "data_offsets": [offset, offset + n]}
        offset += n
    h = hashlib.sha256(_layout(header))
    for a in tensors.values():
        h.update(memoryview(a).cast("B") if a.size else b"")
    meta = dict(meta_strings)
    meta["sha256"] = h.hexdigest()
    header["__metadata__"] = meta
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((-len(hb)) % 8)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for a in tensors.values():
            if a.size:
                fh.write(memoryview(a).cast("B"))
    os.replace(tmp, str(path))
    return meta["sha256"]


def save_model(path, params, metadata=None, shard="4GB", sign=False, check=None):
    """Save a parameter tree as an Apsi checkpoint (.apsim): a safetensors file with units, compressed tensors,
    the tree layout and a SHA-256 kept. Larger than `shard` bytes -> several files + NAME.apsim.index.json.
    sign=True signs with your per-user key, sign="secret" with a shared key (load_model checks it)."""
    from . import __version__
    path = str(path)
    if not path.endswith(".apsim"):
        path += ".apsim"
    check = check or (lambda p: p)
    path = str(check(path))                       # resolved (relative to the program) and permission-checked
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tensors, meta = _entries(params)
    user = metadata.fields if isinstance(metadata, Record) else dict(metadata or {})
    base = {"format": FORMAT, "version": VERSION, "apsi": __version__,
            "units": json.dumps(meta["units"]), "qtensors": json.dumps(meta["qtensors"]),
            "metadata": json.dumps({str(k): v for k, v in user.items()}, default=str)}
    limit = _size(shard)
    shards, cur, size = [], {}, 0
    for name, a in tensors.items():
        if cur and size + a.nbytes > limit:
            shards.append(cur)
            cur, size = {}, 0
        cur[name] = a
        size += a.nbytes
    shards.append(cur)
    total = sum(a.nbytes for a in tensors.values())
    if len(shards) == 1:
        digest = _write_shard(check(path), shards[0], base)
        if sign:
            _write_signature(path, {os.path.basename(path): digest}, base, sign)
        return path
    stem = path[:-len(".apsim")]
    weight_map, digests = {}, {}
    n = len(shards)
    for i, sh in enumerate(shards, 1):
        fn = f"{os.path.basename(stem)}-{i:05d}-of-{n:05d}.apsim"
        digests[fn] = _write_shard(check(os.path.join(os.path.dirname(path), fn)), sh, dict(base, shard=f"{i}/{n}"))
        for name in sh:
            weight_map[name] = fn
    index = {"metadata": dict(base, total_size=total, shards=digests), "weight_map": weight_map}
    if sign:
        index["metadata"]["signature"] = _signature(digests, base, sign)
    idx = check(path + ".index.json")
    with open(idx + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    os.replace(idx + ".tmp", idx)
    return idx


def _signature(digests, base, sign):
    msg = json.dumps({"digests": digests, "meta": base}, sort_keys=True).encode()
    return hmac.new(_key(sign), msg, hashlib.sha256).hexdigest()


def _write_signature(path, digests, base, sign):
    with open(path + ".sig", "w", encoding="utf-8") as fh:
        json.dump({"signature": _signature(digests, base, sign)}, fh)


def _read_shard(path, verify):
    bad = f"{os.path.basename(path)} is not a valid Apsi checkpoint"
    if os.path.getsize(path) < 8:
        raise RuntimeErr(bad + " (too short)")
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    n = int(np.frombuffer(mm[:8], "<u8")[0])
    if n > min(len(mm) - 8, 100 << 20):
        raise RuntimeErr(bad + " (header size out of range)")
    try:
        header = json.loads(bytes(mm[8:8 + n]).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise RuntimeErr(bad + " (unreadable header)")
    if not isinstance(header, dict):
        raise RuntimeErr(bad + " (unreadable header)")
    meta = header.pop("__metadata__", {})
    start = 8 + n
    for name, info in header.items():
        try:
            a, b = (int(x) for x in info["data_offsets"])
            dt = _DT_R[info["dtype"]]
            count = 1
            for d in info["shape"]:
                count *= int(d)
        except (KeyError, TypeError, ValueError):
            raise RuntimeErr(bad + f" (tensor '{name}' is described incorrectly)")
        if not 0 <= a <= b <= len(mm) - start or b - a != count * dt.itemsize:
            raise RuntimeErr(bad + f" (tensor '{name}' lies outside the file)")
    if verify and "sha256" in meta:
        h = hashlib.sha256(_layout(header))
        end = start + max((v["data_offsets"][1] for v in header.values()), default=0)
        step = 1 << 26
        for p in range(start, end, step):
            h.update(mm[p:min(end, p + step)])
        if h.hexdigest() != meta["sha256"]:
            raise RuntimeErr(f"{os.path.basename(path)} is damaged or was modified: its data does not match the "
                             f"SHA-256 stored in the file")
    arrays = {}
    for name, info in header.items():
        a, b = info["data_offsets"]
        dt = _DT_R[info["dtype"]]
        arrays[name] = np.frombuffer(mm, dt, count=(b - a) // dt.itemsize, offset=start + a).reshape(info["shape"])
    return arrays, meta


def load_model(path, verify=True, key=None, check=None):
    """Load an Apsi checkpoint (.apsim, or its .index.json for sharded ones), memory-mapped.

    verify=True checks every shard's SHA-256; key= checks the signature (True = your per-user key, or the shared
    key string); a missing or wrong signature is then an error. Units and compressed tensors come back as saved."""
    from .models import _nest
    from .precision import QTensor, get_format
    check = check or (lambda p: p)
    signed = key is not None and key is not False
    verify = verify or signed                 # a signature only means something if the data is checked too
    path = str(path)
    for cand in (path, path + ".apsim", path + ".apsim.index.json", path + ".index.json"):
        if os.path.exists(check(cand)):
            path = cand
            break
    path = str(check(path))                   # resolved (relative to the program) and permission-checked
    import re as _re
    m = _re.match(r"(.*)-\d{5}-of-\d{5}\.apsim$", path)
    if m and os.path.exists(m.group(1) + ".apsim.index.json"):
        path = m.group(1) + ".apsim.index.json"       # one shard named: the whole checkpoint it belongs to
    if not os.path.exists(path) and os.path.exists(path + ".apsim"):
        path += ".apsim"
    if not os.path.exists(path) and os.path.exists(path + ".index.json"):
        path += ".index.json"
    if path.endswith(".index.json"):
        try:
            with open(check(path), encoding="utf-8") as fh:
                index = json.load(fh)
            index["weight_map"].values(), index["metadata"].get
        except (ValueError, KeyError, TypeError, AttributeError):
            raise RuntimeErr(f"{os.path.basename(path)} is not a valid checkpoint index")
        folder = os.path.dirname(os.path.abspath(path))
        flat, meta = {}, None
        for fn in sorted(set(index["weight_map"].values())):
            if os.path.basename(fn) != fn:
                raise RuntimeErr(f"the index names a shard outside its folder: {fn!r}")
            arrays, m = _read_shard(check(os.path.join(folder, fn)), verify)
            if verify and index["metadata"].get("shards", {}).get(fn) not in (None, m.get("sha256")):
                raise RuntimeErr(f"{fn} does not belong to this checkpoint (its SHA-256 differs from the index)")
            flat.update(arrays)
            meta = meta or m
        base = {k: v for k, v in index["metadata"].items() if k not in ("total_size", "shards", "signature")}
        digests = index["metadata"].get("shards", {})
        sig = index["metadata"].get("signature")
        if meta is not None:
            meta = dict(base)                 # the layout recorded in the index is the one that is signed
    else:
        flat, meta = _read_shard(check(path), verify)
        base = {k: v for k, v in meta.items() if k not in ("sha256",)}
        digests = {os.path.basename(path): meta.get("sha256")}
        sig = None
        if os.path.exists(path + ".sig"):
            with open(path + ".sig", encoding="utf-8") as fh:
                sig = json.load(fh).get("signature")
    if meta is None or meta.get("format") != FORMAT:
        raise RuntimeErr(f"{os.path.basename(path)} is not an Apsi checkpoint (use load_safetensors for plain "
                         f"safetensors files)")
    if signed:
        if not sig:
            raise RuntimeErr(f"{os.path.basename(path)} is not signed")
        if not hmac.compare_digest(sig, _signature(digests, base, key)):
            raise RuntimeErr(f"the signature of {os.path.basename(path)} does not match: the file was not written "
                             f"with this key, or it was changed")
    try:
        units = json.loads(meta.get("units", "{}"))
        qts = json.loads(meta.get("qtensors", "{}"))
        out = {}
        for name, info in qts.items():
            sc = np.asarray(flat.pop(name + "__qscale"), np.float32).reshape(info["scale_shape"])
            dim = Dim(info["dim"]) if info.get("dim") is not None else None
            out[name] = QTensor(flat.pop(name), sc if sc.ndim else float(sc), get_format(info["fmt"]),
                                info["shape"], info["axis"], dim, info["packed"])
        for name, a in flat.items():
            out[name] = Quantity(a, Dim(units[name])) if name in units else a
        return _nest(out)
    except RuntimeErr:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise RuntimeErr(f"{os.path.basename(path)} has an inconsistent Apsi header ({type(e).__name__}: {e})")


def model_info(path):
    """What an Apsi checkpoint holds, from the headers only: tensors, sizes, shards, units, compression."""
    path = str(path)
    files = [path]
    meta = {}
    if path.endswith(".index.json"):
        with open(path, encoding="utf-8") as fh:
            index = json.load(fh)
        folder = os.path.dirname(os.path.abspath(path))
        files = [os.path.join(folder, f) for f in sorted(set(index["weight_map"].values()))]
    tensors, total = 0, 0
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        meta = h.pop("__metadata__", meta)
        tensors += len([k for k in h if not k.endswith("__qscale")])
        total += sum(v["data_offsets"][1] - v["data_offsets"][0] for v in h.values())
    return Record(format=meta.get("format"), apsi=meta.get("apsi"), files=len(files), tensors=tensors,
                  bytes=total, units=len(json.loads(meta.get("units", "{}"))),
                  compressed=len(json.loads(meta.get("qtensors", "{}"))),
                  metadata=Record(**json.loads(meta.get("metadata", "{}"))))


# ============================================================================= whole training state in one checkpoint
_OPT_FIELDS = ("lr", "b1", "b2", "eps", "wd", "momentum", "t", "weight_decay")


def _opt_state(opt):
    """(hyper-parameters + class, tensor trees) of an optimizer object."""
    if opt is None:
        return None, {}
    hyper = {"class": type(opt).__name__}
    trees = {}
    for k, v in vars(opt).items():
        if k in _OPT_FIELDS and isinstance(v, (int, float)):
            hyper[k] = v
        elif v is None or isinstance(v, (int, float, str, bool)):
            hyper[k] = v
        else:
            try:
                from .models import _flatten
                if _flatten(v):
                    trees[k] = v
            except Exception:
                pass
    return hyper, trees


def _opt_restore(hyper, trees):
    from . import stdlib
    cls = getattr(stdlib, hyper.get("class", ""), None)
    if cls is None:
        return None
    opt = cls.__new__(cls)
    for k, v in hyper.items():
        if k != "class":
            setattr(opt, k, v)
    for k, v in trees.items():
        setattr(opt, k, v)
    return opt


def save_checkpoint(path, params, opt=None, step=None, config=None, files=None, metadata=None, shard="4GB",
                    sign=False, check=None, read=None):
    """Everything needed to use or resume a model in one .apsim (or in shards when it is larger than `shard`):
    the weights, the optimizer state (Adam/AdamW/SGD moments and step), the training step, a config record and any
    files (tokenizer.json, vocab, notes) — load_checkpoint gives all of it back. Still a safetensors file."""
    tree = {"params": params}
    hyper, trees = _opt_state(opt)
    if trees:
        tree["optimizer"] = trees
    blobs = {}
    for name, f in (files.fields if isinstance(files, Record) else dict(files or {})).items():
        if isinstance(f, (bytes, bytearray)):
            data = bytes(f)
        elif isinstance(f, str) and os.path.isfile(str((read or (lambda p: p))(f))):
            with open(str((read or (lambda p: p))(f)), "rb") as fh:
                data = fh.read()
        else:
            data = str(f).encode("utf-8")
        blobs[str(name).replace(".", "\u2024")] = np.frombuffer(data, np.uint8).copy() if data else np.zeros(0, np.uint8)
    if blobs:
        tree["files"] = blobs
    user = metadata.fields if isinstance(metadata, Record) else dict(metadata or {})
    cfg = config.fields if isinstance(config, Record) else config
    meta = dict(user, __checkpoint__=json.dumps({"step": step, "optimizer": hyper, "config": cfg}, default=_jsonable))
    return save_model(path, tree, meta, shard, sign, check)


def _jsonable(v):
    if isinstance(v, Record):
        return dict(v.fields)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return str(v)


def load_checkpoint(path, verify=True, key=None, check=None):
    """record(params, optimizer, step, config, files, metadata) from save_checkpoint; plain save_model files give
    record(params=…) with the rest empty, so any checkpoint loads the same way."""
    tree = load_model(path, verify, key, check)
    info = _meta_of(path, check)
    ck = {}
    try:
        ck = json.loads(info.get("__checkpoint__", "{}"))
    except ValueError:
        ck = {}
    fields = tree.fields if isinstance(tree, Record) else {}
    if "__checkpoint__" not in info:
        return Record(params=tree, optimizer=None, step=None, config=None, files=Record(),
                      metadata=Record(**{k: v for k, v in info.items()}))
    trees = fields.get("optimizer")
    opt = _opt_restore(ck.get("optimizer") or {}, trees.fields if isinstance(trees, Record) else {}) \
        if ck.get("optimizer") else None
    files = fields.get("files")
    fd = {}
    if isinstance(files, Record):
        for k, v in files.fields.items():
            fd[k.replace("\u2024", ".")] = bytes(np.asarray(v, np.uint8).tobytes())
    cfg = ck.get("config")
    from .interop import from_py
    return Record(params=fields.get("params"), optimizer=opt, step=ck.get("step"),
                  config=(Record(**{k: from_py(v) for k, v in cfg.items()}) if isinstance(cfg, dict)
                          else from_py(cfg) if cfg is not None else None), files=fd,
                  metadata=Record(**{k: v for k, v in info.items() if k != "__checkpoint__"}))


def _meta_of(path, check=None):
    """The user metadata of a checkpoint (single file or index)."""
    check = check or (lambda p: p)
    path = str(path)
    for cand in (path, path + ".apsim", path + ".apsim.index.json", path + ".index.json"):
        if os.path.exists(check(cand)):
            path = str(check(cand))
            break
    if path.endswith(".index.json"):
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)["metadata"]
    else:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            m = json.loads(fh.read(n)).get("__metadata__", {})
    try:
        user = json.loads(m.get("metadata", "{}"))
    except ValueError:
        user = {}
    return user


def module_functions():
    return {"save_checkpoint": save_checkpoint, "load_checkpoint": load_checkpoint, "save_gguf": save_gguf, "load_gguf": load_gguf, "gguf_info": gguf_info, "save_model": save_model,
            "load_model": load_model, "model_info": model_info}


# ============================================================================= command line: apsi model …
def _load_any(path):
    p = str(path).lower()
    if p.endswith(".gguf"):
        r = load_gguf(path)
        return r.params, dict(r.metadata.fields)
    if p.endswith(".apsim") or p.endswith(".apsim.index.json"):
        return load_model(path), {}
    if p.endswith(".safetensors"):
        from .models import load_safetensors
        return load_safetensors(path), {}
    raise RuntimeErr(f"{path}: unknown model format (use .gguf, .apsim or .safetensors)")


def _plain(params):
    """QTensors -> f32 arrays (for formats that cannot hold Apsi's compressed tensors)."""
    from .precision import QTensor, dequantize
    from .values import tree_map
    return tree_map(lambda v: np.asarray(dequantize(v), np.float32) if isinstance(v, QTensor) else v, params)


def _show(rec, indent="  "):
    for k, v in (rec.fields if isinstance(rec, Record) else rec).items():
        if isinstance(v, Record):
            print(f"{indent}{k}:")
            _show(v, indent + "  ")
        else:
            print(f"{indent}{k}: {v}")


def cmd_model(args):
    from .errors import ApsiError
    try:
        if args.action == "info":
            p = args.file.lower()
            _show(gguf_info(args.file) if p.endswith(".gguf") else model_info(args.file) if ".apsim" in p
                  else __import__("apsi.models", fromlist=["x"]).safetensors_info(args.file), "")
        elif args.action == "verify":
            load_model(args.file, verify=True, key=(args.key if args.key else None))
            print(f"OK: {args.file} is intact" + (" and the signature matches" if args.key else ""))
        elif args.action == "convert":
            if not args.output:
                raise RuntimeErr("convert needs an output file: apsi model convert IN OUT")
            params, meta = _load_any(args.file)
            out = args.output.lower()
            if out.endswith(".gguf"):
                r = save_gguf(args.output, _plain(params), metadata=meta, dtype=args.dtype or "f16")
            elif out.endswith(".apsim"):
                r = save_model(args.output, params, metadata=meta, shard=args.shard,
                               sign=(args.key if args.key else False))
            elif out.endswith(".safetensors"):
                from .models import save_safetensors
                r = save_safetensors(args.output, _plain(params))
            else:
                raise RuntimeErr(f"{args.output}: unknown model format (use .gguf, .apsim or .safetensors)")
            print(f"wrote {r}")
    except (ApsiError, OSError) as e:
        print(f"error: {e}")
        return 1
    return 0


def add_parser(sub):
    p = sub.add_parser("model", help="inspect, verify and convert model files (GGUF, .apsim, safetensors)")
    p.add_argument("action", choices=["info", "verify", "convert"])
    p.add_argument("file")
    p.add_argument("output", nargs="?")
    p.add_argument("--dtype", help="GGUF tensor type: f32, f16, bf16, q8_0, q4_0 (default f16)")
    p.add_argument("--shard", default="4GB", help="largest .apsim file before splitting (default 4GB)")
    p.add_argument("--key", help="shared key for signing (convert to .apsim) or checking (verify)")
    p.set_defaults(fn=cmd_model)
