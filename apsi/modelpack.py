"""Trained models as sealed packages, and libraries that run them without the code that trained them.

  pack_model(path, model, params, password=None, key=None, metadata=None)
      one file (.apsipkg): the model's architecture (its layers and their settings) and its weights, encrypted with
      ChaCha20 (RFC 8439) under a 256-bit key and authenticated with HMAC-SHA256 (encrypt-then-MAC): without the key
      nothing in it can be read, and a changed byte is refused. The key comes from a password (scrypt, 2^15 x 8)
      or is given as 32 random bytes (new_key()).
  load_package(path, password=None, key=None) -> a Packaged model: .predict(x), .metadata, .model, .params
  package_info(path)       what can be read without the key (format, names in the metadata you chose to show, sizes)
  build_library(package, out_dir, name)
      a Python library folder `name/` that holds the package and a three-function API (load, predict, info): it
      needs only the Apsi runtime, never the training program or its folder. Ship the folder (or zip it); the key
      stays with whoever may run the model (embed_key=True puts it inside the library: convenient, not secret).

The architecture is stored as data, not code: each layer's class (only classes of Apsi itself are accepted when
loading) and its settings, so opening a package never runs anything from the file.
"""
import base64
import hashlib
import hmac
import io
import json
import os
import struct

import numpy as np

from .errors import RuntimeErr
from .values import Record

MAGIC = b"APSIPKG1"
FORMAT = "apsi-package"
SCRYPT = dict(n=2 ** 15, r=8, p=1)
_EXTRA = ("config",)                                  # attributes set on a layer after it was made, kept as well


class PackageError(RuntimeErr):
    pass


# ============================================================================= ChaCha20 (RFC 8439), vectorised
def _rotl(v, c):
    return ((v << np.uint32(c)) | (v >> np.uint32(32 - c))).astype(np.uint32)


def _chacha_blocks(key, nonce, counter, n_blocks):
    """n_blocks keystream blocks (64 bytes each) from block number `counter`: all blocks at once as uint32 columns."""
    const = np.array([0x61707865, 0x3320646E, 0x79622D32, 0x6B206574], np.uint32)
    k = np.frombuffer(key, "<u4")
    nn = np.frombuffer(nonce, "<u4")
    ctr = (np.arange(n_blocks, dtype=np.uint64) + np.uint64(counter)).astype(np.uint32)
    s = [np.full(n_blocks, c, np.uint32) for c in const] + [np.full(n_blocks, x, np.uint32) for x in k] + \
        [ctr] + [np.full(n_blocks, x, np.uint32) for x in nn]
    x = [v.copy() for v in s]

    def qr(a, b, c, d):
        x[a] += x[b]; x[d] = _rotl(x[d] ^ x[a], 16)                      # noqa: E702
        x[c] += x[d]; x[b] = _rotl(x[b] ^ x[c], 12)                      # noqa: E702
        x[a] += x[b]; x[d] = _rotl(x[d] ^ x[a], 8)                       # noqa: E702
        x[c] += x[d]; x[b] = _rotl(x[b] ^ x[c], 7)                       # noqa: E702
    with np.errstate(over="ignore"):
        for _ in range(10):
            qr(0, 4, 8, 12); qr(1, 5, 9, 13); qr(2, 6, 10, 14); qr(3, 7, 11, 15)    # noqa: E702
            qr(0, 5, 10, 15); qr(1, 6, 11, 12); qr(2, 7, 8, 13); qr(3, 4, 9, 14)    # noqa: E702
        out = np.stack([x[i] + s[i] for i in range(16)], axis=1)
    return out.astype("<u4").tobytes()


def chacha20(key, nonce, data, counter=1):
    """data XOR the ChaCha20 keystream (RFC 8439: 32-byte key, 12-byte nonce, block counter from `counter`)."""
    if len(key) != 32 or len(nonce) != 12:
        raise PackageError("chacha20: a 32-byte key and a 12-byte nonce")
    data = bytes(data)
    try:                                                                  # the same cipher, in C when installed
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
        enc = Cipher(algorithms.ChaCha20(key, struct.pack("<I", counter) + nonce), mode=None).encryptor()
        return enc.update(data) + enc.finalize()
    except ImportError:
        pass
    out = bytearray()
    step = 1 << 16                                                        # blocks per piece (4 MB)
    nb = (len(data) + 63) // 64
    for b0 in range(0, nb, step):
        nblk = min(step, nb - b0)
        ks = np.frombuffer(_chacha_blocks(key, nonce, counter + b0, nblk), np.uint8)
        piece = np.frombuffer(data, np.uint8, count=min(len(data) - 64 * b0, 64 * nblk), offset=64 * b0)
        out += (piece ^ ks[:len(piece)]).tobytes()
    return bytes(out)


def _keys(master, salt):
    """Encryption key and MAC key from the master key (HKDF-SHA256)."""
    prk = hmac.new(salt, master, hashlib.sha256).digest()
    t1 = hmac.new(prk, b"apsi-package enc\x01", hashlib.sha256).digest()
    t2 = hmac.new(prk, t1 + b"apsi-package mac\x02", hashlib.sha256).digest()
    return t1, t2


def new_key():
    """32 random bytes, as hex (keep it safe: it opens the package)."""
    return os.urandom(32).hex()


def _master(password, key, salt, kdf):
    if key is not None:
        k = bytes.fromhex(key) if isinstance(key, str) else bytes(key)
        if len(k) != 32:
            raise PackageError("the key is 32 bytes (64 hex digits): new_key() makes one")
        return k
    if password is None:
        raise PackageError("this package is encrypted: give password= or key=")
    return hashlib.scrypt(str(password).encode(), salt=salt, n=kdf["n"], r=kdf["r"], p=kdf["p"],
                          maxmem=256 * 1024 * 1024, dklen=32)


# ============================================================================= the architecture as data
def _spec(obj, arrays, depth=0):
    """Layers and their settings -> JSON-able data (arrays go to `arrays`, referenced by index)."""
    from .layers import Layer
    if depth > 200:
        raise PackageError("the model nests too deeply to package")
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        arrays.append(obj)
        return {"$a": len(arrays) - 1}
    if isinstance(obj, (list, tuple)):
        return {"$" + ("l" if isinstance(obj, list) else "t"): [_spec(v, arrays, depth + 1) for v in obj]}
    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj):
            return {"$d": {k: _spec(v, arrays, depth + 1) for k, v in obj.items()}}
        return {"$dk": [[_spec(k, arrays, depth + 1), _spec(v, arrays, depth + 1)] for k, v in obj.items()]}
    if isinstance(obj, (set, frozenset)):
        return {"$s": [_spec(v, arrays, depth + 1) for v in sorted(obj, key=repr)]}
    if isinstance(obj, Record):
        return {"$r": {k: _spec(v, arrays, depth + 1) for k, v in obj.fields.items()}}
    cls = type(obj)
    if isinstance(obj, Layer) or (cls.__module__.startswith("apsi.") and hasattr(obj, "__dict__")):
        if not cls.__module__.startswith("apsi."):
            raise PackageError(f"{cls.__name__} is a layer of your own program: a package holds Apsi's layers only "
                               "(write it with Apsi's layers, or ship your program with the weights instead)")
        made = vars(obj).get("_init_args")
        if made is not None and made[0] is cls:          # made again from the arguments it was made with
            return {"$o": f"{cls.__module__}:{cls.__qualname__}", "a": _spec(list(made[1]), arrays, depth + 1),
                    "k": _spec(dict(made[2]), arrays, depth + 1),
                    "x": {k: _spec(v, arrays, depth + 1) for k, v in vars(obj).items()
                          if k in _EXTRA and not callable(v)}}
        state = {k: _spec(v, arrays, depth + 1) for k, v in vars(obj).items() if not callable(v) or
                 isinstance(v, Layer)}
        return {"$o": f"{cls.__module__}:{cls.__qualname__}", "s": state}
    raise PackageError(f"cannot package a {cls.__name__} inside the model (functions and outside objects are not "
                       "stored: only layers, numbers, text, lists and arrays)")


def _unspec(v, arrays):
    if not isinstance(v, dict):
        return v
    if "$a" in v:
        return arrays[v["$a"]]
    if "$l" in v:
        return [_unspec(x, arrays) for x in v["$l"]]
    if "$t" in v:
        return tuple(_unspec(x, arrays) for x in v["$t"])
    if "$d" in v:
        return {k: _unspec(x, arrays) for k, x in v["$d"].items()}
    if "$dk" in v:
        return {_unspec(k, arrays): _unspec(x, arrays) for k, x in v["$dk"]}
    if "$s" in v:
        return set(_unspec(x, arrays) for x in v["$s"])
    if "$r" in v:
        return Record(**{k: _unspec(x, arrays) for k, x in v["$r"].items()})
    if "$o" in v:
        import importlib
        mod, _, qual = v["$o"].partition(":")
        if not (mod == "apsi" or mod.startswith("apsi.")):
            raise PackageError(f"refused: the package names a class outside Apsi ({mod})")
        cls = importlib.import_module(mod)
        for part in qual.split("."):
            cls = getattr(cls, part)
        if not isinstance(cls, type):
            raise PackageError(f"refused: {v['$o']} is not a class")
        if "a" in v:
            obj = cls(*_unspec(v["a"], arrays), **_unspec(v["k"], arrays))
            for k, x in v.get("x", {}).items():
                setattr(obj, k, _unspec(x, arrays))
            return obj
        obj = cls.__new__(cls)
        for k, x in v["s"].items():
            setattr(obj, k, _unspec(x, arrays))
        return obj
    raise PackageError("a damaged architecture in the package")


def _tree(params, arrays):
    if isinstance(params, Record):
        return {"$r": {k: _tree(v, arrays) for k, v in params.fields.items()}}
    if isinstance(params, (list, tuple)):
        return {"$l" if isinstance(params, list) else "$t": [_tree(v, arrays) for v in params]}
    if isinstance(params, dict):
        return {"$d": {str(k): _tree(v, arrays) for k, v in params.items()}}
    a = np.asarray(params.a if hasattr(params, "a") and not isinstance(params, np.ndarray) else params)
    arrays.append(np.ascontiguousarray(a))
    return {"$a": len(arrays) - 1}


def _npz(arrays):
    buf = io.BytesIO()
    np.savez(buf, **{f"a{i}": a for i, a in enumerate(arrays)})
    return buf.getvalue()


def _unnpz(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        return [z[f"a{i}"] for i in range(len(z.files))]


# ============================================================================= the file
def pack_model(path, model, params, password=None, key=None, metadata=None, public=None):
    """Write the model and its weights as one encrypted file. metadata: anything JSON-able, sealed with the rest;
    public: a few names / numbers anyone may read without the key (package_info). Returns the path."""
    if password is None and key is None:
        raise PackageError("pack_model needs password= or key= (new_key() makes a key)")
    arch, weights = [], []
    payload = json.dumps({"format": FORMAT, "model": _spec(model, arch), "params": _tree(params, weights),
                          "metadata": metadata if metadata is not None else {}},
                         separators=(",", ":"), default=str).encode()
    body = struct.pack("<Q", len(payload)) + payload + _npz(arch + weights) + struct.pack("<Q", len(arch))
    salt, nonce = os.urandom(16), os.urandom(12)
    kdf = dict(SCRYPT) if key is None else None
    k_enc, k_mac = _keys(_master(password, key, salt, kdf or SCRYPT), salt)
    sealed = chacha20(k_enc, nonce, body)
    head = json.dumps({"format": FORMAT, "version": 1, "cipher": "chacha20+hmac-sha256", "salt": salt.hex(),
                       "nonce": nonce.hex(), "kdf": ({"scrypt": kdf} if kdf else "key"),
                       "public": public or {}, "bytes": len(sealed)}, separators=(",", ":")).encode()
    mac = hmac.new(k_mac, MAGIC + struct.pack("<I", len(head)) + head + sealed, hashlib.sha256).digest()
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(MAGIC + struct.pack("<I", len(head)) + head + sealed + mac)
    os.replace(tmp, path)
    return str(path)


def _read(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:8] != MAGIC or len(raw) < 12 + 32:
        raise PackageError(f"{path} is not an Apsi model package")
    hl = struct.unpack("<I", raw[8:12])[0]
    head = json.loads(raw[12:12 + hl])
    return raw, head, hl


def package_info(path):
    """What anyone can read without the key."""
    raw, head, _ = _read(path)
    return Record(format=head["format"], version=head["version"], cipher=head["cipher"],
                  password=isinstance(head["kdf"], dict), public=head.get("public", {}), bytes=len(raw))


class Packaged:
    """A model opened from a package: predict(x) runs it (NumPy by default; backend="gpu" on a card)."""

    def __init__(self, model, params, metadata):
        self.model, self.params, self.metadata = model, params, metadata

    def predict(self, x, backend="numpy", **kw):
        from . import layers as L
        return L.predict(self.model, self.params, x, backend=backend, **kw)

    def __call__(self, x, **kw):
        return self.predict(x, **kw)

    def describe(self):
        return self.model.describe() if hasattr(self.model, "describe") else type(self.model).__name__


def load_package(path, password=None, key=None):
    """Open a package (the same password or key it was packed with). A wrong key or a changed byte is refused
    before anything is decoded."""
    raw, head, hl = _read(path)
    salt, nonce = bytes.fromhex(head["salt"]), bytes.fromhex(head["nonce"])
    kdf = head["kdf"]["scrypt"] if isinstance(head["kdf"], dict) else SCRYPT
    if not isinstance(head["kdf"], dict) and key is None:
        raise PackageError("this package was sealed with a key: give key=")
    k_enc, k_mac = _keys(_master(password, key, salt, kdf), salt)
    sealed, mac = raw[12 + hl:-32], raw[-32:]
    want = hmac.new(k_mac, raw[:12 + hl] + sealed, hashlib.sha256).digest()
    if not hmac.compare_digest(want, mac):
        raise PackageError("wrong password / key, or the package was changed")
    body = chacha20(k_enc, nonce, sealed)
    n = struct.unpack("<Q", body[:8])[0]
    doc = json.loads(body[8:8 + n])
    narch = struct.unpack("<Q", body[-8:])[0]
    arrays = _unnpz(body[8 + n:-8])
    arch, weights = arrays[:narch], arrays[narch:]
    model = _unspec(doc["model"], arch)
    params = _unspec(doc["params"], weights)
    return Packaged(model, params, doc.get("metadata", {}))


# ============================================================================= a library around a package
_LIB_INIT = '''"""{name}: a trained model (made with Apsi). load(...) opens it, predict(x) runs it."""
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.join(_HERE, "model.apsipkg")
_KEY = {key!r}
_model = None


def load(password=None, key=None):
    """Open the model (once): the password or key it was sealed with{embedded}."""
    global _model
    from apsi.modelpack import load_package
    _model = load_package(_PKG, password=password, key=key if key is not None else _KEY)
    return _model


def predict(x, **kw):
    """The model's output for x (load() first, unless the key is inside the library)."""
    if _model is None:
        load()
    return _model.predict(x, **kw)


def info():
    from apsi.modelpack import package_info
    return package_info(_PKG)
'''


def build_library(package, out_dir, name, embed_key=None):
    """A Python library folder out_dir/name holding the package and load / predict / info. Only the Apsi runtime
    is needed to use it (pip install apsi): never the program that trained the model. embed_key: put this key in
    the library (anyone with the library can then run the model: use it for convenience, not for secrecy)."""
    import re
    import shutil
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(name)):
        raise PackageError("the library's name: letters, digits and _ (a Python module name)")
    package_info(package)                                                  # a package, or a clear error
    d = os.path.join(str(out_dir), str(name))
    os.makedirs(d, exist_ok=True)
    shutil.copyfile(package, os.path.join(d, "model.apsipkg"))
    with open(os.path.join(d, "__init__.py"), "w", encoding="utf-8") as f:
        f.write(_LIB_INIT.format(name=name, key=embed_key,
                                 embedded=" (the key is inside this library)" if embed_key else ""))
    with open(os.path.join(str(out_dir), "pyproject.toml"), "w", encoding="utf-8") as f:
        f.write(f'[project]\nname = "{name}"\nversion = "1.0.0"\ndependencies = ["apsi"]\n\n'
                f'[tool.setuptools]\npackages = ["{name}"]\n\n[tool.setuptools.package-data]\n'
                f'{name} = ["model.apsipkg"]\n\n[build-system]\nrequires = ["setuptools>=61"]\n'
                f'build-backend = "setuptools.build_meta"\n')
    return d


def names():
    return dict(pack_model=pack_model, load_package=load_package, package_info=package_info,
                build_library=build_library, new_key=new_key)


def cmd_package(args):
    from .errors import ApsiError
    try:
        if args.action == "info":
            i = package_info(args.file)
            print(f"{args.file}: {i.format} v{i.version}, {i.cipher}, {'password' if i.password else 'key'}, "
                  f"{i.bytes:,} bytes")
            for k, v in (i.public or {}).items():
                print(f"  {k}: {v}")
        elif args.action == "lib":
            if not args.name:
                raise PackageError("lib needs a name: apsi package lib MODEL.apsipkg NAME [--out DIR]")
            d = build_library(args.file, args.out, args.name, embed_key=args.embed_key)
            print(f"wrote {d} (pip install {args.out} or put {args.name} next to your program; "
                  f"it needs only Apsi: pip install apsi)")
        elif args.action == "key":
            print(new_key())
    except (ApsiError, OSError) as e:
        print(f"error: {e}")
        return 1
    return 0


def add_parser(sub):
    p = sub.add_parser("package", help="sealed model packages: info, a library around one (lib), a new key")
    p.add_argument("action", choices=["info", "lib", "key"])
    p.add_argument("file", nargs="?")
    p.add_argument("name", nargs="?")
    p.add_argument("--out", default="dist", help="lib: the folder to write the library to (default dist)")
    p.add_argument("--embed-key", help="lib: put this key inside the library (convenient, not secret)")
    p.set_defaults(fn=cmd_package)
