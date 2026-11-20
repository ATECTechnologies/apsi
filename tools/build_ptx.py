"""Builds the PTX of every GPU training module from apsi/gpu_train_kernels.py into apsi/ptx/ (gzip), keyed by the
source file's hash: shipped with Apsi so a card needs no LLVM build. Run after changing the kernels:
    python tools/build_ptx.py"""
import glob
import gzip
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["APSI_NO_CACHE"] = "1"
from apsi import gpu_train_kernels as K  # noqa: E402

out = os.path.join(os.path.dirname(K.__file__), "ptx")
os.makedirs(out, exist_ok=True)
with open(K.__file__, "rb") as fh:
    src = hashlib.sha256(fh.read()).hexdigest()[:16]
for old in glob.glob(os.path.join(out, "*.ptx.gz")):
    if not old.endswith(f"-{src}.ptx.gz"):
        os.remove(old)
for which in ("main", "tc", "tiled", "tiled_x", "tiled_w", "tiled_h", "tiled_v", "tiled_d", "tiled_dw", "tiled_d2",
              "tiled_dw2", "tiled_sw", "flash"):
    K._cache.pop(which, None)
    text = K.ptx(which)
    path = os.path.join(out, f"{which}-{src}.ptx.gz")
    with gzip.open(path, "wt", encoding="ascii", compresslevel=9) as fh:
        fh.write(text)
    print(f"{which:10} {len(text) / 1e6:6.1f} MB -> {os.path.getsize(path) / 1e6:5.2f} MB")
