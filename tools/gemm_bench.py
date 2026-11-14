"""The matrix products of the 127 M GPT on this card, one by one: every build of the tiled tensor-core product
(x, w, h and the auto choice), each timed over 20 runs on the card's own clock, in TFLOPS. A run takes seconds, so a
change to the product kernels can be measured right away.

    python tools/gemm_bench.py            (from the Apsi folder; writes gemm-bench.txt next to it as well)
"""
import ctypes
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apsi import gpu_train as GT                                   # noqa: E402

# (name, M, N, K, A stored k-contiguous?, B stored n-contiguous?, count per step)
SHAPES = [
    ("q/k/v/o forward      x W", 4096, 1024, 1024, True, True, 40),
    ("mlp up forward       x W", 4096, 4096, 1024, True, True, 10),
    ("mlp down forward     h W", 4096, 1024, 4096, True, True, 10),
    ("q/k/v/o input grad   dy W^T", 4096, 1024, 1024, True, False, 40),
    ("mlp up input grad    dy W^T", 4096, 1024, 4096, True, False, 10),
    ("mlp down input grad  dy W^T", 4096, 4096, 1024, True, False, 10),
    ("q/k/v/o weight grad  x^T dy", 1024, 1024, 4096, False, True, 40),
    ("mlp up weight grad   x^T dy", 1024, 4096, 4096, False, True, 10),
    ("mlp down weight grad h^T dy", 4096, 1024, 4096, False, True, 10),
]


def main():
    ops, why = GT.device("cuda")
    if ops is None:
        print("no GPU:", why)
        return 1
    out = []

    def say(s=""):
        print(s, flush=True)
        out.append(s)
    say(f"{getattr(ops, 'name', '?')}: matrix products of the 127 M GPT, bfloat16 tensor cores, float32 sums")
    rng = np.random.default_rng(0)
    total = {}
    for label, M, N, K, a_kc, b_nc, cnt in SHAPES:
        A, B, Cb = ops.alloc(M * K), ops.alloc(K * N), ops.alloc(M * N)
        sam, sak = (K, 1) if a_kc else (1, M)
        sbk, sbn = (N, 1) if b_nc else (1, K)
        row = []
        a_h = rng.standard_normal(M * K).astype(np.float32)
        b_h = rng.standard_normal(K * N).astype(np.float32)
        ops.put(A, a_h)
        ops.put(B, b_h)
        Am = a_h.reshape(M, K) if a_kc else a_h.reshape(K, M).T
        Bm = b_h.reshape(K, N) if b_nc else b_h.reshape(N, K).T
        ref = Am[:256].astype(np.float64) @ Bm.astype(np.float64)       # the first 256 rows, to check every build
        for v in ("auto", "x", "w", "h", "v", "a"):
            os.environ["APSI_GEMM_VARIANT"] = v
            ops._choice = {}
            ops.fill((Cb, 0), M * N, 0.0)
            ops.gemm((A, 0), (B, 0), (Cb, 0), M, N, K, sam, sak, sbk, sbn, N)
            got = ops.get(Cb, 256 * N).reshape(256, N)
            err = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30))
            for _ in range(3):
                ops.gemm((A, 0), (B, 0), (Cb, 0), M, N, K, sam, sak, sbk, sbn, N, beta=(not a_kc))
            ops.sync()
            e0, e1 = ops._event(), ops._event()
            ops._lib.cuEventRecord(e0, None)
            reps = 20
            for _ in range(reps):
                ops.gemm((A, 0), (B, 0), (Cb, 0), M, N, K, sam, sak, sbk, sbn, N, beta=(not a_kc))
            ops._lib.cuEventRecord(e1, None)
            ops._lib.cuEventSynchronize(e1)
            ms = ctypes.c_float()
            ops._lib.cuEventElapsedTime(ctypes.byref(ms), e0, e1)
            t = ms.value / reps
            row.append((v, t, 2.0 * M * N * K / (t / 1000.0) / 1e12, err))
        os.environ.pop("APSI_GEMM_VARIANT", None)
        bad = [r[0] for r in row if r[3] > 2e-2]
        best = min((r for r in row if r[3] <= 2e-2), key=lambda r: r[1])
        total[label] = best[1] * cnt
        say(f"{label:30s} {M}x{N}x{K}  " + "  ".join(f"{v}:{tf:5.1f}TF" for v, _, tf, _e in row) +
            (f"  WRONG: {','.join(bad)}" if bad else "") + f"   best {best[0]} {best[1]:.3f} ms x{cnt} = {best[1] * cnt:.1f} ms/step")
        for bf in (A, B, Cb):
            ops.free(bf)
    say(f"all products of one step at the best build: {sum(total.values()):.1f} ms")
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gemm-bench.txt"),
                  "w") as fh:
            fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + "\n".join(out) + "\n")
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
