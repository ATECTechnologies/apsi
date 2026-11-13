"""Where a training step's time goes on this card: queued copies against copies that wait, alternated, warm."""
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apsi import gpu_train as GT, layers as L          # noqa: E402
from apsi.stdlib import Adam                             # noqa: E402

ops, why = GT.device("cuda")
print("device", getattr(ops, "name", ops), why)
print("tiled:", getattr(ops, "tiled", None), "| builds:", sorted({k.split("#")[1] for k in ops.fn if "#" in k}),
      "| variant_why:", getattr(ops, "variant_why", None), "| tc_why:", getattr(ops, "tc_why", None))
sys.stdout.flush()
from apsi import gpu_bench as B                          # noqa: E402
B._check_flow(ops, sys.stdout)
sys.stdout.flush()
rng = np.random.default_rng(4)
lm = L.transformer_lm(256, d=128, heads=2, layers=2, context=128)
batches = [(rng.integers(0, 256, (8, 128)), rng.integers(0, 256, (8, 128))) for _ in range(24)]
stats = {"put": 0.0, "wait": 0.0, "n": 0}
orig_put, orig_async = GT.Gpu.put, GT.Gpu._put_async


def put(self, b, a, off=0):
    t0 = time.perf_counter()
    orig_put(self, b, a, off)
    stats["put"] += time.perf_counter() - t0
    stats["n"] += 1


GT.Gpu.put = put


def run(mode, label, fresh=True):
    os.environ["APSI_GPU_ASYNC_COPY"] = mode
    if fresh:
        ops.drop_ring()
    stats.update(put=0.0, n=0)
    ops.sync()
    t0 = time.perf_counter()
    r = GT.fit_batches(lm, lm.init(seed=0), batches, "cross_entropy", opt=Adam(1e-3), epochs=2, verbose=False,
                       precision="auto", ops=ops)
    ops.sync()
    dt = time.perf_counter() - t0
    r_ = ops.__dict__.get("_ring")
    print(f"{label:<28} {1e3 * dt / 48:7.2f} ms a step   puts {stats['n']:4} taking {1e3 * stats['put']:7.1f} ms   "
          f"loss {r.history[-1]:.5f}   queue {len(r_['q']) if r_ else '-'}")
    return dt


run("0", "warm-up")
for rep in range(3):
    run("0", "copies that wait")
    run("1", "queued copies")
    run("1", "queued copies, ring kept", fresh=False)
ops.start_profile()
run("1", "queued copies, profiled")
prof = ops.stop_profile()
tot = sum(v[0] for v in prof.values())
print(f"kernels on the card: {1e3 * tot / 48:.2f} ms a step")
for k, v in sorted(prof.items(), key=lambda kv: -kv[1][0])[:15]:
    print(f"  {k:<40} {1e3 * v[0]:8.2f} ms  x{v[1]}")
