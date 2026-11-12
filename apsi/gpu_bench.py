"""apsi gpu bench: how fast Apsi's own GPU trainer trains on this machine, and whether every kernel is right here.

    apsi gpu bench              the 127 M-parameter GPT (10 blocks, d=1024, 16 heads, context 128, batch 32), bf16
    apsi gpu bench 11m          the 10.9 M GPT of docs/GPU.md
    apsi gpu bench modern       a 127 M LLaMA / Qwen3-style model (grouped heads, rotary positions, q/k norms, SwiGLU)
    apsi gpu bench check        every new kernel and layer on this card against the exact reference (the Emulator)
    apsi gpu bench cards        one model split across every card against one card (needs 2 cards or more)

Reports tokens per second (steady, after the first steps), the time of each kind of kernel, the temperature, the
power and the energy per token, and ends with one JSON line to send back when asking for tuning. The data is random
tokens made here: nothing is downloaded."""
import json
import sys
import threading
import time

import numpy as np

CONFIGS = {
    "127m": dict(vocab=256, d=1024, heads=16, layers=10, context=128, batch=32),
    "11m": dict(vocab=256, d=384, heads=6, layers=6, context=128, batch=32),
    "modern": dict(vocab=256, d=1024, heads=16, layers=10, context=128, batch=32, kv_heads=4, rope=True,
                   qk_norm=True, norm="rms", ffn="swiglu"),
}


class _Sampler:
    """Temperature and power of every card, five times a second, while a run goes on."""

    def __init__(self):
        from . import thermal
        self.rep, self.rows, self.stop = thermal.gpu_report, [], threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                self.rows.append((time.perf_counter(), self.rep()))
            except Exception:                               # noqa: BLE001
                pass
            self.stop.wait(0.2)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.t.join(timeout=2)

    def summary(self):
        if not self.rows:
            return {}
        temps = [r[0]["temperature_c"] for _, r in self.rows if r and r[0].get("temperature_c") is not None]
        pw = [r[0]["power_w"] for _, r in self.rows if r and r[0].get("power_w")]
        mhz = [r[0]["graphics_mhz"] for _, r in self.rows if r and r[0].get("graphics_mhz")]
        out = {"card": self.rows[-1][1][0]["name"] if self.rows[-1][1] else None}
        if temps:
            out.update(temperature_c_mean=round(float(np.mean(temps)), 1), temperature_c_max=max(temps))
        if pw:
            out.update(power_w_mean=round(float(np.mean(pw)), 1), power_w_max=max(pw))
        if mhz:
            out.update(clock_mhz_mean=round(float(np.mean(mhz))))
        return out


def _data(cfg, steps, seed=0):
    rng = np.random.default_rng(seed)
    n = cfg["batch"] * steps
    ids = rng.integers(0, cfg["vocab"], (n, cfg["context"] + 1))
    return ids[:, :-1], ids[:, 1:]


def bench(which="127m", steps=40, precision="auto", out=sys.stdout):
    from . import layers as L, gpu_train as GT
    from .stdlib import adamw
    cfg = dict(CONFIGS[which])
    ops, why = GT.device("cuda")
    if ops is None:
        print(f"no GPU: {why}", file=out)
        return 1
    block = {k: cfg.pop(k) for k in ("kv_heads", "rope", "qk_norm", "norm", "ffn") if k in cfg}
    batch = cfg.pop("batch")
    model = L.transformer_lm(cfg["vocab"], d=cfg["d"], heads=cfg["heads"], layers=cfg["layers"],
                             context=cfg["context"], **block)
    params = model.init(seed=0)
    count = sum(int(np.asarray(a).size) for a in __import__("apsi.values", fromlist=["x"]).tree_leaves(params)
                if isinstance(a, np.ndarray))
    X, Y = _data(dict(cfg, batch=batch), steps)
    tokens = X.size
    print(f"{which}: {count / 1e6:.1f} M parameters, batch {batch} x {cfg['context']}, {steps} steps, "
          f"precision {precision}, {getattr(ops, 'name', '?')}", file=out)
    warm = 3                                                  # first steps: the products are timed and chosen
    GT.fit(model, params, X[:warm * batch], Y[:warm * batch], loss="cross_entropy", opt=adamw(lr=1e-3), epochs=1,
           batch_size=batch, verbose=False, precision=precision, ops=ops)
    with _Sampler() as smp:
        ops.sync()
        t0 = time.perf_counter()
        r = GT.fit(model, params, X, Y, loss="cross_entropy", opt=adamw(lr=1e-3), epochs=1, batch_size=batch,
                   verbose=False, precision=precision, ops=ops)
        ops.sync()
        wall = time.perf_counter() - t0
    tps = tokens / wall
    card = smp.summary()
    # the kernels of a few steps, timed on the card's own clock
    prof = {}
    try:
        ops.start_profile()
        GT.fit(model, r.params, X[:4 * batch], Y[:4 * batch], loss="cross_entropy", opt=adamw(lr=1e-3), epochs=1,
               batch_size=batch, verbose=False, precision=precision, ops=ops)
        prof = ops.stop_profile()
    except Exception as e:                                    # noqa: BLE001
        print(f"(no kernel profile: {e})", file=out)
    k_total = sum(t for t, _ in prof.values())
    k_tps = 4 * batch * cfg["context"] / k_total if k_total else None
    print(f"\n  {tps:,.0f} tokens/s over {wall:.1f} s (loss {r.history[0]:.3f})", file=out)
    if k_tps:
        print(f"  {k_tps:,.0f} tokens/s if the card did nothing but these kernels "
              f"(the rest is waiting: launches, the processor, cooling)", file=out)
    if card.get("temperature_c_mean") is not None:
        line = f"  {card.get('card')}: {card.get('temperature_c_mean')} C mean, {card.get('temperature_c_max')} C max"
        if card.get("power_w_mean"):
            jt = card["power_w_mean"] / tps
            card["joules_per_1k_tokens"] = round(1000 * jt, 2)
            line += f", {card['power_w_mean']} W mean ({1000 * jt:.2f} J per 1,000 tokens)"
        if card.get("clock_mhz_mean"):
            line += f", {card['clock_mhz_mean']} MHz"
        print(line, file=out)
    if r.gpu_temperature:
        print(f"  temperature control: {r.gpu_temperature}", file=out)
    if prof:
        groups = {}
        for name, (t, c) in prof.items():
            g = name.split(" ")[0]
            gt, gc = groups.get(g, (0.0, 0))
            groups[g] = (gt + t, gc + c)
        print("\n  kernel                       share    ms/step", file=out)
        for g, (t, c) in sorted(groups.items(), key=lambda kv: -kv[1][0])[:14]:
            print(f"  {g:<28} {100 * t / k_total:5.1f} %  {1000 * t / 4:8.2f}", file=out)
    rec = {"bench": which, "params_m": round(count / 1e6, 1), "precision": precision, "tokens_per_s": round(tps),
           "kernel_tokens_per_s": round(k_tps) if k_tps else None, "card": card,
           "kernels_ms_per_step": {g: round(1000 * t / 4, 3) for g, (t, _) in prof.items()} if prof else {},
           "apsi": __import__("apsi").__version__}
    print("\nsend this line back for tuning:\n" + json.dumps(rec, separators=(",", ":")), file=out)
    return 0


def check(out=sys.stdout):
    """The new layers on this card against the Emulator: a short training each, the same losses expected."""
    from . import layers as L, gpu_train as GT, graphs as GR
    from .stdlib import Adam
    ops, why = GT.device("cuda")
    if ops is None:
        print(f"no GPU: {why}", file=out)
        return 1
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 50, (32, 24))
    lab = np.roll(ids, -1, 1)
    N = 60
    e = rng.integers(0, N, (240, 2)).tolist()
    g = L.graph(edges=e, n=N, types=rng.integers(0, 3, 240), batch=np.repeat(np.arange(4), 15))
    Xg, yg, mask = rng.normal(size=(N, 16)), rng.integers(0, 3, N), rng.random(N) < 0.6
    cases = [
        ("gpt", L.transformer_lm(50, d=64, heads=4, layers=2, context=24), ids, lab, {}),
        ("llama/qwen3 block", L.transformer_lm(50, d=64, heads=4, layers=2, context=24, kv_heads=2, rope=True,
                                               qk_norm=True, norm="rms", ffn="swiglu"), ids, lab, {}),
        ("mixture of experts", L.transformer_lm(50, d=64, heads=4, layers=2, context=24, ffn="moe", experts=4,
                                                top_k=2, norm="rms"), ids, lab, {}),
        ("gat", L.gat_net(16, 8, 3, dropout=0.0), Xg, yg, dict(graph=g, mask=mask)),
        ("gcn / gin", L.Sequential([L.GCN(16, 32), L.Activation("elu"), L.GIN(32, 3)]), Xg, yg,
         dict(graph=g, mask=mask)),
        ("graph transformer", GR.graph_transformer_net(16, 32, 3, layers=2, heads=4, relations=3, max_keys=8), Xg,
         yg, dict(graph=g, mask=mask)),
    ]
    bad = 0
    for name, m, X, Y, kw in cases:
        runs = []
        for o in (ops, GT.Emulator(tc=ops.tc)):
            r = GT.fit(m, m.init(seed=0), X, Y, loss="cross_entropy", opt=Adam(3e-3), epochs=3,
                       batch_size=None if kw else 8, verbose=False, precision="fp32", ops=o, **kw)
            runs.append(np.asarray(r.history))
        err = float(np.max(np.abs(runs[0] - runs[1]) / (np.abs(runs[1]) + 1e-6)))
        ok = err < 1e-3
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<22} losses {runs[0][0]:.4f} -> {runs[0][-1]:.4f}  "
              f"(reference {runs[1][-1]:.4f}, difference {err:.1e})", file=out)
    print("every kernel agrees on this card" if not bad else f"{bad} differ: send this output back", file=out)
    return 1 if bad else 0


def cards(out=sys.stdout):
    """One model split across every card against the same training on one card."""
    from . import layers as L, gpu_train as GT
    from .stdlib import Adam
    many = GT.devices_for("cuda:all") or []
    if len(many) < 2:
        print(f"{len(many)} card(s): the split needs two or more", file=out)
        return 1
    rng = np.random.default_rng(0)
    m = L.transformer_lm(256, d=512, heads=8, layers=8, context=128, kv_heads=2, rope=True, norm="rms",
                         ffn="swiglu")
    X, Y = _data(dict(vocab=256, context=128, batch=16), 8)
    kw = dict(loss="cross_entropy", epochs=2, batch_size=16, verbose=False, precision="fp32")
    t0 = time.perf_counter()
    one = GT.fit(m, m.init(seed=0), X, Y, opt=Adam(1e-3), ops=many[0], **kw)
    t1 = time.perf_counter()
    split = GT.fit(m, m.init(seed=0), X, Y, opt=Adam(1e-3), ops=many, parallel="pipeline", **kw)
    t2 = time.perf_counter()
    err = float(np.max(np.abs(np.asarray(one.history) - split.history)))
    print(f"  one card: {one.history}  {t1 - t0:.1f} s", file=out)
    print(f"  {split.device}: {split.history}  {t2 - t1:.1f} s", file=out)
    print(f"  stages {split.stages}, replicas {split.replicas}, micro-batches {split.micro_batches}; "
          f"difference {err:.1e} ({'ok' if err < 1e-3 else 'DIFFERENT'})", file=out)
    return 0 if err < 1e-3 else 1


def main(what="127m", precision="auto", steps=40):
    w = str(what or "127m").lower()
    if w == "check":
        return check()
    if w == "cards":
        return cards()
    if w not in CONFIGS:
        print(f"apsi gpu bench {what}: {', '.join(CONFIGS)}, check or cards")
        return 2
    return bench(w, int(steps), precision)
