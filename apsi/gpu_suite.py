"""`apsi gpu test`: everything Apsi does on an NVIDIA card, checked on this machine, in one report.

    apsi gpu test                   # report in ./apsi-gpu-test/ (report.txt and report.json)
    apsi gpu test --out DIR --quick # the checks only, no speed runs

Every part prints PASS or FAIL and goes on after a failure:
  1. the card: driver, temperature, power
  2. every kernel and layer on the card against the exact reference (the same as `apsi gpu bench check`)
  3. matrix products: every build's error and speed (`apsi gpu bench gemm`)
  4. an Apsi program on the card: a transformer trained with a checkpoint, stopped, resumed to the same loss as one
     uninterrupted run, predicted with, saved and read back; bfloat16 moments; a graph network; gradient
     accumulation with batches of their own shape
  5. speed: the 11 M and 127 M GPTs and the LLaMA / Qwen3-style model (`apsi gpu bench`)
  6. the largest GPT that trains on this card (`apsi gpu bench max`)
  7. several cards, when there are several (`apsi gpu bench cards`)
"""
import io
import json
import os
import sys
import tempfile
import textwrap
import time
import traceback

PROGRAM = textwrap.dedent('''
    use layers
    use nn
    let ids = randint(0, 64, [96, 33], seed=1)
    let X = ids[:, 0:32]
    let Y = ids[:, 1:33]
    let lm = transformer_lm(64, d=64, heads=4, layers=2, context=32, kv_heads=2, rope=true, norm="rms", ffn="swiglu")
    let p0 = lm.init(seed=0)
    fn batches(epoch) = [(X[i * 8:(i + 1) * 8], Y[i * 8:(i + 1) * 8]) for i in 0..12]
    # one uninterrupted run of 24 steps
    let whole = fit_batches(lm, p0, batches, loss="cross_entropy", opt=adamw(lr=3e-3), steps=24, backend="gpu",
                            verbose=false)
    # the same run stopped after 12 steps and resumed from its checkpoint
    let half = fit_batches(lm, p0, batches, loss="cross_entropy", opt=adamw(lr=3e-3), steps=12, backend="gpu",
                           checkpoint="run.npz", save_every=6, verbose=false)
    let rest = fit_batches(lm, p0, batches, loss="cross_entropy", opt=adamw(lr=3e-3), steps=24, backend="gpu",
                           checkpoint="run.npz", save_every=6, verbose=false)
    let a = whole.history[-1]
    let b = rest.history[-1]
    print(f"RESULT resume: one run {a:.6f}, stopped and resumed {b:.6f}")
    assert abs(a - b) <= 1e-4 * max(1.0, abs(a)), "a resumed run must end where one run ends"
    assert whole.history[-1] < whole.history[0], "the loss must fall"
    let out = predict(lm, whole.params, X[0:4], backend="gpu")
    print(f"RESULT predict: shape {shape(out)}")
    save_params("lm.npz", whole.params)
    let back = load_params("lm.npz", lm.init(seed=0))
    let out2 = predict(lm, back, X[0:4], backend="gpu")
    assert max(abs(out - out2)) < 1e-5, "saved weights must give the same answers"
    print("RESULT save and load: same answers")
    let half_state = fit_batches(lm, p0, batches, loss="cross_entropy", opt=adamw(lr=3e-3, state="bf16"),
                                 steps=24, backend="gpu", verbose=false)
    let rel = abs(half_state.history[-1] - a) / abs(a)
    print(f"RESULT bfloat16 moments: loss {half_state.history[-1]:.4f} against {a:.4f} ({100 * rel:.2f} %)")
    assert rel < 0.05, "bfloat16 moments must follow float32 moments closely"
    let acc = fit_batches(lm, p0, batches, loss="cross_entropy", opt=adamw(lr=3e-3), steps=6, accumulate=4,
                          backend="gpu", verbose=false)
    print(f"RESULT accumulate 4: loss {acc.history[0]:.4f} -> {acc.history[-1]:.4f}")
    let n = 80
    let g = graph(edges=randint(0, n, [320, 2], seed=3), n=n)
    let feats = randn([n, 12], seed=4)
    let labels = randint(0, 3, [n], seed=5)
    let net = gat_net(12, 16, 3, dropout=0.0)
    let gr = fit(net, net.init(seed=0), feats, labels, loss="cross_entropy", opt=adam(lr=0.01), epochs=20,
                 graph=g, backend="gpu", verbose=false)
    print(f"RESULT graph network: loss {gr.history[0]:.4f} -> {gr.history[-1]:.4f}, card memory {gr.gpu_memory_mb} MB")
    assert gr.history[-1] < gr.history[0], "the graph network must learn"
    print("ALL OK")
''')


def _program(out):
    """Part 4: an Apsi program on the card, through the language itself."""
    from .interpreter import Interpreter
    d = tempfile.mkdtemp(prefix="apsi-gpu-test-")
    path = os.path.join(d, "gpu_program.apsi")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(PROGRAM)
    old = os.getcwd()
    os.chdir(d)
    buf = io.StringIO()
    real = sys.stdout
    try:
        sys.stdout = buf
        it = Interpreter()
        it.run(PROGRAM, path)
    finally:
        sys.stdout = real
        os.chdir(old)
    text = buf.getvalue()
    out.write(text)
    return 0 if "ALL OK" in text else 1


def run(out_dir="apsi-gpu-test", quick=False):
    from . import gpu_bench, gpu_train as GT
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, "report.txt")
    log = open(txt_path, "w", encoding="utf-8")

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            sys.__stdout__.flush()
            log.write(s)
            log.flush()

        def flush(self):
            pass
    tee = Tee()
    report = {"apsi": __import__("apsi").__version__, "python": sys.version.split()[0],
              "started": time.strftime("%Y-%m-%d %H:%M:%S"), "parts": []}

    def part(name, fn):
        tee.write(f"\n== {name}\n")
        t0 = time.perf_counter()
        extra = None
        try:
            rc = fn(tee)
            if isinstance(rc, tuple):
                rc, extra = rc
            ok = rc in (0, None)
        except Exception:                                   # noqa: BLE001 - one failed part never stops the rest
            tee.write(traceback.format_exc())
            ok = False
        dt = time.perf_counter() - t0
        tee.write(f"== {'PASS' if ok else 'FAIL'}: {name} ({dt:.1f} s)\n")
        report["parts"].append({"part": name, "ok": ok, "seconds": round(dt, 1), **({"result": extra} if extra else {})})
        return ok

    def card(o):
        from . import thermal
        rep = thermal.gpu_report()
        for r in rep or []:
            o.write(f"GPU {r['index']} {r['name']}: {r['temperature_c']} C, {r['power_w']} W of {r['power_limit_w']} W, "
                    f"{r['graphics_mhz']} MHz\n")
        ops, why = GT.device("cuda")
        if ops is None:
            o.write(f"no GPU for Apsi: {why}\n")
            return 1
        o.write(f"Apsi's trainer on: {getattr(ops, 'name', '?')}\n")
        report["card"] = (rep or [{}])[0].get("name")
        return 0

    have = part("1. the card", card)
    if not have:
        tee.write("\nno usable NVIDIA card: the remaining parts need one.\n")
    else:
        part("2. every kernel and layer against the exact reference", lambda o: gpu_bench.check(out=o))
        part("3. matrix products: every build", lambda o: gpu_bench.gemm(out=o))
        part("4. an Apsi program on the card (train, resume, predict, save, bf16 moments, graphs)", _program)
        if not quick:
            for which in ("11m", "127m", "modern"):
                part(f"5. speed: {which}", lambda o, w=which: gpu_bench.bench(w, 30, "auto", out=o))
            part("6. the largest GPT that trains here", lambda o: gpu_bench.largest(out=o))
        try:
            import ctypes  # noqa: F401
            n_cards = len(__import__("apsi.thermal", fromlist=["x"]).gpu_report() or [])
        except Exception:                                   # noqa: BLE001
            n_cards = 1
        if n_cards > 1:
            part("7. several cards", lambda o: gpu_bench.cards(out=o))
    passed = sum(p["ok"] for p in report["parts"])
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["passed"], report["total"] = passed, len(report["parts"])
    tee.write(f"\nRESULT: {passed} of {len(report['parts'])} parts passed"
              + ("" if passed == len(report["parts"]) else "  <-- see the FAIL lines") + "\n")
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    tee.write(f"report: {os.path.abspath(txt_path)}\n")
    log.close()
    return 0 if passed == len(report["parts"]) else 1
