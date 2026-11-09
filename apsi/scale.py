"""`use scale` and `apsi plan`: plan training and inference of large models (1B … 2T parameters).

    use scale
    let p = plan(params="1T", active="32B", tokens="15T", gpu="H100", days=60)
    print(p.gpus, p.layout, p.memory_per_gpu_gb, p.days)

Everything is computed from standard, published accounting, so a plan can be checked by hand:

* **Compute**: training FLOPs = 6·N_active·D + 12·L·d·s·D (the attention term), N = parameters, D = tokens,
  L = layers, d = model width, s = sequence length (Kaplan et al. 2020; Hoffmann et al. 2022, "Chinchilla").
  Time = FLOPs / (GPUs × peak × MFU); MFU (model FLOPs utilisation) defaults to 40 %.
* **Model states** (mixed precision AdamW): 2 bytes weights + 2 bytes gradients + 12 bytes FP32 master weights and
  two Adam moments = 16 bytes per parameter (Rajbhandari et al. 2020, "ZeRO"), sharded across all GPUs with
  ZeRO-3 / FSDP. 8-bit Adam moments: 10 bytes. FP8 training keeps BF16 master weights in the same budget.
* **Activations** per layer per micro-batch (Korthikanti et al. 2022): s·b·d·(34 + 5·a·s/d) bytes without
  recomputation, 34·s·b·d with selective recomputation, 2·s·b·d with full recomputation — divided by the tensor-
  parallel degree (sequence parallelism); with 1F1B pipelining the first stage holds up to PP micro-batches.
* **Layout**: tensor parallel (TP, within a node, ≤ 8) × pipeline parallel (PP) × data parallel (DP). The planner
  picks the cheapest layout and recomputation level whose memory fits in 90 % of each GPU.
* **Mixture of experts**: memory counts all parameters (`params`), compute counts the active ones (`active`).
"""
import math

GPUS = {
    # name: (memory GB, dense BF16 TFLOPS, dense FP8 TFLOPS or None, NVLink domain = max tensor parallel)
    "GTX1660": (6, 10.0, None, 1),          # FP16 tensor-less Turing: practical FP16 throughput
    "RTX3090": (24, 71.0, None, 2),         # consumer cards: tensor parallel over PCIe only pays off up to 2
    "RTX4090": (24, 165.0, 330.0, 2),
    "RTX5090": (32, 210.0, 420.0, 2),
    "A100-40": (40, 312.0, None, 8),
    "A100": (80, 312.0, None, 8),
    "L40S": (48, 362.0, 733.0, 8),
    "H100": (80, 989.0, 1979.0, 8),
    "H200": (141, 989.0, 1979.0, 8),
    "B200": (192, 2250.0, 4500.0, 8),
    "GB200": (186, 2500.0, 5000.0, 72),
    "MI300X": (192, 1307.0, 2615.0, 8),
    "MI325X": (256, 1307.0, 2615.0, 8),
}

_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9, "G": 1e9, "T": 1e12, "P": 1e15}


def number(x):
    """'21B' -> 2.1e10, '1T' -> 1e12, '15T' tokens, or a plain number."""
    if isinstance(x, (int, float)):
        return float(x)
    t = str(x).strip().upper().replace("_", "").replace(",", "")
    if t and t[-1] in _SUFFIX:
        return float(t[:-1]) * _SUFFIX[t[-1]]
    return float(t)


def _shape(n):
    """A typical transformer shape for n parameters: n ≈ 12·L·d², d/L ≈ 128 (rounded to multiples of 128)."""
    d = (n * 128 / 12) ** (1 / 3)
    layers = max(2, n / (12 * d * d))
    layers = int(round(layers / 8)) * 8 if layers > 12 else int(round(layers))   # pipeline-friendly depth
    heads = max(8, int(round((n / (12 * layers)) ** 0.5 / 128 / 8)) * 8)   # head size 128, heads % 8 == 0
    d = heads * 128
    return layers, d, heads


def _fmt(n):
    for suf, v in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= v:
            return f"{n / v:.3g}{suf}"
    return f"{n:.3g}"


def plan(params, tokens=None, gpu="H100", gpus=None, days=None, active=None, layers=None, width=None, heads=None,
         seq=4096, batch_tokens=None, precision="bf16", optimizer="adamw", mfu=0.40, gpu_memory=None,
         micro_batch=1, gpus_per_node=8, max_gpus=200_000):
    """Plan a training run. Returns a record with the layout, memory per GPU, GPU count and duration.

    params: total parameters ("21B", "1T", 2e12). active: active parameters per token for mixture-of-experts.
    tokens: training tokens (default: 20 per active parameter, the Chinchilla-optimal amount).
    gpu: a name from scale.gpu_table() or a record(memory=…, tflops=…). Give `gpus` (count) or `days` (target;
    default 90 days when neither is given).
    precision: "bf16" or "fp8". optimizer: "adamw" (16 B/param), "adam8bit" (10 B/param), "sgd" (8 B/param).
    """
    from .values import Record
    n_total = number(params)
    n_active = number(active) if active is not None else n_total
    if n_active > n_total:
        raise ValueError("active parameters cannot exceed total parameters")
    d_tokens = number(tokens) if tokens is not None else 20 * n_active
    L0, d0, h0 = _shape(n_active if active is not None else n_total)
    L = int(layers) if layers else L0
    d = int(width) if width else d0
    a = int(heads) if heads else h0
    s = int(seq)
    if isinstance(gpu, str):
        key = gpu.upper().replace(" ", "").replace("NVIDIA", "")
        key = {"H100SXM": "H100", "A10080": "A100", "A100-80": "A100", "4090": "RTX4090", "3090": "RTX3090",
               "1660": "GTX1660", "5090": "RTX5090"}.get(key, key)
        if key not in GPUS:
            raise ValueError(f"unknown GPU {gpu!r}; known: {', '.join(GPUS)} (or pass gpu_memory= and a record)")
        mem_gb, bf16, fp8, node = GPUS[key]
        gname = key
    else:
        mem_gb, bf16, fp8, node, gname = gpu["memory"], gpu["tflops"], None, 8, "custom"
    if gpu_memory:
        mem_gb = float(gpu_memory)
    if gpus is None and days is None:
        days = 90
    peak = (fp8 if precision == "fp8" and fp8 else bf16) * 1e12
    if precision == "fp8" and not fp8:
        precision = "bf16"
    bytes_per_param = {"adamw": 16.0, "adam": 16.0, "adam8bit": 10.0, "sgd": 8.0}.get(optimizer)
    if bytes_per_param is None:
        raise ValueError("optimizer must be adamw, adam8bit or sgd")
    budget = 0.90 * mem_gb * 2 ** 30
    flops = 6 * n_active * d_tokens + 12 * L * d * s * d_tokens
    gbatch = number(batch_tokens) if batch_tokens else max(s, min(16e6, 2 ** round(math.log2(
        max(s, 2e6 * (n_active / 1e9) ** 0.25)))))           # common practice: ~2–16M tokens per step
    micro_tokens = s * micro_batch

    best = None
    max_dp_batch = max(1, int(gbatch // micro_tokens))     # data parallel cannot exceed sequences per step
    for tp in (1, 2, 4, 8):
        if tp > node or d % tp or a % tp:
            continue
        for pp in [p for p in range(1, min(L, 128) + 1) if L % p == 0]:
            for recompute, per_layer in (("none", s * micro_batch * d * (34 + 5 * a * s / d)),
                                         ("selective", 34.0 * s * micro_batch * d),
                                         ("full", 2.0 * s * micro_batch * d)):
                act = per_layer / tp * (L / pp) * min(pp, 8)
                # data-parallel degree needed to shard the model states (ZeRO-3 / FSDP) into the rest of memory
                free = budget - act - 2 * 2 ** 30                        # 2 GiB for buffers / fragmentation
                if free <= 0:
                    continue
                states = n_total * bytes_per_param / (tp * pp)
                dp_min = max(1, math.ceil(states / free))
                if dp_min > max_dp_batch:
                    continue
                g_fit = tp * pp * dp_min
                penalty = {"none": 1.0, "selective": 1.02, "full": 1.33}[recompute]  # extra compute
                if gpus is not None:
                    if int(gpus) % (tp * pp) or int(gpus) // (tp * pp) < dp_min:
                        continue
                    g = int(gpus)
                else:
                    dp = dp_min
                    for _ in range(3):                                   # the bubble depends on dp
                        bub = 1 + (pp - 1) / max(1, gbatch / micro_tokens / dp)
                        g_time = flops * penalty * bub / (peak * mfu * float(days) * 86400)
                        dp = max(dp_min, math.ceil(g_time / (tp * pp)))
                    dp = min(dp, max_dp_batch, max(dp_min, int(max_gpus) // (tp * pp)))   # batch / cluster limits
                    g = tp * pp * dp
                dp = g // (tp * pp)
                if dp > max_dp_batch:
                    continue
                bubble = 1 + (pp - 1) / max(1, gbatch / micro_tokens / dp)      # pipeline idle time
                eff_days = flops * penalty * bubble / (g * peak * mfu) / 86400
                mem = (states / dp + act + 2 * 2 ** 30) / 2 ** 30
                if gpus is None:                  # meet the target first, then use as few GPUs as possible
                    cost = (round(max(eff_days / float(days), 1.0), 2), g, eff_days)
                else:
                    cost = (eff_days, tp * pp, g)
                if best is None or cost < best[0]:
                    best = (cost, tp, pp, dp, recompute, mem, eff_days, g, g_fit)
                break                                                    # least recompute that fits wins
    if best is None and batch_tokens is None and gbatch < 64e6:
        # very large models: larger global batches (up to 64M tokens, as in practice) keep pipelines busy
        return plan(params, tokens, gpu, gpus, days, active, layers, width, heads, seq, min(64e6, gbatch * 2),
                    precision, optimizer, mfu, gpu_memory, micro_batch, gpus_per_node, max_gpus)
    if best is None:
        raise ValueError(f"a {_fmt(n_total)}-parameter model cannot be laid out on {gname} ({mem_gb} GB) "
                         f"with these settings (memory, or data parallelism wider than the batch); try more GPUs, "
                         f"a GPU with more memory, a larger batch_tokens or a shorter seq")
    _, tp, pp, dp, recompute, mem, eff_days, g, g_fit = best
    notes = []
    if days is not None and gpus is None and eff_days > 1.15 * float(days):
        notes.append(f"the {days:g}-day target is not reachable on {gname} with at most {int(max_gpus):,} GPUs; "
                     f"the fastest layout takes {eff_days:.0f} days")
    if g > 100_000:
        notes.append(f"{g:,} GPUs is at the scale of the largest clusters built so far (~100,000–200,000 GPUs)")
    if n_active == n_total and n_total >= 400e9:
        notes.append("frontier models of this size are usually mixture-of-experts: e.g. active=\"37B\" cuts compute "
                     "about " + f"{n_total / 37e9:.0f}x while memory stays the same")
    return Record(
        params=n_total, active=n_active, tokens=d_tokens, layers=L, width=d, heads=a, seq=s,
        gpu=gname, gpu_memory_gb=mem_gb, gpus=g, min_gpus=g_fit, nodes=math.ceil(g / gpus_per_node),
        layout=f"TP {tp} x PP {pp} x DP {dp}", tensor_parallel=tp, pipeline_parallel=pp, data_parallel=dp,
        recompute=recompute, memory_per_gpu_gb=round(mem, 1), precision=precision, optimizer=optimizer,
        flops=flops, days=round(eff_days, 2), gpu_hours=round(eff_days * 24 * g),
        batch_tokens=gbatch, steps=math.ceil(d_tokens / gbatch),
        model_states_gb=round(n_total * bytes_per_param / 2 ** 30, 1),
        dataset_gb=round(d_tokens * 4 / 2 ** 30, 1), notes=notes,
        summary=(f"{_fmt(n_total)} params" + (f" ({_fmt(n_active)} active)" if n_active != n_total else "")
                 + f", {_fmt(d_tokens)} tokens: {g:,} x {gname} ({math.ceil(g / gpus_per_node):,} node"
                 f"{'s' if math.ceil(g / gpus_per_node) != 1 else ''}), {tp}x{pp}x{dp} TPxPPxDP, "
                 f"{recompute} recompute, {mem:.0f} GB/GPU, {eff_days:.1f} days at {mfu:.0%} MFU"))


def inference(params, gpu="H100", precision="bf16", context=8192, batch=1, kv_heads=None, layers=None,
              width=None, active=None):
    """Memory to serve a model: weights + KV cache, and the GPUs needed."""
    from .values import Record
    n = number(params)
    na = number(active) if active is not None else n
    L0, d0, h0 = _shape(na)
    L = int(layers) if layers else L0
    d = int(width) if width else d0
    kvh = int(kv_heads) if kv_heads else max(1, h0 // 8)          # grouped-query attention, 8 queries per KV head
    head_dim = d // h0
    wbytes = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "fp4": 0.5, "int4": 0.5}[precision]
    weights = n * wbytes
    kv = 2 * L * kvh * head_dim * 2 * context * batch                    # K and V in bf16
    key = gpu.upper().replace(" ", "") if isinstance(gpu, str) else "H100"
    mem_gb = GPUS.get(key, GPUS["H100"])[0]
    total = weights + kv + 2 * 2 ** 30
    return Record(params=n, precision=precision, weights_gb=round(weights / 2 ** 30, 1),
                  kv_cache_gb=round(kv / 2 ** 30, 2), total_gb=round(total / 2 ** 30, 1),
                  gpus=max(1, math.ceil(total / (0.9 * mem_gb * 2 ** 30))), gpu=key)


def gpu_table():
    from .values import Record
    return [Record(name=k, memory_gb=v[0], bf16_tflops=v[1], fp8_tflops=v[2], gpus_per_node=v[3])
            for k, v in GPUS.items()]


def module():
    return {"plan": plan, "inference": inference, "gpu_table": gpu_table, "number": number}


# ----------------------------------------------------------------------------- apsi plan
def cmd_plan(args):
    try:
        p = plan(args.params, tokens=args.tokens, gpu=args.gpu, gpus=args.gpus, days=args.days, active=args.active,
                 layers=args.layers, width=args.width, seq=args.seq, precision=args.precision,
                 optimizer=args.optimizer, mfu=args.mfu)
        inf = inference(args.params, gpu=args.gpu, precision=args.serve_precision, active=args.active)
    except ValueError as e:
        print(f"apsi plan: {e}")
        return 2
    f = p.fields
    print(f"Training plan: {f['summary']}\n")
    rows = [("Model", f"{_fmt(f['params'])} parameters" + (f", {_fmt(f['active'])} active per token (MoE)"
                                                          if f['active'] != f['params'] else "")),
            ("Shape", f"{f['layers']} layers, width {f['width']}, {f['heads']} heads, sequence {f['seq']}"),
            ("Data", f"{_fmt(f['tokens'])} tokens (~{f['dataset_gb']:,.0f} GB as 32-bit token ids), "
                     f"{f['steps']:,} steps of {_fmt(f['batch_tokens'])} tokens"),
            ("Compute", f"{f['flops']:.3g} FLOPs, {f['precision']}, MFU {args.mfu:.0%}"),
            ("Hardware", f"{f['gpus']:,} x {f['gpu']} ({f['gpu_memory_gb']} GB) in {f['nodes']:,} node(s); "
                         f"at least {f['min_gpus']:,} GPUs just to fit"),
            ("Layout", f"{f['layout']} (tensor x pipeline x data parallel), ZeRO-3/FSDP sharding, "
                       f"{f['recompute']} activation recomputation"),
            ("Memory/GPU", f"{f['memory_per_gpu_gb']} GB (model states in total: {f['model_states_gb']:,} GB)"),
            ("Duration", f"{f['days']} days, {f['gpu_hours']:,} GPU-hours"),
            ("Serving", f"{inf.fields['weights_gb']:,} GB weights in {args.serve_precision} + "
                        f"{inf.fields['kv_cache_gb']} GB KV cache (8K context) -> {inf.fields['gpus']} x "
                        f"{inf.fields['gpu']}")]
    w = max(len(r[0]) for r in rows)
    for k, v in rows:
        print(f"  {k:<{w}}  {v}")
    for n in f["notes"]:
        print(f"  {'Note':<{w}}  {n}")
    print("\nThe numbers follow standard accounting (6·N·D FLOPs, 16 bytes/parameter for AdamW, Korthikanti et al."
          "\nactivation memory); see `apsi doc scale`. Real runs differ by the framework and network used.")
    return 0


def add_parser(sub):
    p = sub.add_parser("plan", help="plan training of a large model: GPUs, layout, memory per GPU, days")
    p.add_argument("params", help="parameters, e.g. 21B, 500B, 1T, 2T")
    p.add_argument("--tokens", help="training tokens (default: 20 per active parameter)")
    p.add_argument("--active", help="active parameters per token for mixture-of-experts models, e.g. 32B")
    p.add_argument("--gpu", default="H100", help=f"GPU type: {', '.join(GPUS)} (default H100)")
    p.add_argument("--gpus", type=int, help="number of GPUs you have")
    p.add_argument("--days", type=float, help="target duration in days (the GPU count follows)")
    p.add_argument("--layers", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--precision", choices=["bf16", "fp8"], default="bf16")
    p.add_argument("--optimizer", choices=["adamw", "adam8bit", "sgd"], default="adamw")
    p.add_argument("--mfu", type=float, default=0.40, help="model FLOPs utilisation (default 0.40)")
    p.add_argument("--serve-precision", default="fp8", choices=["bf16", "fp8", "int8", "fp4", "int4"])
    p.set_defaults(fn=cmd_plan)
