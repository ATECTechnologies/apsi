"""`apsi cache`: what Apsi keeps on disk to start fast, how big it is, and cleaning it safely.

    apsi cache                          every cache, its size and what is stale
    apsi cache models                   downloaded models: size, versions, when last used
    apsi cache clean                    only what can never be used again (old kernel builds, half downloads,
                                        int8 copies of models that are gone, compiled programs not used in 30 days)
    apsi cache clean models --model Qwen/Qwen2.5-3B-Instruct     one model
    apsi cache clean all --dry-run      what `all` would remove

Nothing in a cache is needed for correctness: anything removed is rebuilt (or downloaded again) when needed.
"""
import os
import shutil
import time

KINDS = {
    "models": "downloaded models and datasets (Hub)",
    "gpu": "GPU kernels built for this Apsi version",
    "int8": "int8 copies of models for the processor",
    "native": "machine code of @native functions",
    "compiled": "compiled Apsi programs",
}


def _home():
    from .security import config_dir
    return config_dir()


def roots():
    c = os.path.join(_home(), "cache")
    return {"models": os.environ.get("APSI_HUB_CACHE") or os.path.join(os.path.expanduser("~"), ".apsi", "hub"),
            "gpu": os.path.join(c, "gpu"), "int8": os.path.join(c, "cpu-int8"),
            "native": os.path.join(c, "native"), "compiled": os.path.join(c, "compiled")}


def size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return total


def last_used(path):
    t = 0.0
    for r, _, fs in os.walk(path) if os.path.isdir(path) else [(os.path.dirname(path), [], [os.path.basename(path)])]:
        for f in fs:
            try:
                st = os.stat(os.path.join(r, f))
                t = max(t, st.st_atime, st.st_mtime)
            except OSError:
                pass
    return t


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024


def models():
    """[(repo, kind, folder, bytes, versions, last used)] of the Hub cache."""
    out = []
    base = roots()["models"]
    for kind in ("models", "datasets"):
        d = os.path.join(base, kind)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            f = os.path.join(d, name)
            if os.path.isdir(f):
                vers = [v for v in os.listdir(f) if os.path.isdir(os.path.join(f, v))]
                out.append((name.replace("--", "/"), kind[:-1], f, size(f), len(vers), last_used(f)))
    return out


def stale():
    """[(path, why)] that no future run can use."""
    out = []
    r = roots()
    from . import hub_gpu
    cur = hub_gpu._cache_path()
    keep = {os.path.basename(cur), os.path.basename(cur).replace("hub-kernels-", "hub-tc-")} if cur else set()
    if os.path.isdir(r["gpu"]):
        for f in os.listdir(r["gpu"]):
            if (f.startswith("hub-kernels-") or f.startswith("hub-tc-")) and f not in keep:
                out.append((os.path.join(r["gpu"], f), "kernels of another Apsi version"))
            elif f.endswith(".tmp"):
                out.append((os.path.join(r["gpu"], f), "unfinished write"))
    if os.path.isdir(r["models"]):
        for root, _, fs in os.walk(r["models"]):
            for f in fs:
                p = os.path.join(root, f)
                if (f.endswith(".part") or f.endswith(".tmp")) and time.time() - os.path.getmtime(p) > 7 * 86400:
                    out.append((p, "a download stopped more than a week ago"))
    if os.path.isdir(r["int8"]):
        for d in os.listdir(r["int8"]):
            p = os.path.join(r["int8"], d)
            if not os.path.exists(os.path.join(p, "complete")) and time.time() - os.path.getmtime(p) > 3600:
                out.append((p, "an int8 copy that was never finished"))
            elif time.time() - last_used(p) > 60 * 86400:
                out.append((p, "an int8 copy not used in 60 days"))
    if os.path.isdir(r["compiled"]):
        for root, _, fs in os.walk(r["compiled"]):
            for f in fs:
                p = os.path.join(root, f)
                try:
                    if time.time() - max(os.path.getatime(p), os.path.getmtime(p)) > 30 * 86400:
                        out.append((p, "a compiled program not used in 30 days"))
                except OSError:
                    pass
    return out


def _remove(p):
    if os.path.isdir(p) and not os.path.islink(p):
        shutil.rmtree(p, ignore_errors=True)
    else:
        try:
            os.remove(p)
        except OSError:
            pass


def cmd(args):
    action = (args.what or ["show"])[0]
    rest = (args.what or [])[1:]
    r = roots()
    if action == "show":
        print("Apsi caches (all safe to remove; rebuilt when needed)")
        total = 0
        for k, desc in KINDS.items():
            n = size(r[k]) if os.path.exists(r[k]) else 0
            total += n
            print(f"  {k:9} {human(n):>10}  {desc}\n  {'':9} {'':>10}  {r[k]}")
        st = stale()
        sn = sum(size(p) for p, _ in st)
        print(f"  {'total':9} {human(total):>10}")
        if st:
            print(f"\n{len(st)} stale item(s), {human(sn)}: `apsi cache clean` removes them")
        return 0
    if action == "models":
        ms = models()
        if not ms:
            print("no models downloaded")
            return 0
        for repo, kind, f, n, vers, used in sorted(ms, key=lambda m: -m[3]):
            when = time.strftime("%Y-%m-%d", time.localtime(used)) if used else "-"
            print(f"  {human(n):>10}  {repo}  ({kind}, {vers} version{'s' if vers != 1 else ''}, last used {when})")
        print(f"  {human(sum(m[3] for m in ms)):>10}  in {r['models']}")
        return 0
    if action == "path":
        print(_home())
        return 0
    if action == "clean":
        what = rest[0] if rest else "stale"
        targets = []
        if what == "stale":
            targets = stale()
        elif what == "all":
            targets = [(r[k], KINDS[k]) for k in KINDS if os.path.exists(r[k])]
        elif what in KINDS:
            if what == "models" and args.model:
                folder = None
                for repo, kind, f, *_ in models():
                    if repo == args.model:
                        folder = f
                if folder is None:
                    print(f"apsi cache: '{args.model}' is not downloaded (apsi cache models lists them)")
                    return 1
                targets = [(folder, f"model {args.model}")]
            else:
                targets = [(r[what], KINDS[what])] if os.path.exists(r[what]) else []
        else:
            print(f"apsi cache clean: one of stale, all, {', '.join(KINDS)}")
            return 2
        if args.older:
            limit = time.time() - float(args.older) * 86400
            targets = [(p, w) for p, w in targets if last_used(p) < limit]
        n = sum(size(p) for p, _ in targets)
        for p, why in targets:
            print(f"  {'would remove' if args.dry_run else 'removed'} {p}  ({why})")
            if not args.dry_run:
                _remove(p)
        print(f"{'would free' if args.dry_run else 'freed'} {human(n)}" if targets else "nothing to remove")
        return 0
    print("apsi cache: show, models, path or clean")
    return 2


def add_parser(sub):
    p = sub.add_parser("cache", help="show and clean Apsi's caches (models, GPU kernels, compiled code)")
    p.add_argument("what", nargs="*", help="show | models | path | clean [stale|all|models|gpu|int8|native|compiled]")
    p.add_argument("--model", help="with `clean models`: one model, e.g. Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--older", help="with clean: only what was not used in this many days")
    p.add_argument("--dry-run", action="store_true", help="with clean: show what would be removed")
    p.set_defaults(fn=cmd)
