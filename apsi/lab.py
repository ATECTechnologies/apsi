"""`use lab` — reproducible research: experiment logs, provenance, seeds, result formatting and tables.

  set_seed(seed)                        one seed for Apsi, numpy and Python's random module
  provenance()                          Apsi/Python/numpy versions, OS, CPU, GPUs, time (UTC), seed
  experiment(name, params, folder)      an experiment log: log(step, metrics), note(text), result(metrics),
                                        finish() -> one JSON-lines file per run with provenance (needs --allow-write)
  load_runs(folder, name)               read experiment logs back as records (needs --allow-read)
  sigfig(x, n), format_measurement(value, uncertainty, unit)   "9.81(3) m/s^2" style (uncertainty in 1–2 digits)
  table_markdown(rows, headers), table_latex(rows, headers, caption, label), table_csv(rows, headers)
"""
import datetime
import json
import math
import os
import platform
import random
import time
import uuid

import numpy as np

from .errors import RuntimeErr
from .units import Quantity
from .values import Record

_SEED = [None]


def set_seed(seed):
    """Seed every random generator Apsi uses (Apsi's random module, numpy's legacy and Python's random)."""
    s = int(seed)
    from . import ops
    ops.seed(s)
    np.random.seed(s % (2 ** 32))
    random.seed(s)
    _SEED[0] = s
    return s


def provenance():
    """What produced a result: software versions, machine and time (no user names, host names or paths)."""
    from . import __version__
    try:
        from .hardware import _cpu_name, _gpus
        cpu = _cpu_name()
        gpus = [g.get("name", "GPU") if isinstance(g, dict) else str(g) for g in _gpus()]
    except Exception:                                    # pragma: no cover - hardware probing is best-effort
        cpu, gpus = None, []
    return Record(apsi=__version__, python=platform.python_version(), numpy=np.__version__,
                  os=f"{platform.system()} {platform.release()}", machine=platform.machine(),
                  cpu=cpu if cpu is not None else platform.processor() or platform.machine(),
                  cores=os.cpu_count(), gpus=gpus, seed=_SEED[0],
                  time_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))


def _plain(v):
    if isinstance(v, Record):
        return {k: _plain(x) for k, x in v.fields.items()}
    if isinstance(v, Quantity):
        from .units import unit_string
        return {"value": _plain(v.v), "unit": unit_string(v.dim)}
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 10000 else {"shape": list(v.shape), "mean": float(np.mean(v))}
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def sigfig(x, n=3):
    """Round to n significant figures."""
    v = float(x.v if isinstance(x, Quantity) else x)
    if v == 0 or not math.isfinite(v):
        return v
    return round(v, -int(math.floor(math.log10(abs(v)))) + int(n) - 1)


def format_measurement(value, uncertainty, unit=None):
    """Value with its standard uncertainty in parenthesis notation, rounded by the usual rule (uncertainty to two
    significant digits when they start with 1 or 2, otherwise one): 9.8123 ± 0.0312 -> "9.81(3)"."""
    v = float(value.v if isinstance(value, Quantity) else value)
    u = abs(float(uncertainty.v if isinstance(uncertainty, Quantity) else uncertainty))
    if u == 0 or not math.isfinite(u):
        s = f"{v:g}"
    else:
        e = int(math.floor(math.log10(u)))
        lead = int(u / 10 ** e)
        digits = 2 if lead < 3 else 1
        dec = -(e - digits + 1)
        ur = round(u, dec)
        if ur >= 10 ** (e + 1):                          # rounding moved to the next decade
            dec -= 1
        if dec > 0:
            s = f"{v:.{dec}f}({int(round(u * 10 ** dec))})"
        else:
            q = 10 ** (-dec)
            s = f"{int(round(v / q) * q)}({int(round(u / q) * q)})"
    return f"{s} {unit}" if unit else s


def _rows(rows, headers):
    if isinstance(rows, Record):
        rows = [rows]
    if rows and isinstance(rows[0], Record):
        headers = headers or list(rows[0].fields)
        rows = [[r.fields.get(h, "") for h in headers] for r in rows]
    if headers is None:
        raise RuntimeErr("give headers, or rows as records")
    return [list(map(str, headers))], [[_cell(c) for c in r] for r in rows]


def _cell(c):
    if isinstance(c, float):
        return f"{c:.6g}"
    if isinstance(c, Quantity):
        return str(c)
    return str(c)


def table_markdown(rows, headers=None):
    """A Markdown table from lists or records."""
    (h,), body = _rows(rows, headers)
    out = ["| " + " | ".join(h) + " |", "|" + "|".join("---" for _ in h) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def _tex(s):
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    return s


def table_latex(rows, headers=None, caption=None, label=None):
    """A LaTeX table (booktabs) from lists or records; text is escaped."""
    (h,), body = _rows(rows, headers)
    out = [r"\begin{table}[ht]", r"\centering"]
    if caption:
        out.append(r"\caption{" + _tex(str(caption)) + "}")
    if label:
        out.append(r"\label{" + str(label) + "}")
    out += [r"\begin{tabular}{" + "l" * len(h) + "}", r"\toprule", " & ".join(_tex(x) for x in h) + r" \\",
            r"\midrule"]
    out += [" & ".join(_tex(x) for x in r) + r" \\" for r in body]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def table_csv(rows, headers=None):
    """CSV text (RFC 4180 quoting)."""
    (h,), body = _rows(rows, headers)

    def q(s):
        return '"' + s.replace('"', '""') + '"' if any(c in s for c in ',"\n') else s
    return "\n".join(",".join(q(x) for x in r) for r in [h] + body) + "\n"


def install(interp):
    from .security import check_path

    def experiment(name, params=None, folder="experiments"):
        """Start an experiment log. Returns a record with log(step, metrics), note(text), result(metrics),
        finish() and the run id. Every call appends one JSON line to FOLDER/NAME.jsonl."""
        nm = str(name)
        if not nm or any(c in nm for c in "/\\:") or nm.startswith("."):
            raise RuntimeErr("experiment name must be a plain file name")
        path = check_path(interp, os.path.join(str(folder), nm + ".jsonl"), "write", "writing experiment logs")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        run = uuid.uuid4().hex[:12]
        t0 = time.time()
        state = {"steps": 0, "done": False}

        def write(kind, **data):
            if state["done"]:
                raise RuntimeErr("this experiment run is finished")
            rec = {"run": run, "kind": kind, "t": round(time.time() - t0, 6)}
            rec.update({k: _plain(v) for k, v in data.items()})
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        write("start", name=nm, params=_plain(params) if params is not None else {}, provenance=_plain(provenance()))

        def log(step=None, metrics=None, **kw):
            m = dict(metrics.fields) if isinstance(metrics, Record) else dict(metrics or {})
            m.update(kw)
            state["steps"] += 1
            write("log", step=step if step is not None else state["steps"], metrics=m)

        def note(text):
            write("note", text=str(text))

        def result(metrics=None, **kw):
            m = dict(metrics.fields) if isinstance(metrics, Record) else dict(metrics or {})
            m.update(kw)
            write("result", metrics=m)

        def finish():
            write("finish", seconds=round(time.time() - t0, 3), logs=state["steps"])
            state["done"] = True
            return path

        return Record(run=run, path=path, log=log, note=note, result=result, finish=finish)

    def load_runs(folder="experiments", name=None):
        """All runs in an experiment folder (or one experiment): list of record(run, name, params, logs, result,
        notes, provenance, finished)."""
        base = check_path(interp, str(folder), "read", "reading experiment logs")
        files = [name + ".jsonl"] if name else sorted(f for f in os.listdir(base) if f.endswith(".jsonl"))
        runs = {}
        for f in files:
            p = os.path.join(base, f)
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    d = runs.setdefault(r.get("run"), {"run": r.get("run"), "name": f[:-6], "params": {}, "logs": [],
                                                      "result": None, "notes": [], "provenance": None,
                                                      "finished": False})
                    k = r.get("kind")
                    if k == "start":
                        d["params"], d["provenance"] = r.get("params", {}), r.get("provenance")
                    elif k == "log":
                        d["logs"].append(dict(step=r.get("step"), **(r.get("metrics") or {})))
                    elif k == "note":
                        d["notes"].append(r.get("text"))
                    elif k == "result":
                        d["result"] = r.get("metrics")
                    elif k == "finish":
                        d["finished"] = True

        def rec(v):
            if isinstance(v, dict):
                return Record(**{str(k): rec(x) for k, x in v.items()})
            if isinstance(v, list):
                return [rec(x) for x in v]
            return v
        return [rec(d) for d in runs.values()]

    return {"experiment": experiment, "load_runs": load_runs}


def module():
    return {k: v for k, v in globals().items()
            if callable(v) and not k.startswith("_") and getattr(v, "__module__", "") == __name__
            and k not in ("module", "install")}
