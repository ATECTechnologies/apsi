"""`use tokens` — send fewer tokens to a language model without losing a single byte of information.

Every transform here is **reversible**: `unpack(pack(x)) == x`, `expand(dedupe(t)) == t`, `patch(old, diff(old, new))
== new`. Nothing is summarised, dropped or approximated, so nothing is lost. What shrinks is redundancy that models
pay for but do not need: repeated JSON keys, quoting and spacing, repeated lines, and resending a whole document when
only a few lines changed.

    use tokens
    let rows = load_json("orders.json")          # [{"id": 1, "customer": "ann", "total": 12.5}, …]
    let text = tokens.pack(rows)                 # a table: keys once, one compact line per record
    print(tokens.savings(json_str(rows, indent=2), text))   # tokens before → after, and the % saved
    assert tokens.unpack(text) == rows           # exact

    let short = tokens.dedupe(log_text)          # repeated lines written once, referenced after
    let delta = tokens.diff(old_doc, new_doc)    # only what changed (send it with the previous version cached)
    print(tokens.count(text))                    # tiktoken when installed, otherwise a close estimate

Pick what fits your data — they combine (`dedupe(pack(x))`), and `best(x)` tries them and keeps the smallest.

When the context is still too long for the budget, `focus(context, question, budget)` keeps the parts the question
needs (chosen on GLA's code and text graphs) and marks what it left out, so any part can be put back with
`unfold()`; `fit_chat(messages, budget)` does the same for a conversation (see token_focus.py).
"""
import difflib
import json
import re

_ENC = {}


# ------------------------------------------------------------------ counting
_EST = re.compile(r"[A-Za-z]{1,12}|\d{1,3}|[^\x00-\x7f]|[^\sA-Za-z\d]|\s+")


def count(text, model="gpt-4o"):
    """Tokens in text: exact with tiktoken installed (o200k / cl100k), otherwise an estimate (±10–15 %)."""
    text = text if isinstance(text, str) else _dumps(text)
    enc = _encoder(model)
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    n = 0
    for m in _EST.finditer(text):
        s = m.group()
        if s.isspace():
            n += 0 if s == " " else max(1, len(s) // 8)       # a single space joins the next word's token
        elif not s.isascii():
            n += 1 if len(s.encode()) <= 2 else 2
        else:
            n += 1
    return n


def _encoder(model):
    if model in _ENC:
        return _ENC[model]
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")
    except Exception:                       # noqa: BLE001 - not installed: estimate
        enc = None
    _ENC[model] = enc
    return enc


def exact():
    """True when count() is exact (tiktoken installed)."""
    return _encoder("gpt-4o") is not None


class Savings:
    def __init__(self, before, after):
        self.before, self.after = before, after
        self.saved = before - after
        self.percent = 100.0 * self.saved / before if before else 0.0
        self.ratio = before / after if after else float("inf")

    def __repr__(self):
        how = "" if exact() else " (estimated)"
        return f"{self.before:,} → {self.after:,} tokens{how}: {self.percent:.1f} % saved ({self.ratio:.2f}× fewer)"

    __str__ = __repr__


def savings(before, after, model="gpt-4o"):
    return Savings(count(before, model), count(after, model))


# ------------------------------------------------------------------ JSON as tables
def _dumps(v):
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def _uniform(v):
    if not isinstance(v, list) or len(v) < 2 or not all(isinstance(r, dict) for r in v):
        return None
    keys = list(v[0].keys())
    if not keys or any(list(r.keys()) != keys for r in v):
        return None
    return keys


def pack(value):
    """Compact, exactly reversible text for a JSON value. A list of records with the same keys becomes a table:
    `@table ["id","name"]` then one line per record (`1,"ann"`). Anything else is minified JSON."""
    keys = _uniform(value)
    if keys is None:
        return _dumps(value)
    lines = ["@table " + _dumps(keys)]
    for r in value:
        lines.append(_dumps([r[k] for k in keys])[1:-1])
    return "\n".join(lines)


def unpack(text):
    if text.startswith("@table "):
        head, _, body = text.partition("\n")
        keys = json.loads(head[7:])
        out = []
        for line in body.split("\n") if body else []:
            vals = json.loads("[" + line + "]")
            out.append(dict(zip(keys, vals)))
        return out
    return json.loads(text)


def pack_jsonl(text):
    """JSON Lines → pack() of the records (exact for the parsed values)."""
    return pack([json.loads(line) for line in text.splitlines() if line.strip()])


# ------------------------------------------------------------------ repeated lines
def dedupe(text, min_chars=24, min_count=2):
    """Lines that repeat are written once in a header and referenced as §N where they occur."""
    text = text.replace("§", "§§")                         # escape: § in the text never looks like a reference
    lines = text.split("\n")
    freq = {}
    for ln in lines:
        if len(ln) >= min_chars:
            freq[ln] = freq.get(ln, 0) + 1
    reps = [ln for ln, c in freq.items() if c >= min_count]
    if not reps:
        return "@plain\n" + text
    ids = {ln: i + 1 for i, ln in enumerate(reps)}
    head = [f"@refs {len(reps)}"] + reps
    body = [f"§{ids[ln]}" if ln in ids else ln for ln in lines]
    return "\n".join(head + body)


def expand(text):
    if text.startswith("@plain\n"):
        return text[7:].replace("§§", "§")
    head, _, rest = text.partition("\n")
    n = int(head.split()[1])
    parts = rest.split("\n")
    reps, body = parts[:n], parts[n:]
    out = []
    for ln in body:
        if ln.startswith("§") and not ln.startswith("§§") and ln[1:].isdigit():
            out.append(reps[int(ln[1:]) - 1])
        else:
            out.append(ln)
    return "\n".join(out).replace("§§", "§")


# ------------------------------------------------------------------ deltas
def diff(old, new):
    """Only what changed, as line operations: `=N` keep N lines, `-N` drop N lines, `+text` add a line."""
    a, b = old.split("\n"), new.split("\n")
    ops = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            ops.append(f"={i2 - i1}")
            continue
        if i2 > i1:
            ops.append(f"-{i2 - i1}")
        ops.extend("+" + line for line in b[j1:j2])
    return "\n".join(ops)


def patch(old, delta):
    a = old.split("\n")
    out, i = [], 0
    for op in delta.split("\n") if delta else []:
        if op.startswith("="):
            n = int(op[1:])
            out.extend(a[i:i + n])
            i += n
        elif op.startswith("-"):
            i += int(op[1:])
        elif op.startswith("+"):
            out.append(op[1:])
    return "\n".join(out)


# ------------------------------------------------------------------ choose
def best(value, model="gpt-4o"):
    """Try the transforms that apply and return (text, method) with the fewest tokens; unpack with restore()."""
    cands = []
    if isinstance(value, str):
        cands.append(("plain", value))
        cands.append(("dedupe", dedupe(value)))
    else:
        p = pack(value)
        cands.append(("pack", p))
        cands.append(("pack+dedupe", dedupe(p)))
    method, text = min(cands, key=lambda c: count(c[1], model))
    return text, method


def restore(text, method):
    if method == "plain":
        return text
    if method == "dedupe":
        return expand(text)
    if method == "pack":
        return unpack(text)
    if method == "pack+dedupe":
        return unpack(expand(text))
    raise ValueError(f"unknown method '{method}'")


def module(interp=None):
    from . import token_focus as F
    return {"count": count, "exact": exact, "savings": savings, "pack": pack, "unpack": unpack,
            "pack_jsonl": pack_jsonl, "dedupe": dedupe, "expand": expand, "diff": diff, "patch": patch,
            "best": best, "restore": restore, "focus": F.focus, "unfold": F.unfold, "fit_chat": F.fit_chat,
            "outline": F.outline}
