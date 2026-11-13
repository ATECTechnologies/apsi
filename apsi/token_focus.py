"""Send a model only the part of a long context that matters for the question, built on GLA's graphs.

`use tokens` (tokens.py) is lossless: it removes redundancy. This module goes further when a context is too long for
the budget: it keeps what the question needs and replaces the rest with short, numbered markers that say exactly what
was left out (which lines, which functions), so nothing is lost silently and any part can be put back.

How it chooses (the same structure GLA reads):
  * the context is cut into units: for Python, every top-level statement, function, class and method (from the
    abstract syntax tree); for other code and for text, blocks between blank lines, cut into sentences if long;
  * the units are joined into a graph: for Python, GLA's code property graph (calls, data flow from definitions to
    uses, the system dependence graph), so a function brings the functions it calls and the names it reads; for
    text, neighbouring units and units that share rare words;
  * each unit is scored by the question's words it contains (rarer words count more; snake_case and camelCase are
    split into their pieces), then the scores spread over the graph (a personalised PageRank), so what the relevant
    code depends on scores too;
  * units are taken by score until the budget (in the model's own tokens) is full; a function that does not fit
    whole may still appear as its outline (signature, first docstring line, the names it calls);
  * the kept units are written in their original order; each run of left-out units becomes one marker
    `[[omitted 3: lines 40-88: def parse, def lex]]` and `unfold()` restores any of them exactly.

    use tokens
    let f = tokens.focus(source_code, "why does parse() drop the last token?", budget=1500)
    print(f.text)                      # what to send
    print(f.saved)                     # tokens before -> after
    let more = tokens.unfold(f, [2])   # the same text with omitted block 2 written out again

    let msgs = tokens.fit_chat(history, budget=6000)   # system + recent turns verbatim, older turns focused

Everything is plain text in and out, so it works with any model: Apsi's own (`use hub`), a model behind an API, or a
model you trained.
"""
import ast
import math
import re
from collections import Counter, defaultdict

from . import tokens as T

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_PIECES = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_STOP = frozenset("""a an the and or but if then else of to in on at by for with from as is are was were be been being
it its this that these those there here what which who whom whose why how when where do does did done not no yes can
could should would will shall may might must i you he she we they me him her us them my your our their mine yours
self cls def class return import pass none true false print let fn var""".split())


def _terms(text):
    """Lower-case words and their snake_case / camelCase pieces (stop words dropped)."""
    out = []
    for w in _WORD.findall(text):
        lw = w.lower()
        if lw not in _STOP and len(lw) > 1:
            out.append(lw)
        parts = [p.lower() for p in _PIECES.findall(w)]
        if len(parts) > 1:
            out.extend(p for p in parts if p not in _STOP and len(p) > 1)
    return out


class Unit:
    __slots__ = ("i", "start", "end", "text", "name", "outline", "terms", "kind")

    def __init__(self, i, start, end, text, name="", outline=None, kind="text"):
        self.i, self.start, self.end, self.text, self.name = i, start, end, text, name
        self.outline, self.kind = outline, kind
        self.terms = Counter(_terms(text + " " + name))


# ------------------------------------------------------------------ cutting into units
def _python_units(src, big=60):
    """Units that cover every line once: top-level statements, functions, classes and methods; a function or block
    longer than `big` lines is cut further into its header and the statements of its body (recursively), so a long
    function can keep only the part that matters."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    cuts = []                                             # (first line, name, outline)

    def outline(node):
        head = lines[node.lineno - 1].rstrip("\n")
        k = node.lineno
        while not head.rstrip().endswith(":") and k < node.end_lineno:   # a signature over several lines
            head += "\n" + lines[k].rstrip("\n")
            k += 1
        ind = re.match(r"\s*", lines[node.lineno - 1]).group() + "    "
        doc = ast.get_docstring(node)
        calls = sorted({n.func.id if isinstance(n.func, ast.Name) else n.func.attr for n in ast.walk(node)
                        if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))})
        body = []
        if doc:
            body.append(f'{ind}"""{doc.strip().splitlines()[0]}"""')
        if calls:
            body.append(f"{ind}# calls: {', '.join(calls[:12])}{' …' if len(calls) > 12 else ''}")
        body.append(f"{ind}...")
        return head + "\n" + "\n".join(body) + "\n"

    def first(node):
        return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])

    def walk(node, owner):
        name, out = "", None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name, out = (f"def {owner}.{node.name}" if owner else f"def {node.name}"), outline(node)
        elif isinstance(node, ast.ClassDef):
            name = f"class {node.name}"
        cuts.append((first(node), name, out))
        body = getattr(node, "body", None)
        if node.end_lineno - first(node) + 1 > big and isinstance(body, list) and body:
            sub_owner = node.name if isinstance(node, ast.ClassDef) else owner
            inner = name or owner
            kids = list(body) + list(getattr(node, "orelse", []) or []) + list(getattr(node, "finalbody", []) or [])
            for h in getattr(node, "handlers", []) or []:
                kids.append(h)
            kids.sort(key=first)
            for kid in kids:
                if first(kid) <= node.lineno:
                    continue
                if isinstance(kid, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    walk(kid, sub_owner)
                else:
                    before = len(cuts)
                    walk(kid, sub_owner)
                    if len(cuts) > before:            # a piece of a long function or block: named after it
                        f0, nm, o = cuts[before]
                        cuts[before] = (f0, nm or (f"{inner} (part)" if inner else ""), o)
    for node in tree.body:
        walk(node, "")
    cuts.sort(key=lambda c: c[0])
    dedup = []
    for c in cuts:
        if dedup and dedup[-1][0] == c[0]:
            continue
        dedup.append(c)
    if not dedup or dedup[0][0] > 1:
        dedup.insert(0, (1, "", None))
    units = []
    for k, (a, name, out) in enumerate(dedup):
        z = dedup[k + 1][0] - 1 if k + 1 < len(dedup) else len(lines)
        if z < a:
            continue
        units.append(Unit(len(units), a, z, "".join(lines[a - 1:z]), name, out, kind="code"))
    return _merge_small(units, lines)


def _merge_small(units, lines):
    """Neighbouring plain statements (imports, constants, a few lines of a long function) as one unit: fewer markers,
    the same lines."""
    out = []
    for u in units:
        p = out[-1] if out else None
        if p is not None and u.start == p.end + 1 and (u.end - p.start) < 24 and \
                ((not u.name and not p.name) or (u.name.endswith("(part)") and u.name == p.name)):
            out[-1] = Unit(p.i, p.start, u.end, "".join(lines[p.start - 1:u.end]), p.name, p.outline, kind=p.kind)
        else:
            out.append(u)
    for i, u in enumerate(out):
        u.i = i
    return out


def _text_units(text, max_lines=40, long_par=600):
    """Units that cover the text exactly once: blocks between blank lines (a heading starts a new block; a block
    of more than `max_lines` lines is cut), a long paragraph cut into sentences."""
    lines = text.splitlines(keepends=True)
    starts = [0]                                          # line indices where a unit starts
    run = 0
    for k, ln in enumerate(lines):
        if k == 0:
            continue
        prev_blank = not lines[k - 1].strip()
        heading = ln.lstrip().startswith("#")
        if (prev_blank and ln.strip()) or (heading and k != starts[-1]) or run >= max_lines:
            starts.append(k)
            run = 0
        run += 1
    units = []
    for a_i, a in enumerate(starts):
        z = starts[a_i + 1] if a_i + 1 < len(starts) else len(lines)
        body = "".join(lines[a:z])
        if not body:
            continue
        if len(body) > long_par and z - a <= 3:           # one long paragraph: its sentences
            pos = 0
            for m in _SENT.finditer(body):
                units.append(Unit(len(units), a + 1, z, body[pos:m.end()], _title(body[pos:m.end()])))
                pos = m.end()
            units.append(Unit(len(units), a + 1, z, body[pos:], _title(body[pos:])))
        else:
            units.append(Unit(len(units), a + 1, z, body, _title(body)))
    return units


def _title(text):
    t = text.strip().splitlines()[0] if text.strip() else ""
    t = t.lstrip("#*-> ").strip()
    return (t[:48] + "…") if len(t) > 48 else t


# ------------------------------------------------------------------ the graph
def _python_edges(src, units):
    """Edges between units from GLA's code property graph: calls, data flow, dependences."""
    from .gla_graph import _Builder
    b = _Builder()
    b.visit(ast.parse(src))
    b.finish()
    starts = [u.start for u in units]

    def unit_of(line):
        lo, hi = 0, len(units) - 1
        if not line:
            return None
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= line:
                lo = mid
            else:
                hi = mid - 1
        return lo if units[lo].start <= line <= units[lo].end else None
    w = {"invokes": 1.0, "inter_proc": 1.0, "dataflow": 0.8, "data_dep": 0.8, "control_dep": 0.3, "maps_to": 0.3}
    edges = defaultdict(float)
    nodes = b.nodes
    for a, rel, z in b.edges:
        ua, uz = unit_of(nodes[a]["line"]), unit_of(nodes[z]["line"])
        if ua is None or uz is None or ua == uz:
            continue
        wt = w.get(rel, 0.0)
        if wt:
            edges[(ua, uz)] = max(edges[(ua, uz)], wt)
    # a use of a name links to the unit that defines it (classes, functions, module constants)
    defs = {}
    for u in units:
        if u.name:
            defs[u.name.split()[-1].split(".")[-1]] = u.i
    for u in units:
        for t in set(_WORD.findall(u.text)):
            d = defs.get(t)
            if d is not None and d != u.i:
                edges[(u.i, d)] = max(edges[(u.i, d)], 0.6)
    return edges


def _text_edges(units, idf):
    edges = defaultdict(float)
    for u in units[1:]:
        edges[(u.i - 1, u.i)] = 0.5
        edges[(u.i, u.i - 1)] = 0.5
    rare = defaultdict(list)                              # units that share a rare word
    for u in units:
        for t in u.terms:
            if idf.get(t, 0) > 1.5:
                rare[t].append(u.i)
    for t, us in rare.items():
        if 1 < len(us) <= 6:
            for a in us:
                for z in us:
                    if a != z:
                        edges[(a, z)] = max(edges[(a, z)], 0.4)
    return edges


def _rank(units, edges, base, alpha=0.35, steps=12):
    """Personalised PageRank: what the relevant units depend on scores too."""
    n = len(units)
    out_w = defaultdict(float)
    for (a, _), wt in edges.items():
        out_w[a] += wt
    tot = sum(base) or 1.0
    p = [b / tot for b in base]
    s = p[:]
    for _ in range(steps):
        nxt = [(1 - alpha) * p[i] for i in range(n)]
        for (a, z), wt in edges.items():
            if out_w[a]:
                nxt[z] += alpha * s[a] * wt / out_w[a]
        s = nxt
    return s


# ------------------------------------------------------------------ choosing
class Focus:
    """The result of focus(): .text (to send), .kept / .outlined / .omitted (unit numbers), .saved, .blocks
    (number -> (what .text shows, the exact original lines)), .tokens (of .text)."""

    def __init__(self, text, units, kept, outlined, blocks, before, after, model):
        self.text, self.units, self.kept, self.outlined = text, units, kept, outlined
        self.blocks, self.model = blocks, model
        self.tokens = after
        self.saved = T.Savings(before, after)
        self.omitted = [u.i for u in units if u.i not in kept and u.i not in outlined]

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"Focus({len(self.kept)} of {len(self.units)} parts whole, {len(self.outlined)} as outlines, {self.saved})"


def _is_python(text):
    try:
        ast.parse(text)
    except (SyntaxError, ValueError):
        return False
    return bool(re.search(r"^\s*(def |class |import |from \w)", text, re.M))


def focus(context, query="", budget=2000, model="gpt-4o", keep=None, kind="auto", map=True, outlines=True):
    """The parts of `context` that matter for `query`, within `budget` tokens of `model` (see the module text).

    keep: line numbers or names ("def parse") that must stay whole. kind: "auto", "python" or "text".
    map: start with a one-line table of contents of what was left out. outlines: functions that do not fit whole may
    stay as their outline. A context that already fits is returned unchanged."""
    context = str(context)
    before = T.count(context, model)
    py = _is_python(context) if kind == "auto" else kind == "python"
    units = _python_units(context) if py else _text_units(context)
    if before <= budget or not units:
        return Focus(context, units, {u.i for u in units}, set(), {}, before, before, model)
    df = Counter()
    for u in units:
        df.update(set(u.terms))
    n = len(units)
    idf = {t: math.log(1 + n / c) for t, c in df.items()}
    q = Counter(_terms(str(query)))
    base = []
    for u in units:
        size = sum(u.terms.values()) or 1
        sc = sum(idf.get(t, 0.0) * qc * (1 + math.log(u.terms[t])) for t, qc in q.items() if u.terms.get(t))
        sc = sc / (1 + 0.15 * math.log(size))                     # long units do not win by length alone
        if u.name and any(t in _terms(u.name) for t in q):
            sc += 2.0 * max(idf.values() or [1.0])                    # the question names this function
        base.append(sc + 1e-3)
    if not q:
        base = [1.0 + (2.0 if u.i == 0 else 0.0) + (0.5 if u.name else 0.0) for u in units]
    edges = _python_edges(context, units) if py else _text_edges(units, idf)
    score = _rank(units, edges, base)
    must = set()
    for k in (keep or []):
        for u in units:
            if (isinstance(k, int) and u.start <= k <= u.end) or (isinstance(k, str) and k in u.name):
                must.add(u.i)
    cost = {u.i: T.count(u.text, model) for u in units}
    ocost = {u.i: T.count(u.outline, model) for u in units if u.outline}
    marker = 14                                                       # one omission marker, roughly
    head = (24 + 2 * min(len(units), 200)) if map else 0
    room = budget - head
    kept, outlined = set(), set()
    for i in sorted(must):
        kept.add(i)
        room -= cost[i]
    order = sorted(range(n), key=lambda i: -score[i])
    for i in order:
        if i in kept:
            continue
        if cost[i] + marker <= room:
            kept.add(i)
            room -= cost[i]
        elif outlines and i in ocost and ocost[i] + marker <= room and score[i] > 0:
            outlined.add(i)
            room -= ocost[i]
    for _ in range(n + 1):                                            # the markers' real cost: drop the weakest
        text, blocks = _render(context, units, kept, outlined, py, map)
        after = T.count(text, model)
        if after <= budget:
            break
        weakest = min((i for i in kept | outlined if i not in must), key=lambda i: score[i], default=None)
        if weakest is None:
            break
        kept.discard(weakest)
        outlined.discard(weakest)
    return Focus(text, units, kept, outlined, blocks, before, after, model)


def _render(context, units, kept, outlined, py, with_map):
    """(text, blocks): blocks[k] = (what the text shows for block k, the exact original lines)."""
    out, blocks, run = [], {}, []
    comment = "# " if py else ""

    def flush():
        if not run:
            return
        k = len(blocks) + 1
        names = [units[i].name for i in run if units[i].name]
        names = list(dict.fromkeys(names))
        a, z = units[run[0]].start, units[run[-1]].end
        what = f": {', '.join(names[:6])}{' …' if len(names) > 6 else ''}" if names else ""
        shown = f"{comment}[[omitted {k}: lines {a}-{z}{what}]]\n"
        out.append(shown)
        blocks[k] = (shown, "".join(units[i].text for i in run))
        run.clear()
    for u in units:
        if u.i in kept:
            flush()
            out.append(u.text)
        elif u.i in outlined:
            flush()
            k = len(blocks) + 1
            shown = f"{comment}[[outline {k}: lines {u.start}-{u.end}]]\n{u.outline}"
            out.append(shown)
            blocks[k] = (shown, u.text)
        else:
            run.append(u.i)
    flush()
    body = "".join(out)
    if with_map and blocks:
        left = ", ".join(str(k) for k in blocks)
        body = f"{comment}[[map: {len(kept)} parts whole, {len(outlined)} outlined; shortened parts {left} " \
               f"can be asked for by number]]\n" + body
    return body, blocks


def unfold(f, which=None):
    """The focused text with shortened parts written out again (all of them when which is None): unfold(f) is the
    original context exactly (but for the map line)."""
    text = f.text
    if text.startswith(("[[map:", "# [[map:")):
        text = text.split("\n", 1)[1]
    keys = list(f.blocks) if which is None else [int(k) for k in which]
    for k in keys:
        if k in f.blocks:
            shown, original = f.blocks[k]
            text = text.replace(shown, original, 1)
    return text


def fit_chat(messages, budget=8000, model="gpt-4o", keep_last=4):
    """A chat history in `budget` tokens: the system messages and the last `keep_last` messages verbatim, older
    messages focused on the latest user message (dropped whole, oldest first, only if focusing is not enough)."""
    msgs = [dict(m) for m in messages]
    if not msgs:
        return msgs

    def total(ms):
        return sum(T.count(str(m.get("content", "")), model) + 4 for m in ms)
    if total(msgs) <= budget:
        return msgs
    query = next((str(m.get("content", "")) for m in reversed(msgs) if m.get("role") == "user"), "")
    fixed = {i for i, m in enumerate(msgs) if m.get("role") == "system"} | \
        set(range(max(0, len(msgs) - keep_last), len(msgs)))
    older = [i for i in range(len(msgs)) if i not in fixed]
    room = budget - total([msgs[i] for i in fixed])
    if older and room > 0:
        sizes = {i: T.count(str(msgs[i].get("content", "")), model) for i in older}
        whole = sum(sizes.values()) or 1
        for i in older:
            share = max(32, int(room * sizes[i] / whole) - 4)
            f = focus(str(msgs[i].get("content", "")), query, budget=share, model=model, map=False)
            msgs[i]["content"] = f.text
    while total(msgs) > budget and older:                            # still too long: the oldest go first
        i = older.pop(0)
        msgs[i] = None
        msgs = [m for m in msgs if m is not None]
        older = [j - 1 for j in older]
        fixed = {j - 1 if j > i else j for j in fixed}
    return msgs


def outline(source):
    """Python source as its outline: every function's signature, first docstring line and the names it calls."""
    units = _python_units(str(source))
    return "".join(u.outline if u.outline else u.text for u in units)
