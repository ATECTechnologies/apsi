"""Code and text -> GLA graphs (GLA.py's _PythonCPGBuilder and GraphTensorizer, for gla_batch).

  code_graph(source)          Python source -> a code property graph: AST (parent -> child, preorder next), CFG (flow,
                              branches into blocks), scope- and order-aware data flow (def -> use, PDG data and control
                              dependences), the call graph (call -> func) and the system dependence graph (call sites
                              -> entries), every node with GLA's structural codes (depth, sibling, in / out degree,
                              line bucket) and hashed text
  text_graph(text)            words in order (next) and their neighbours two apart (near)
  sample(code, text, answer)  a gla_sample from source code, a text about it and the answer's token ids
"""
import ast
import hashlib
import math
import re
from collections import defaultdict

from . import gla as GL

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\sA-Za-z0-9_]")
_PIECES = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def vocab_id(tok):
    """A token's text id (GLA: sha1 % 8192)."""
    return int(hashlib.sha1(tok.encode()).hexdigest(), 16) % GL.TOKEN_BUCKETS


def hash_id(tok):
    """A token's sub-token id (GLA: md5 % 4096) - the first snake_case / camelCase piece."""
    parts = _PIECES.findall(tok.split(".")[-1]) or [tok]
    return int(hashlib.md5(parts[0].lower().encode()).hexdigest(), 16) % GL.SUBTOKEN_BUCKETS


class _Builder(ast.NodeVisitor):
    """GLA's _PythonCPGBuilder (v79): DFG / PDG built while walking, with a scope stack (functions, lambdas and
    comprehensions open scopes; global / nonlocal redirect), right-hand sides before targets."""

    def __init__(self):
        self.nodes, self.edges = [], []
        self.func_map, self.call_sites = {}, []
        self._prev_ast = self._prev_cfg = None
        self._parents = []
        self._scopes, self._redirect = [{}], [{}]

    def add(self, ntype, label, line=0):
        self.nodes.append(dict(ntype=ntype, label=str(label), line=int(line or 0)))
        return len(self.nodes) - 1

    def edge(self, a, b, rel):
        self.edges.append((a, rel, b))

    @staticmethod
    def label(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return repr(node.value)[:64]
        if isinstance(node, ast.Attribute):
            return f"{_Builder.label(node.value)}.{node.attr}"
        return type(node).__name__

    @staticmethod
    def root_name(node):
        cur = node
        for _ in range(25):
            if isinstance(cur, ast.Name):
                return cur
            if isinstance(cur, (ast.Attribute, ast.Subscript)):
                cur = cur.value
            else:
                return None
        return None

    def _target_scope(self, name):
        r = self._redirect[-1].get(name)
        if r == "global":
            return self._scopes[0]
        if r == "nonlocal":
            for sc in reversed(self._scopes[1:-1]):
                if name in sc:
                    return sc
            return self._scopes[-2] if len(self._scopes) > 1 else self._scopes[0]
        return self._scopes[-1]

    def _link(self, dfg):
        cur = next((v for v in reversed(self._parents) if v is not None), None)
        if cur is not None:
            self.edge(cur, dfg, "maps_to")

    def define(self, name, line, prior_use=False):
        if prior_use:
            self.use(name, line)
        d = self.add("dfg", f"def:{name}", line)
        self._link(d)
        p = self.add("pdg", f"dep:{name}", line)
        self.edge(d, p, "maps_to")
        self._target_scope(name)[name] = (d, p)

    def use(self, name, line):
        r = self._redirect[-1].get(name)
        scopes = [self._scopes[0]] if r == "global" else (
            list(reversed(self._scopes[1:-1])) or [self._scopes[0]]) if r == "nonlocal" else list(reversed(self._scopes))
        for sc in scopes:
            if name in sc:
                d0, p0 = sc[name]
                u = self.add("dfg", f"use:{name}", line)
                self._link(u)
                self.edge(d0, u, "dataflow")
                pu = self.add("pdg", f"use_dep:{name}", line)
                self.edge(p0, pu, "data_dep")
                return

    def params(self, args, line):
        for a in (list(args.posonlyargs) + list(args.args) + ([args.vararg] if args.vararg else [])
                  + list(args.kwonlyargs) + ([args.kwarg] if args.kwarg else [])):
            self.define(a.arg, getattr(a, "lineno", line) or line)

    def visit(self, node):
        self._parents.append(None)
        try:
            return self._visit(node)
        finally:
            self._parents.pop()

    def _scoped(self, fn):
        self._scopes.append({})
        self._redirect.append({})
        try:
            fn()
        finally:
            self._redirect.pop()
            self._scopes.pop()

    def _visit(self, node):
        line = getattr(node, "lineno", 0) or 0
        lab = self.label(node)
        a = self.add("ast", lab, line)
        par = self._parents[-2] if len(self._parents) >= 2 else None
        if par is not None:
            self.edge(par, a, "child")
        self._parents[-1] = a
        c = self.add("cfg", f"cfg:{lab}", line)
        self.edge(a, c, "maps_to")
        if self._prev_ast is not None:
            self.edge(self._prev_ast, a, "next")
        if self._prev_cfg is not None:
            self.edge(self._prev_cfg, c, "flow")
        self._prev_ast, self._prev_cfg = a, c
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            f = self.add("func", node.name, line)
            self.func_map[node.name] = f
            self.edge(f, a, "contains")
            blk = self.add("block", f"body:{node.name}", line)
            self.edge(c, blk, "branch")
            for dec in node.decorator_list:
                self.visit(dec)
            for dflt in list(node.args.defaults) + list(node.args.kw_defaults):
                if dflt is not None:
                    self.visit(dflt)
            if node.returns is not None:
                self.visit(node.returns)

            def body():
                self.params(node.args, line)
                for st in node.body:
                    self.visit(st)
            self._scoped(body)
            return c
        if isinstance(node, ast.Lambda):
            for dflt in list(node.args.defaults) + list(node.args.kw_defaults):
                if dflt is not None:
                    self.visit(dflt)

            def body():
                self.params(node.args, line)
                self.visit(node.body)
            self._scoped(body)
            return c
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            def body():
                for gen in node.generators:
                    self.visit(gen.iter)
                    self.visit(gen.target)
                    for cond in gen.ifs:
                        self.visit(cond)
                if isinstance(node, ast.DictComp):
                    self.visit(node.key)
                    self.visit(node.value)
                else:
                    self.visit(node.elt)
            self._scoped(body)
            return c
        if isinstance(node, ast.ExceptHandler):
            if node.type is not None:
                self.visit(node.type)
            if node.name:
                self.define(node.name, line)
            for st in node.body:
                self.visit(st)
            return c
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                self._redirect[-1][name] = "global" if isinstance(node, ast.Global) else "nonlocal"
            return c
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                bound = al.asname or al.name.split(".")[0]
                if bound and bound != "*":
                    self.define(bound, line)
            return c
        if isinstance(node, (ast.For, ast.AsyncFor)):
            br = self.add("block", f"branch:{type(node).__name__}", line)
            self.edge(c, br, "branch")
            self.visit(node.iter)
            self.visit(node.target)
            for st in node.body + node.orelse:
                self.visit(st)
            return c
        if isinstance(node, (ast.If, ast.While)):
            br = self.add("block", f"branch:{type(node).__name__}", line)
            self.edge(c, br, "branch")
            ctrl = self.add("pdg", f"ctrl:{lab}", line)
            self.edge(c, ctrl, "maps_to")
            self.visit(node.test)
            for stmts in (node.body, node.orelse):
                if not stmts:
                    continue
                entry = self.visit(stmts[0])
                for st in stmts[1:]:
                    self.visit(st)
                if entry is not None:
                    pb = self.add("pdg", f"ctrl_body:{lab}", line)
                    self.edge(entry, pb, "maps_to")
                    self.edge(ctrl, pb, "control_dep")
            return c
        if isinstance(node, ast.NamedExpr):
            self.visit(node.value)
            self.visit(node.target)
            return c
        if isinstance(node, ast.AugAssign):
            self.visit(node.value)
            if isinstance(node.target, ast.Name):
                self.define(node.target.id, line, prior_use=True)
            else:
                self.visit(node.target)
                r = self.root_name(node.target)
                if r is not None:
                    self.define(r.id, line, prior_use=True)
            return c
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is not None:
                self.visit(node.value)
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                self.visit(t)
            return c
        if isinstance(node, ast.Call):
            fn = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(
                node.func, ast.Attribute) else "unknown"
            cid = self.add("call", fn, line)
            self.edge(a, cid, "child")
            self.call_sites.append((cid, fn, line))
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                self.define(node.id, line)
            else:
                self.use(node.id, line)
            return c
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, ast.Store):
            for ch in ast.iter_child_nodes(node):
                self.visit(ch)
            r = self.root_name(node)
            if r is not None:
                self.define(r.id, line)
            return c
        for ch in ast.iter_child_nodes(node):
            self.visit(ch)
        return c

    def finish(self):
        for cid, fn, line in self.call_sites:          # call graph, then the system dependence graph
            if fn in self.func_map:
                self.edge(cid, self.func_map[fn], "invokes")
        for cid, fn, line in self.call_sites:
            if fn not in self.func_map:
                continue
            s = self.add("sdg", f"callsite:{fn}", line)
            d = self.add("sdg", f"entry:{fn}", line)
            self.edge(s, d, "inter_proc")
            self.edge(cid, s, "child")
            self.edge(s, self.func_map[fn], "invokes")


def _with_ids(nodes, edges, slot=0, fam=2):
    """GLA's tensorizer: hashed text and the five structural codes of every node."""
    n = len(nodes)
    parent, sib, indeg, outdeg = [-1] * n, [0] * n, [0] * n, [0] * n
    kids = defaultdict(int)
    for a, rel, b in edges:
        outdeg[a] += 1
        indeg[b] += 1
        if rel in ("child", "contains") and parent[b] < 0 and a != b:
            parent[b], sib[b] = a, kids[a]
            kids[a] += 1
    depth = [-1] * n

    def dep(i):
        chain = []
        while i >= 0 and depth[i] < 0 and len(chain) < 64:
            chain.append(i)
            i = parent[i]
        d = depth[i] if i >= 0 and depth[i] >= 0 else -1
        for j in reversed(chain):
            d += 1
            depth[j] = d
    lines = [nd["line"] for nd in nodes]
    lo = min(lines) if lines else 0
    span = max(1, (max(lines) if lines else 0) - lo)
    out = []
    for i, nd in enumerate(nodes):
        if depth[i] < 0:
            dep(i)
        lab = nd["label"]
        tok = lab.split(":", 1)[-1] if nd["ntype"] in ("dfg", "pdg", "sdg", "cfg", "block") else lab
        out.append(GL.node(nd["ntype"], vocab_id(tok), min(max(depth[i], 0), 31), min(sib[i], 31),
                           line=min(int(31 * (lines[i] - lo) / span), 31), slot=slot, fam=fam, sub=hash_id(tok),
                           in_deg=min(int(math.log2(1 + indeg[i])), 15),
                           out_deg=min(int(math.log2(1 + outdeg[i])), 15)))
    return out


def code_graph(source, max_nodes=None):
    """Python source -> (nodes, edges) for a gla_sample: GLA's code property graph (ast, cfg, dfg, pdg, call, sdg,
    func, block nodes). max_nodes: keep the first nodes only (the edges among them)."""
    b = _Builder()
    b.visit(ast.parse(source))
    b.finish()
    nodes, edges = b.nodes, b.edges
    if max_nodes is not None and len(nodes) > int(max_nodes):
        k = int(max_nodes)
        nodes, edges = nodes[:k], [e for e in edges if e[0] < k and e[2] < k]
    return _with_ids(nodes, edges), [(a, r, z) for a, r, z in edges]


def text_graph(text, max_words=512):
    """Text -> (nodes, edges): a word node per token, next and near (two apart) edges."""
    words = _WORD.findall(str(text))[:int(max_words)] or ["<empty>"]
    nodes = [GL.node("word", vocab_id(w), 0, min(i, 31), line=min(int(31 * i / max(1, len(words) - 1)), 31),
                     sub=hash_id(w), fam=8) for i, w in enumerate(words)]
    edges = [(i - 1, "next", i) for i in range(1, len(words))] + [(i - 2, "near", i) for i in range(2, len(words))]
    return nodes, edges


def sample(code, text, answer, valid=1.0, issues=None, max_nodes=None):
    """A gla_sample: the graph of `code` (Python source), the graph of `text`, the answer's token ids."""
    cn, ce = code_graph(code, max_nodes)
    tn, te = text_graph(text)
    return GL.Sample(cn, ce, tn, te, list(answer), valid, issues)


def names():
    return dict(gla_code_graph=code_graph, gla_text_graph=text_graph, gla_from_code=sample)
