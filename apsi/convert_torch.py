"""PyTorch → native Apsi: rewrites the PyTorch idioms of a Python program before `apsi convert` translates it.

The result runs entirely on Apsi (no Python, no PyTorch):

  class Net(nn.Module) with __init__/forward     -> fn Net(...) returning custom_layer(init, forward): parameters
                                                    become an explicit tree, self.layer(x) -> layer(p.layer, x)
  nn.Linear / Conv2d / LSTM / Embedding / …        -> the `use layers` equivalents with PyTorch's defaults
  model = Net(...)                                 -> model + model_params = model.init(seed=0)
  model(x)                                         -> model(model_params, x)
  opt = torch.optim.Adam(model.parameters(), …)    -> opt = adam(…)
  opt.zero_grad(); loss = …; loss.backward();      -> loss, grads = value_and_grad(fn(p) => …)(model_params)
  opt.step()                                          model_params = opt.step(model_params, grads)
  clip_grad_norm_(model.parameters(), m)           -> grads = clip_grad_norm(grads, m)
  nn.CrossEntropyLoss / MSELoss / L1Loss / BCE…    -> the same losses (mean reduction)
  DataLoader(TensorDataset(X, y), batch_size, shuffle) -> mini-batches, reshuffled every epoch
  with torch.no_grad(), .item(), .detach(), .to(device), model.eval() / train() -> removed (not needed)
  torch.save(model.state_dict(), p) / load_state_dict(torch.load(p)) -> save_model / load_model (.apsim)
  tensor methods x.sum(dim=1), x.view(…), x.size(0), x.float(), x.argmax(1) -> Apsi functions

Everything that cannot be translated exactly is reported with its line number.
"""
import ast
import copy

APSI = "__apsi__"


def _compat_names():
    from .torch_compat import module
    return set(module())


COMPAT = _compat_names()
OPTIM_NAMES = {"SGD", "Adam", "AdamW", "Adamax", "RMSprop", "Adagrad", "NAdam", "RAdam"}
SCHED_NAMES = {"StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "LinearLR",
               "ConstantLR", "LambdaLR", "MultiplicativeLR", "PolynomialLR", "OneCycleLR", "CyclicLR",
               "ReduceLROnPlateau", "SequentialLR", "ChainedScheduler"}
LOSS_NAMES = {"CrossEntropyLoss", "NLLLoss", "MSELoss", "L1Loss", "SmoothL1Loss", "HuberLoss", "BCELoss",
              "BCEWithLogitsLoss", "KLDivLoss"}
DROP_KW = {"device", "dtype", "inplace", "foreach", "fused", "capturable", "differentiable"}

# PyTorch layer -> (Apsi layer, positional parameter names in PyTorch order, defaults that differ in Apsi)
LAYERS = {
    "Linear": ("layers.linear", ["in_features", "out_features", "bias"], {}),
    "Conv2d": ("layers.conv2d_layer", ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
               {"padding": 0, "stride": 1}),
    "MaxPool2d": ("layers.max_pool", ["kernel_size", "stride"], {}),
    "AvgPool2d": ("layers.avg_pool", ["kernel_size", "stride"], {}),
    "AdaptiveAvgPool2d": ("layers.global_avg_pool", ["output_size"], {}),
    "Flatten": ("layers.flatten_layer", [], {}),
    "Dropout": ("layers.dropout_layer", ["p"], {"p": 0.5}),
    "LayerNorm": ("layers.layer_norm_layer", ["normalized_shape", "eps"], {}),
    "RMSNorm": ("layers.rms_norm_layer", ["normalized_shape", "eps"], {}),
    "BatchNorm1d": ("layers.batch_norm_layer", ["num_features", "eps"], {}),
    "BatchNorm2d": ("layers.batch_norm_layer", ["num_features", "eps"], {}),
    "Embedding": ("layers.embedding", ["num_embeddings", "embedding_dim"], {}),
    "LSTM": ("layers.lstm_layer", ["input_size", "hidden_size"], {}),
    "GRU": ("layers.gru_layer", ["input_size", "hidden_size"], {}),
    "MultiheadAttention": ("layers.multi_head_attention", ["embed_dim", "num_heads"], {}),
    "Identity": ("layers.identity", [], {}),
}
ACTS = {"ReLU": "relu", "GELU": "gelu_exact", "Tanh": "tanh", "Sigmoid": "sigmoid", "SiLU": "silu", "Softmax": "softmax",
        "LeakyReLU": "leaky_relu", "ELU": "elu", "Mish": "mish", "Softplus": "softplus"}
LOSSES = {"CrossEntropyLoss": "cross_entropy", "MSELoss": "mse", "L1Loss": "l1_loss",
          "BCEWithLogitsLoss": "bce_with_logits", "BCELoss": "bce", "NLLLoss": "nll_loss", "SmoothL1Loss": "huber_loss",
          "HuberLoss": "huber_loss"}
OPTIMS = {"Adam": "adam", "AdamW": "adamw", "SGD": "sgd", "RMSprop": "rmsprop", "Adagrad": "adagrad"}
# tensor methods that become torch.<name>(x, …) and are then translated by the main converter
TENSOR_FUNCS = {"sum", "mean", "std", "var", "prod", "max", "min", "argmax", "argmin", "cumsum", "abs", "sqrt",
                "exp", "log", "log10", "log2", "sin", "cos", "tan", "tanh", "sigmoid", "relu", "softmax",
                "log_softmax", "matmul", "transpose", "flatten", "clamp", "clip", "floor", "ceil", "round",
                "norm", "dot", "outer", "sort", "roll", "where", "reshape"}
IDENTITY_METHODS = {"item", "detach", "cpu", "cuda", "numpy", "clone", "contiguous", "requires_grad_", "tolist"}
DTYPE_METHODS = {"float": "float32", "double": "float64", "half": "float16", "long": "int64", "int": "int32"}


def _name(n):
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        b = _name(n.value)
        return f"{b}.{n.attr}" if b else None
    return None


def _apsi(path, *args, **kw):
    """AST for a call to an Apsi function: __apsi__.<path>(args, kw)."""
    func = ast.Name(APSI, ast.Load())
    for part in path.split("."):
        func = ast.Attribute(func, part, ast.Load())
    return ast.Call(func, list(args), [ast.keyword(k, v) for k, v in kw.items()])


def _apsi_ref(path):
    node = ast.Name(APSI, ast.Load())
    for part in path.split("."):
        node = ast.Attribute(node, part, ast.Load())
    return node


def _c(v):
    return ast.Constant(v)


def _load(name):
    return ast.Name(name, ast.Load())


def _store(name):
    return ast.Name(name, ast.Store())


def _assign(name, value, like=None):
    a = ast.Assign([_store(name)], value)
    if like is not None:
        ast.copy_location(a, like)
    return a


def _expr(call, like=None):
    e = ast.Expr(call)
    if like is not None:
        ast.copy_location(e, like)
    return e


def _file_fn(path_node, op):
    """models.save_model / load_model (.apsim), or the safetensors functions when the file name says so."""
    if isinstance(path_node, ast.Constant) and isinstance(path_node.value, str) and \
            path_node.value.lower().endswith(".safetensors"):
        return f"models.{op}_safetensors"
    return f"models.{op}_model"


class TorchRewriter:
    def __init__(self, tree, warn):
        self.tree = tree
        self.warn = warn                       # warn(node, message)
        self.torch_alias = {}                  # local name -> torch dotted name
        self.module_classes = {}               # class name -> ClassDef
        self.models = {}                       # variable -> params variable
        self.optims = {}                       # optimizer variable -> model variable
        self.losses = {}                       # loss-module variable -> kind
        self.loaders = set()
        self.scheds = set()
        self.tmp = 0
        self.changed = False

    # -------------------------------------------------------------- helpers
    def new(self, base):
        self.tmp += 1
        return f"_{base}{self.tmp}"

    def full(self, node):
        """Dotted torch name of a node (resolving import aliases), or None."""
        d = _name(node)
        if not d:
            return None
        root = d.split(".")[0]
        if root in self.torch_alias:
            return self.torch_alias[root] + d[len(root):]
        return None

    def torch_kind(self, call, prefix):
        """'Linear' for nn.Linear(...) etc. when call constructs torch.<prefix>.<Name>."""
        if not isinstance(call, ast.Call):
            return None
        f = self.full(call.func)
        if f and f.startswith(prefix + "."):
            return f[len(prefix) + 1:]
        return None

    # -------------------------------------------------------------- entry
    def run(self):
        for s in ast.walk(self.tree):
            if isinstance(s, ast.Import):
                for a in s.names:
                    if a.name == "torch" or a.name.startswith("torch."):
                        self.torch_alias[a.asname or a.name.split(".")[0]] = a.name if a.asname else "torch"
            elif isinstance(s, ast.ImportFrom) and s.module and (s.module == "torch" or s.module.startswith("torch.")
                                                                 or s.module == "safetensors.torch"):
                for a in s.names:
                    self.torch_alias[a.asname or a.name] = f"{s.module}.{a.name}"
        if not self.torch_alias:
            return False
        for s in self.tree.body:
            if isinstance(s, ast.ClassDef) and any((self.full(b) or _name(b) or "").endswith("Module") for b in s.bases):
                self.module_classes[s.name] = s
        self.tree.body = self.stmts(self.tree.body, top=True)
        ast.fix_missing_locations(self.tree)
        return True

    # -------------------------------------------------------------- statements
    def stmts(self, body, top=False, ctx=None):
        out = []
        body = self.training_steps(body, ctx)
        for s in body:
            r = self.stmt(s, ctx)
            if r is None:
                continue
            out.extend(r if isinstance(r, list) else [r])
        return out or [ast.Pass()]

    def stmt(self, s, ctx):
        if isinstance(s, ast.ClassDef) and s.name in self.module_classes:
            self.changed = True
            return self.convert_module(s)
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            return s
        if getattr(s, "_apsi_done", False):
            return s
        if isinstance(s, ast.With):
            ctxs = [self.full(i.context_expr.func) if isinstance(i.context_expr, ast.Call) else None for i in s.items]
            if all(c in ("torch.no_grad", "torch.inference_mode", "torch.enable_grad", "torch.autocast",
                         "torch.cuda.amp.autocast", "torch.amp.autocast") for c in ctxs):
                return self.stmts(s.body, ctx=ctx)
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
            f = s.value.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                v, m = f.value.id, f.attr
                if v in self.models and m in ("to", "cuda", "cpu", "double", "float", "half", "zero_grad",
                                              "requires_grad_", "share_memory"):
                    return None
                if v in self.models and m in ("train", "eval"):
                    s.value.args = [self.expr(a, ctx) for a in s.value.args]
                    return s
                if v in self.models and m == "load_state_dict" and s.value.args:
                    return _assign(self.models[v], _apsi("torch.load_state_dict", _load(v), _load(self.models[v]),
                                                          self.expr(s.value.args[0], ctx)), s)
                if v in self.optims and m == "zero_grad":
                    return None
                if m == "step" and v in self.scheds:
                    s.value.args = [self.expr(a, ctx) for a in s.value.args]
                    return s
                if m == "step" and v not in self.optims and v != "self":
                    self.warn(s, f"{v}.step(): not recognised as an optimizer or scheduler step")
                    return s
            full = self.full(f)
            if full in ("torch.manual_seed", "torch.cuda.manual_seed", "torch.cuda.manual_seed_all"):
                return _expr(_apsi("seed", self.expr(s.value.args[0], ctx)), s) if full == "torch.manual_seed" \
                    else None
            if full in ("torch.set_num_threads", "torch.backends.cudnn.benchmark", "torch.set_float32_matmul_precision",
                        "torch.set_default_dtype", "torch.use_deterministic_algorithms"):
                return None
            if full == "torch.save" and len(s.value.args) >= 2:
                obj = s.value.args[0]
                if isinstance(obj, ast.Call) and isinstance(obj.func, ast.Attribute) and obj.func.attr == "state_dict" \
                        and isinstance(obj.func.value, ast.Name) and obj.func.value.id in self.models:
                    mv = obj.func.value.id
                    obj_e = _apsi("torch.state_dict", _load(mv), _load(self.models[mv]))
                else:
                    obj_e = self.expr(obj, ctx)
                return _expr(_apsi(_file_fn(s.value.args[1], "save"), self.expr(s.value.args[1], ctx), obj_e), s)
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
            return self.assign(s, s.targets[0].id, s.value, ctx)
        if ctx and ctx.get("self_attrs") is not None:
            r = self.forward_special(s, ctx)
            if r is not None:
                return r
        if isinstance(s, ast.FunctionDef):
            s.body = self.stmts(s.body, ctx=ctx)
            return s
        for field in ("body", "orelse", "finalbody"):
            if hasattr(s, field) and isinstance(getattr(s, field), list) and getattr(s, field) and \
                    isinstance(getattr(s, field)[0], ast.stmt):
                setattr(s, field, self.stmts(getattr(s, field), ctx=ctx))
        if isinstance(s, ast.Try):
            for h in s.handlers:
                h.body = self.stmts(h.body, ctx=ctx)
        return self.expr_in_stmt(s, ctx)

    def forward_special(self, s, ctx):
        """Statements inside forward(): loops over nn.ModuleList and tuple results of LSTM/GRU/attention."""
        attrs, local = ctx["self_attrs"], ctx["local"]
        if isinstance(s, ast.For):
            it = s.iter
            enum = isinstance(it, ast.Call) and _name(it.func) == "enumerate" and len(it.args) == 1
            src = it.args[0] if enum else it
            if isinstance(src, ast.Attribute) and isinstance(src.value, ast.Name) and src.value.id == "self" and \
                    attrs.get(src.attr) == "modlist":
                tgt = s.target.elts[1] if enum and isinstance(s.target, ast.Tuple) else s.target
                if not isinstance(tgt, ast.Name):
                    return None
                pv = self.new("p_" + tgt.id)
                zipped = ast.Call(_load("zip"), [_load(local[src.attr]), ast.Attribute(_load("p"), src.attr,
                                                                                        ast.Load())], [])
                pair = ast.Tuple([_store(tgt.id), _store(pv)], ast.Store())
                if enum:
                    s.target = ast.Tuple([s.target.elts[0], pair], ast.Store())
                    s.iter = ast.Call(_load("enumerate"), [zipped], [])
                else:
                    s.target = pair
                    s.iter = zipped
                inner = dict(ctx)
                inner["loopvars"] = dict(ctx.get("loopvars", {}), **{tgt.id: pv})
                s.body = self.stmts(s.body, ctx=inner)
                return s
        return None

    def _bind_state(self, target, value, like):
        if isinstance(target, ast.Name):
            return [ast.copy_location(ast.Assign([target], value), like)]
        if isinstance(target, ast.Tuple):
            out = []
            for k, e in enumerate(target.elts):
                out.extend(self._bind_state(e, value if k == 0 else _c(None), like))
            return out
        return []

    def expr_in_stmt(self, s, ctx):
        for field, value in ast.iter_fields(s):
            if isinstance(value, ast.expr):
                setattr(s, field, self.expr(value, ctx))
            elif isinstance(value, list):
                setattr(s, field, [self.expr(v, ctx) if isinstance(v, ast.expr) else v for v in value])
        return s

    def assign(self, s, name, value, ctx):
        if ctx and ctx.get("no_models"):                       # inside a module's __init__: plain values
            s.value = self.expr(value, None)
            return s
        # model construction
        if isinstance(value, ast.Call):
            fn = _name(value.func)
            kind = self.torch_kind(value, "torch.nn")
            if (fn in self.module_classes) or (kind and kind in COMPAT and kind not in LOSS_NAMES and
                                               kind[0].isupper()):
                pname = f"{name}_params"
                self.models[name] = pname
                self.changed = True
                return [_assign(name, self.expr(value, ctx), s),
                        _assign(pname, ast.Call(ast.Attribute(_load(name), "init", ast.Load()), [],
                                                [ast.keyword("seed", _c(0))]), s)]
            if kind in LOSS_NAMES:
                return _assign(name, self.torch_call(kind, value, ctx), s)
            okind = self.torch_kind(value, "torch.optim")
            if okind in OPTIM_NAMES:
                model = None
                if value.args:
                    a0 = value.args[0]
                    if isinstance(a0, ast.Call) and isinstance(a0.func, ast.Attribute) and \
                            isinstance(a0.func.value, ast.Name) and a0.func.attr == "parameters":
                        model = a0.func.value.id
                    elif isinstance(a0, (ast.List, ast.Tuple)) and any(isinstance(e, ast.Dict) for e in a0.elts):
                        self.warn(s, f"{okind} with parameter groups: one learning rate is used for all parameters")
                if model not in self.models:
                    self.warn(s, f"optimizer {okind}: could not find which model it trains")
                self.optims[name] = model
                return _assign(name, self.optimizer(okind, value, s), s)
            if okind and okind.startswith("lr_scheduler."):
                sk = okind.split(".", 1)[1]
                if sk in SCHED_NAMES:
                    self.scheds.add(name)
                    return _assign(name, self.torch_call(sk, value, ctx), s)
                self.warn(s, f"torch.optim.{okind} is not available natively")
                return None
            if self.torch_kind(value, "torch.utils.data") == "DataLoader":
                return self.dataloader(s, name, value, ctx)
            full = self.full(value.func)
            if full == "torch.device":
                return _assign(name, _c("cpu"), s)
            if full == "torch.load" and value.args:
                return _assign(name, _apsi(_file_fn(value.args[0], "load"), self.expr(value.args[0], ctx)), s)
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and \
                isinstance(value.func.value, ast.Name) and value.func.value.id in self.models and \
                value.func.attr in ("to", "cuda", "cpu", "double", "float"):
            self.models[name] = self.models[value.func.value.id]
            return _assign(name, _load(value.func.value.id), s)
        s.value = self.expr(value, ctx)
        return s

    # -------------------------------------------------------------- pieces
    def kw(self, call, names, ctx=None):
        """Arguments of a torch call by PyTorch parameter name."""
        out = {}
        for i, a in enumerate(call.args):
            if i < len(names):
                out[names[i]] = self.expr(a, ctx)
        for k in call.keywords:
            if k.arg:
                out[k.arg] = self.expr(k.value, ctx)
        return out

    def torch_call(self, kind, call, ctx, skip_first=False):
        """__apsi__.torch.<kind>(same arguments as in PyTorch)."""
        args = [self.expr(x, ctx) for x in call.args[(1 if skip_first else 0):]]
        kws = {k.arg: self.expr(k.value, ctx) for k in call.keywords if k.arg and k.arg not in DROP_KW}
        return _apsi(f"torch.{kind}", *args, **kws)

    def layer(self, kind, call, ctx):
        """nn.X(...) -> the native PyTorch-compatible layer with the same arguments."""
        if kind == "ModuleList":
            return self.expr(call.args[0], ctx) if call.args else ast.List([], ast.Load())
        if kind in COMPAT:
            return self.torch_call(kind, call, ctx)
        self.warn(call, f"nn.{kind} is not available natively yet")
        return None

    def optimizer(self, kind, call, node):
        names = {"SGD": ["lr", "momentum", "dampening", "weight_decay", "nesterov"],
                 "Adam": ["lr", "betas", "eps", "weight_decay", "amsgrad"],
                 "AdamW": ["lr", "betas", "eps", "weight_decay", "amsgrad"],
                 "RMSprop": ["lr", "alpha", "eps", "weight_decay", "momentum", "centered"],
                 "Adagrad": ["lr", "lr_decay", "weight_decay", "initial_accumulator_value", "eps"],
                 "Adamax": ["lr", "betas", "eps", "weight_decay"], "NAdam": ["lr", "betas", "eps", "weight_decay"],
                 "RAdam": ["lr", "betas", "eps", "weight_decay"]}[kind]
        kws = {n: self.expr(v) for n, v in zip(names, call.args[1:])}
        kws.update({k.arg: self.expr(k.value) for k in call.keywords if k.arg and k.arg not in DROP_KW})
        return _apsi(f"torch.{kind}", **kws)

    def dataloader(self, s, name, call, ctx):
        self.loaders.add(name)
        return _assign(name, self.torch_call("DataLoader", call, ctx), s)

    # -------------------------------------------------------------- training steps
    def training_steps(self, body, ctx):
        """Find `L.backward()` … `opt.step()` and rebuild them as value_and_grad + opt.step."""
        out = list(body)
        i = 0
        while i < len(out):
            s = out[i]
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Attribute) \
                    and s.value.func.attr == "backward" and not s.value.args:
                j = next((k for k in range(i + 1, len(out)) if self._is_opt_step(out[k])), None)
                if j is None:
                    self.warn(s, ".backward() without an optimizer step: use grad() / value_and_grad() directly")
                    i += 1
                    continue
                opt = out[j].value.func.value.id
                model = self.optims.get(opt)
                if model is None:
                    self.warn(s, "training step: the optimizer's model is unknown")
                    i += 1
                    continue
                start = 0
                for k in range(i - 1, -1, -1):
                    if self._is_zero_grad(out[k]) or self._is_opt_step(out[k]):
                        start = k + 1
                        break
                seg = [x for x in out[start:i] if not self._is_zero_grad(x)]
                between = out[i + 1:j]
                new = self.build_step(s, seg, s.value.func.value, model, opt, between, out[j + 1:])
                for x in new:
                    x._apsi_done = True
                out[start:j + 1] = new
                i = start + len(new)
                self.changed = True
                continue
            i += 1
        return out

    def _is_opt_step(self, s):
        return isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Attribute) \
            and s.value.func.attr == "step" and isinstance(s.value.func.value, ast.Name) \
            and s.value.func.value.id in self.optims

    def _is_zero_grad(self, s):
        return isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Attribute) \
            and s.value.func.attr == "zero_grad"

    def build_step(self, node, seg, loss_expr, model, opt, between, rest):
        """One training step: the statements that form the loss become a function of the parameters; the step is
        value_and_grad_aux of that function (values needed later, e.g. logits, come back as aux: nothing runs
        twice), optional clipping, then opt.step(params, grads)."""
        pname = self.models[model]
        lossfn = self.new("loss_of")
        grads = self.new("grads")
        p = "_p"
        lname = loss_expr.id if isinstance(loss_expr, ast.Name) else self.new("loss")
        assigned = []
        for x in seg:
            for n in ast.walk(x):
                if isinstance(n, ast.Assign):
                    for t in n.targets:
                        for e in (t.elts if isinstance(t, ast.Tuple) else [t]):
                            if isinstance(e, ast.Name) and e.id != lname and e.id not in assigned:
                                assigned.append(e.id)
        used_later = {n.id for x in rest + between for n in ast.walk(x) if isinstance(n, ast.Name)}
        need = [n for n in assigned if n in used_later]
        inner = []
        for x in [copy.deepcopy(x) for x in seg]:
            r = self.stmt(x, {"params": {model: p}})
            inner.extend(r if isinstance(r, list) else ([r] if r is not None else []))
        loss_v = self.expr(copy.deepcopy(loss_expr), {"params": {model: p}})
        aux_name = self.new("aux")
        if need:
            aux = ast.Call(_apsi_ref("record"), [], [ast.keyword(n, _load(n)) for n in need])
            inner.append(ast.Return(ast.List([loss_v, aux], ast.Load())))
        else:
            inner.append(ast.Return(loss_v))
        fdef = ast.FunctionDef(name=lossfn, args=ast.arguments(posonlyargs=[], args=[ast.arg(p)], vararg=None,
                                                               kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                               body=inner, decorator_list=[], returns=None, type_params=[])
        ast.copy_location(fdef, node)
        vg = ast.Assign([ast.Tuple([_store(lname), _store(aux_name), _store(grads)], ast.Store())],
                        ast.Call(_apsi("torch.value_and_grad_aux", _load(lossfn)), [_load(pname)], []))
        ast.copy_location(vg, node)
        stmts = [fdef, vg]
        for n in need:
            stmts.append(_assign(n, ast.Attribute(_load(aux_name), n, ast.Load()), node))
        for b in between:                                  # gradient clipping and similar
            if isinstance(b, ast.Expr) and isinstance(b.value, ast.Call):
                f = self.full(b.value.func) or ""
                if f.endswith("clip_grad_norm_") or f.endswith("clip_grad_norm"):
                    rest_args = [self.expr(x) for x in b.value.args[1:]]
                    kws = {k.arg: self.expr(k.value) for k in b.value.keywords if k.arg in ("max_norm", "norm_type")}
                    stmts.append(_assign(grads, _apsi("torch.clip_grad_norm", _load(grads), *rest_args, **kws), b))
                    continue
                if f.endswith("clip_grad_value_"):
                    rest_args = [self.expr(x) for x in b.value.args[1:]]
                    kws = {k.arg: self.expr(k.value) for k in b.value.keywords if k.arg == "clip_value"}
                    stmts.append(_assign(grads, _apsi("torch.clip_grad_value", _load(grads), *rest_args, **kws), b))
                    continue
            r = self.stmt(b, None)
            stmts.extend(r if isinstance(r, list) else ([r] if r is not None else []))
        stmts.append(_assign(pname, ast.Call(ast.Attribute(_load(opt), "step", ast.Load()),
                                             [_load(pname), _load(grads)], []), node))
        return stmts

    # -------------------------------------------------------------- nn.Module -> custom layer
    def convert_module(self, cls):
        methods = {m.name: m for m in cls.body if isinstance(m, ast.FunctionDef)}
        init = methods.get("__init__")
        fwd = methods.get("forward")
        if fwd is None:
            self.warn(cls, f"class {cls.name}: no forward() method")
            return cls
        attrs = {}                                     # attribute -> kind: module / modlist / param / buffer / const
        init_args = init.args.args[1:] if init else []
        argnames = {a.arg for a in init_args}
        local = {}
        rnn = {}

        def lname(attr):
            if attr not in local:
                local[attr] = attr + "_" if attr in argnames or attr in ("p", "train", "seed") else attr
            return local[attr]

        body = []
        if init:
            for st in init.body:
                if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
                    continue
                if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) and \
                        isinstance(st.value.func, ast.Attribute) and st.value.func.attr == "__init__":
                    continue
                if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) and \
                        isinstance(st.value.func, ast.Attribute) and st.value.func.attr == "register_buffer" and \
                        len(st.value.args) >= 2 and isinstance(st.value.args[0], ast.Constant):
                    attr = st.value.args[0].value
                    attrs[attr] = "const"
                    body.append(_assign(lname(attr), self.expr(st.value.args[1]), st))
                    continue
                if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Attribute) \
                        and isinstance(st.targets[0].value, ast.Name) and st.targets[0].value.id == "self":
                    attr = st.targets[0].attr
                    v = st.value
                    kind = self.torch_kind(v, "torch.nn")
                    if kind == "Parameter":
                        attrs[attr] = "param"
                        body.append(_assign(lname(attr), _apsi("tensor", self.expr(v.args[0])), st))
                        continue
                    if kind == "ModuleList":
                        attrs[attr] = "modlist"
                    elif (kind and kind in COMPAT and kind[0].isupper()) or \
                            (isinstance(v, ast.Call) and _name(v.func) in self.module_classes):
                        attrs[attr] = "module"
                        if kind in ("LSTM", "GRU"):
                            rnn[attr] = kind
                        elif kind == "MultiheadAttention":
                            rnn[attr] = "attention"
                    else:
                        attrs[attr] = attrs.get(attr, "const")
                    body.append(_assign(lname(attr), self.expr(self.self_refs(v, attrs, local)), st))
                    continue
                r = self.stmt(self.self_refs(copy.deepcopy(st), attrs, local, init_stmt=True), {"no_models": True})
                body.extend(r if isinstance(r, list) else ([r] if r is not None else []))
        # init(seed)
        seed = ast.Name("seed", ast.Load())
        fields = []
        k = 0
        for attr, kind in attrs.items():
            k += 1
            sd = ast.BinOp(ast.BinOp(seed, ast.Mult(), _c(1009)), ast.Add(), _c(k))
            if kind == "module":
                fields.append(ast.keyword(attr, ast.Call(ast.Attribute(_load(lname(attr)), "init", ast.Load()), [],
                                                         [ast.keyword("seed", sd)])))
            elif kind == "modlist":
                i_ = self.new("i")
                m_ = self.new("m")
                comp = ast.ListComp(
                    ast.Call(ast.Attribute(_load(m_), "init", ast.Load()), [],
                             [ast.keyword("seed", ast.BinOp(sd, ast.Add(), ast.BinOp(_load(i_), ast.Mult(),
                                                                                       _c(7919))))]),
                    [ast.comprehension(ast.Tuple([_store(i_), _store(m_)], ast.Store()),
                                       ast.Call(_load("enumerate"), [_load(lname(attr))], []), [], 0)])
                fields.append(ast.keyword(attr, comp))
            elif kind == "param":
                fields.append(ast.keyword(attr, _load(lname(attr))))
        if any(kd == "param" for kd in attrs.values()):
            self.warn(cls, f"class {cls.name}: nn.Parameter values are created once when the model is built "
                           f"(init(seed) returns them unchanged)")
        init_fn = ast.FunctionDef(name="init_", args=ast.arguments(posonlyargs=[], args=[ast.arg("seed")],
                                                                   vararg=None, kwonlyargs=[], kw_defaults=[],
                                                                   kwarg=None, defaults=[_c(0)]),
                                  body=[ast.Return(ast.Call(_apsi_ref("record"), [], fields))], decorator_list=[],
                                  returns=None, type_params=[])
        # helper methods and forward
        helpers = [m for n, m in methods.items() if n not in ("__init__", "forward") and not n.startswith("__")]
        hnames = {m.name for m in helpers}
        ctx = {"self_attrs": attrs, "local": local, "helpers": hnames, "train": "train", "params": {}, "rnn": rnn}
        funcs = []
        for m in helpers + [fwd]:
            fname = "forward_" if m is fwd else m.name
            args = [ast.arg("p")] + m.args.args[1:]
            defaults = [self.expr(d) for d in m.args.defaults]
            fdef = ast.FunctionDef(
                name=fname, args=ast.arguments(posonlyargs=[], args=args + [ast.arg("train")], vararg=None,
                                               kwonlyargs=[], kw_defaults=[], kwarg=None,
                                               defaults=defaults + [_c(False)]),
                body=self.stmts(copy.deepcopy(m.body), ctx=ctx), decorator_list=[], returns=None, type_params=[])
            ast.copy_location(fdef, m)
            funcs.append(fdef)
        parts = ast.Call(_apsi_ref("record"), [], [ast.keyword(a, _load(lname(a))) for a, kd in attrs.items()
                                                if kd in ("module", "modlist")])
        ret = ast.Return(_apsi("torch.Module", _load("init_"), _load("forward_"), _c(cls.name), parts))
        doc = ast.get_docstring(cls)
        fn_body = ([ast.Expr(_c(doc))] if doc else []) + body + [init_fn] + funcs + [ret]
        defaults = init.args.defaults if init else []
        new = ast.FunctionDef(name=cls.name, args=ast.arguments(
            posonlyargs=[], args=[ast.arg(a.arg) for a in init_args], vararg=None, kwonlyargs=[], kw_defaults=[],
            kwarg=None, defaults=[self.expr(d) for d in defaults]), body=fn_body, decorator_list=[], returns=None,
            type_params=[])
        ast.copy_location(new, cls)
        return new

    def self_refs(self, node, attrs, local, init_stmt=False):
        """Inside __init__: self.attr -> its local variable; self.layers.append(x) -> list append."""
        class R(ast.NodeTransformer):
            def visit_Attribute(s, n):
                s.generic_visit(n)
                if isinstance(n.value, ast.Name) and n.value.id == "self" and n.attr in local:
                    return ast.copy_location(ast.Name(local[n.attr], n.ctx), n)
                return n
        return R().visit(node)

    # -------------------------------------------------------------- expressions
    def expr(self, node, ctx=None):
        rw = self

        class X(ast.NodeTransformer):
            def visit_Attribute(s, n):
                s.generic_visit(n)
                if ctx and ctx.get("self_attrs") is not None and isinstance(n.value, ast.Name) and n.value.id == "self":
                    attrs, local = ctx["self_attrs"], ctx["local"]
                    if n.attr == "training":
                        return ast.copy_location(_load("train"), n)
                    kind = attrs.get(n.attr)
                    if kind == "param":
                        return ast.copy_location(ast.Attribute(_load("p"), n.attr, ast.Load()), n)
                    if kind in ("const", "module", "modlist"):
                        return ast.copy_location(_load(local[n.attr]), n)
                if isinstance(n.value, ast.Name) and n.value.id in rw.models and n.attr in ("training",):
                    return _c(False)
                if n.attr in ("values", "indices") and isinstance(n.value, ast.Tuple) and len(n.value.elts) == 2:
                    return n.value.elts[0 if n.attr == "values" else 1]
                return n

            def visit_Call(s, n):
                f = n.func
                # self.layer(x) / self.layers[i](x) / loop variable over self.layers
                if ctx and ctx.get("self_attrs") is not None:
                    attrs, local = ctx["self_attrs"], ctx["local"]
                    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self":
                        if attrs.get(f.attr) == "module":
                            n.args = [s.visit(a) for a in n.args]
                            n.keywords = [ast.keyword(k.arg, s.visit(k.value)) for k in n.keywords]
                            return rw.layer_call(_load(local[f.attr]), ast.Attribute(_load("p"), f.attr, ast.Load()),
                                                 n, ctx)
                        if f.attr in ctx.get("helpers", ()):
                            n.args = [s.visit(a) for a in n.args]
                            return ast.Call(_load(f.attr), [_load("p")] + n.args,
                                            [ast.keyword(k.arg, s.visit(k.value)) for k in n.keywords] +
                                            [ast.keyword("train", _load("train"))])
                    if isinstance(f, ast.Subscript) and isinstance(f.value, ast.Attribute) and \
                            isinstance(f.value.value, ast.Name) and f.value.value.id == "self" and \
                            attrs.get(f.value.attr) == "modlist":
                        idx = s.visit(f.slice)
                        n.args = [s.visit(a) for a in n.args]
                        n.keywords = [ast.keyword(k.arg, s.visit(k.value)) for k in n.keywords]
                        return rw.layer_call(ast.Subscript(_load(local[f.value.attr]), idx, ast.Load()),
                                             ast.Subscript(ast.Attribute(_load("p"), f.value.attr, ast.Load()),
                                                           copy.deepcopy(idx), ast.Load()), n, ctx)
                    loopvars = ctx.get("loopvars", {})
                    if isinstance(f, ast.Name) and f.id in loopvars:
                        n.args = [s.visit(a) for a in n.args]
                        n.keywords = [ast.keyword(k.arg, s.visit(k.value)) for k in n.keywords]
                        return rw.layer_call(_load(f.id), _load(loopvars[f.id]), n, ctx)
                # model(x) outside the class
                if isinstance(f, ast.Name) and f.id in rw.models:
                    n.args = [s.visit(a) for a in n.args]
                    n.keywords = [ast.keyword(k.arg, s.visit(k.value)) for k in n.keywords]
                    params = (ctx or {}).get("params", {}).get(f.id) or rw.models[f.id]
                    return ast.copy_location(ast.Call(_load(f.id), [_load(params)] + n.args, n.keywords), n)
                # nn.X(...) layer construction / functional losses
                kind = rw.torch_kind(n, "torch.nn")
                if kind and (kind in COMPAT or kind == "ModuleList"):
                    n.args = [s.visit(a) for a in n.args]
                    n.keywords = [ast.keyword(k.arg, s.visit(k.value)) for k in n.keywords]
                    out = rw.layer(kind, n, ctx)
                    if out is not None:
                        return ast.copy_location(out, n)
                fk = rw.torch_kind(n, "torch.nn.functional")
                if fk and fk in COMPAT:
                    n.args = [s.visit(a) for a in n.args]
                    kws = [ast.keyword(k.arg, s.visit(k.value)) for k in n.keywords if k.arg not in DROP_KW]
                    if fk == "dropout" and not any(k.arg == "training" for k in kws):
                        kws.append(ast.keyword("training", _c(True)))
                    return ast.copy_location(ast.Call(_apsi_ref(f"torch.{fk}"), n.args, kws), n)
                if fk in ("gelu",):
                    n.args = [s.visit(a) for a in n.args]
                    appr = next((k.value for k in n.keywords if k.arg == "approximate"), _c("none"))
                    return ast.copy_location(_apsi("gelu", n.args[0], approximate=s.visit(appr)), n)
                tk = rw.torch_kind(n, "torch.utils.data")
                if tk in ("TensorDataset", "DataLoader"):
                    n.args = [s.visit(a) for a in n.args]
                    n.keywords = [ast.keyword(k.arg, s.visit(k.value)) for k in n.keywords]
                    return ast.copy_location(rw.torch_call(tk, n, ctx), n)
                full = rw.full(f)
                if full == "safetensors.torch.load_file" and n.args:
                    return ast.copy_location(_apsi("models.load_safetensors", s.visit(n.args[0])), n)
                if full == "safetensors.torch.save_file" and len(n.args) >= 2:
                    return ast.copy_location(_apsi("models.save_safetensors", s.visit(n.args[1]), s.visit(n.args[0])), n)
                if full in ("torch.transpose", "torch.permute", "torch.unsqueeze", "torch.squeeze") and n.args:
                    fn_ = {"torch.transpose": "swapaxes", "torch.permute": "permute", "torch.unsqueeze": "unsqueeze",
                           "torch.squeeze": "squeeze"}[full]
                    args = [s.visit(a) for a in n.args] + [s.visit(k.value) for k in n.keywords if k.arg in ("dim",
                                                                                                         "dims")]
                    return ast.copy_location(ast.Call(_apsi_ref(fn_), args, []), n)
                if full in ("torch.max", "torch.min") and len(n.args) >= 2:
                    x_, d = s.visit(n.args[0]), s.visit(n.args[1])
                    res = ast.Call(_apsi_ref(full[6:]), [x_], [ast.keyword("axis", d)])
                    idx = ast.Call(_apsi_ref("arg" + full[6:]), [copy.deepcopy(x_)], [ast.keyword("axis", copy.deepcopy(d))])
                    return ast.copy_location(ast.Tuple([res, idx], ast.Load()), n)
                if full == "torch.cuda.is_available":
                    return _c(False)
                if full in ("torch.no_grad", "torch.inference_mode"):
                    return _c(None)
                if full == "torch.load" and n.args:
                    return _apsi(_file_fn(n.args[0], "load"), s.visit(n.args[0]))
                # sum(p.numel() for p in model.parameters())
                if isinstance(f, ast.Name) and f.id == "sum" and n.args and isinstance(n.args[0], ast.GeneratorExp):
                    g = n.args[0]
                    it = g.generators[0].iter
                    if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr == "parameters" \
                            and isinstance(it.func.value, ast.Name) and it.func.value.id in rw.models:
                        return _apsi("param_count", _load(rw.models[it.func.value.id]))
                # tensor / model methods
                if isinstance(f, ast.Attribute):
                    m = f.attr
                    recv = f.value
                    if isinstance(recv, ast.Name) and recv.id in rw.models:
                        if m in ("parameters", "named_parameters"):
                            return _load(rw.models[recv.id])
                        if m == "state_dict":
                            return _apsi("torch.state_dict", _load(recv.id), _load(rw.models[recv.id]))
                        if m in ("eval", "train", "to", "cuda", "cpu"):
                            return _load(recv.id)
                    if isinstance(recv, ast.Name) and recv.id in rw.loaders:
                        return n
                    if rw.full(recv) is not None:              # torch.nn.functional.relu(x) etc.: not a method
                        s.generic_visit(n)
                        return n
                    n.args = [s.visit(a) for a in n.args]
                    n.keywords = [ast.keyword(k.arg, s.visit(k.value)) for k in n.keywords]
                    rv = s.visit(recv)
                    is_self = isinstance(recv, ast.Name) and recv.id == "self"
                    if is_self or (isinstance(recv, ast.Name) and recv.id in rw.optims):
                        n.func = ast.Attribute(rv, m, ast.Load())
                        return n
                    if m in IDENTITY_METHODS and not n.args:
                        return rv
                    if m in ("to",):
                        return rv
                    if m in DTYPE_METHODS and not n.args:
                        return ast.copy_location(ast.Call(ast.Attribute(_load("torch"), "tensor", ast.Load()), [rv],
                                                          [ast.keyword("dtype", ast.Attribute(
                                                              _load("torch"), DTYPE_METHODS[m], ast.Load()))]), n)
                    if m == "size":
                        sh = _apsi("shape", rv)
                        return ast.Subscript(sh, n.args[0], ast.Load()) if n.args else sh
                    if m in ("view", "reshape") and n.args:
                        shape = n.args[0] if len(n.args) == 1 else ast.Tuple(n.args, ast.Load())
                        return _apsi("reshape", rv, shape) if len(n.args) == 1 else \
                            ast.Call(_apsi_ref("reshape"), [rv] + n.args, [])
                    if m == "t" and not n.args:
                        return _apsi("transpose", rv)
                    if m == "pow" and len(n.args) == 1:
                        return ast.BinOp(rv, ast.Pow(), n.args[0])
                    if m in ("unsqueeze", "squeeze", "permute") or (m == "transpose" and len(n.args) == 2):
                        fn_ = {"unsqueeze": "unsqueeze", "squeeze": "squeeze", "permute": "permute",
                               "transpose": "swapaxes"}[m]
                        args = n.args + [k.value for k in n.keywords if k.arg in ("dim", "dims")]
                        return ast.copy_location(ast.Call(_apsi_ref(fn_), [rv] + args, []), n)
                    if m in ("max", "min") and (n.args or any(k.arg == "dim" for k in n.keywords)):
                        d = n.args[0] if n.args else next(k.value for k in n.keywords if k.arg == "dim")
                        kd = [k for k in n.keywords if k.arg == "keepdim"]
                        res = ast.Call(_apsi_ref(m), [rv], [ast.keyword("axis", d)] +
                                       [ast.keyword("keepdims", k.value) for k in kd])
                        idx = ast.Call(_apsi_ref("arg" + m), [copy.deepcopy(rv)], [ast.keyword("axis", copy.deepcopy(d))])
                        return ast.copy_location(ast.Tuple([res, idx], ast.Load()), n)
                    if m in TENSOR_FUNCS and rw.torch_alias:
                        tname = next((a for a, full_ in rw.torch_alias.items() if full_ == "torch"), "torch")
                        call = ast.Call(ast.Attribute(_load(tname), m, ast.Load()), [rv] + n.args, n.keywords)
                        return ast.copy_location(call, n)
                    n.func = ast.Attribute(rv, m, ast.Load())
                    return n
                s.generic_visit(n)
                return n

        return X().visit(node)

    def layer_call(self, layer, params, call, ctx):
        """Apply a layer: layer(params, args..., keywords..., train=train)."""
        tr = _load("train") if ctx and ctx.get("self_attrs") is not None else _c(False)
        kws = [k for k in call.keywords if k.arg != "train"]
        return ast.Call(layer, [params] + list(call.args), kws + [ast.keyword("train", tr)])


def rewrite(tree, warn):
    """Rewrite PyTorch idioms in place. Returns True when the program uses PyTorch."""
    rw = TorchRewriter(tree, warn)
    used = rw.run()
    return used, rw
