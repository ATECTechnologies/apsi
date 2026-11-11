"""`use config` — settings in their own files, layered, overridable, checked.

    use config
    let cfg = config.load("config/default.toml", "config/local.toml?", env="APP", args=true)
    print(cfg.server.port, cfg.get("train.lr", 3e-4))
    config.save("run/used.toml", cfg)              # exactly what this run used

* Files: .toml, .json, .yaml/.yml (with PyYAML), .ini/.cfg, .env. A path ending in "?" is optional (local overrides
  that only some machines have). Later files override earlier ones key by key; tables are merged, not replaced.
* `env="APP"`: environment variables APP_SERVER__PORT=9000 override server.port (two underscores between levels);
  values are read as numbers / true / false / JSON lists when they look like them. Needs --allow-env.
* `args=true`: `--set server.port=9000` (or `server.port=9000`) on the program's command line override too.
* `defaults=record(...)`: values used when no file sets them; their types are the schema: a file that gives text
  where the default is a number is an error naming the file and key. `required=["db.url"]` lists keys that must
  be set. `${OTHER.KEY}` and `${ENV:NAME}` inside text values are replaced.
* The result is an ordinary record (cfg.server.port) with `cfg.get("a.b", default)`; `config.sources(cfg)` tells
  which file, variable or argument set each key.
"""
import configparser
import json
import os
import re
import sys

from .errors import RuntimeErr
from .values import Record


class ConfigError(RuntimeErr):
    pass


# ------------------------------------------------------------------ reading files
def read_file(path):
    low = path.lower()
    try:
        if low.endswith(".toml"):
            from .project import read_toml
            return read_toml(path)
        if low.endswith(".json"):
            with open(path, encoding="utf-8-sig") as fh:
                return json.load(fh)
        if low.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError:
                raise ConfigError(f"{path}: reading YAML needs PyYAML (pip install pyyaml); TOML and JSON need nothing")
            with open(path, encoding="utf-8-sig") as fh:
                return yaml.safe_load(fh) or {}
        if low.endswith((".ini", ".cfg", ".conf")):
            cp = configparser.ConfigParser(interpolation=None)
            cp.optionxform = str
            with open(path, encoding="utf-8-sig") as fh:
                cp.read_file(fh)
            out = {k: _guess(v) for k, v in cp.defaults().items()}
            for sec in cp.sections():
                node = out
                for part in sec.split("."):
                    node = node.setdefault(part, {})
                for k, v in cp.items(sec, raw=True):
                    if k not in cp.defaults() or cp.defaults()[k] != v:
                        node[k] = _guess(v)
            return out
        if low.endswith(".env") or os.path.basename(low).startswith(".env"):
            from .envfile import parse
            with open(path, encoding="utf-8-sig") as fh:
                flat = parse(fh.read())
            out = {}
            for k, v in flat.items():
                _set(out, k.lower().replace("__", "."), _guess(v))
            return out
    except ConfigError:
        raise
    except (ValueError, configparser.Error) as e:
        raise ConfigError(f"{path}: {e}")
    raise ConfigError(f"{path}: unknown settings format (use .toml, .json, .yaml, .ini or .env)")


def _guess(v):
    """Text from env/ini/args as the value it looks like."""
    if not isinstance(v, str):
        return v
    t = v.strip()
    low = t.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null"):
        return None
    if re.fullmatch(r"[+-]?\d[\d_]*", t):
        return int(t.replace("_", ""))
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", t):
        return float(t)
    if t[:1] in "[{":
        try:
            return json.loads(t)
        except ValueError:
            pass
    return v


def _set(d, dotted, value):
    parts = dotted.split(".")
    for p in parts[:-1]:
        nxt = d.get(p)
        if not isinstance(nxt, dict):
            nxt = d[p] = {}
        d = nxt
    d[parts[-1]] = value


def _get(d, dotted, default=None):
    for p in dotted.split("."):
        if isinstance(d, Record):
            d = d.fields
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def _plain(v):
    if isinstance(v, Record):
        return {k: _plain(x) for k, x in v.fields.items()}
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    try:
        from .interop import to_py
        return to_py(v)
    except Exception:
        return v


def merge(base, over, where, sources, prefix=""):
    for k, v in over.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge(base[k], v, where, sources, key + ".")
        else:
            base[k] = v
            for s in [s for s in sources if s == key or s.startswith(key + ".")]:
                del sources[s]
            _mark(v, key, where, sources)
    return base


def _mark(v, key, where, sources):
    if isinstance(v, dict):
        for k, x in v.items():
            _mark(x, f"{key}.{k}", where, sources)
    else:
        sources[key] = where


def _check_types(data, defaults, where_of, prefix=""):
    for k, d in defaults.items():
        key = f"{prefix}{k}"
        if k not in data:
            continue
        v = data[k]
        if isinstance(d, dict):
            if not isinstance(v, dict):
                raise ConfigError(f"{where_of.get(key, 'settings')}: '{key}' should be a table of settings")
            _check_types(v, d, where_of, key + ".")
            continue
        if d is None or v is None:
            continue
        ok = (isinstance(v, bool) == isinstance(d, bool)) and (
            isinstance(v, type(d)) or (isinstance(d, float) and isinstance(v, int) and not isinstance(v, bool)))
        if isinstance(d, (list, tuple)):
            ok = isinstance(v, (list, tuple))
        if not ok:
            kind = {bool: "true/false", int: "a whole number", float: "a number", str: "text"}.get(type(d), type(d).__name__)
            raise ConfigError(f"{where_of.get(key, 'settings')}: '{key}' should be {kind} (like the default "
                              f"{json.dumps(d)}), not {json.dumps(v, default=str)}")
        if isinstance(d, float) and isinstance(v, int):
            data[k] = float(v)


_REF = re.compile(r"\$\{([^}]+)\}")


def _expand(data, root, env_ok, depth=0):
    for k, v in list(data.items()):
        if isinstance(v, dict):
            _expand(v, root, env_ok, depth)
        elif isinstance(v, str) and "${" in v:
            def rep(m):
                name = m.group(1)
                if name.startswith("ENV:"):
                    if not env_ok:
                        raise ConfigError(f"'{k}' uses ${{{name}}}: reading the environment needs --allow-env")
                    return os.environ.get(name[4:], "")
                got = _get(root, name)
                if got is None:
                    raise ConfigError(f"'{k}' uses ${{{name}}}, which is not set")
                return str(got)
            for _ in range(10):                   # references to references
                nv = _REF.sub(rep, v)
                if nv == v:
                    break
                v = nv
            data[k] = v


def _rec(v):
    if isinstance(v, dict):
        return Record(**{str(k): _rec(x) for k, x in v.items()})
    if isinstance(v, list):
        return [_rec(x) for x in v]
    return v


class _Sources(dict):
    pass


_WHERE = {}


def load(files, env=None, args=False, defaults=None, required=None, argv=None, read=None, env_ok=True):
    data, sources = {}, {}
    if defaults is not None:
        merge(data, _plain(defaults), "default", sources)
    for f in files:
        f = str(f)
        optional = f.endswith("?")
        f = f[:-1] if optional else f
        path = read(f) if read else f
        if not os.path.exists(str(path)):
            if optional:
                continue
            raise ConfigError(f"settings file '{f}' does not exist (end the name with ? if it is optional)")
        merge(data, read_file(str(path)), os.path.basename(str(path)), sources)
    if env:
        if not env_ok:
            raise ConfigError(f"config.load(env=\"{env}\") reads the environment: run with --allow-env")
        pre = str(env).upper().rstrip("_") + "_"
        over = {}
        for name, v in sorted(os.environ.items()):
            if name.upper().startswith(pre) and len(name) > len(pre):
                _set(over, name[len(pre):].lower().replace("__", "."), _guess(v))
        for k in over:
            merge(data, {k: over[k]}, "environment", sources)
    if args:
        av = list(sys.argv[1:] if argv is None else argv)
        over = {}
        i = 0
        while i < len(av):
            a = av[i]
            if a == "--set" and i + 1 < len(av):
                a = av[i + 1]
                i += 1
            elif a.startswith("--set="):
                a = a[6:]
            if "=" in a and not a.startswith("-") and re.match(r"[A-Za-z_][\w.\-]*=", a):
                k, _, v = a.partition("=")
                _set(over, k, _guess(v))
            i += 1
        merge(data, over, "command line", sources)
    if defaults is not None:
        _check_types(data, _plain(defaults), sources)
    _expand(data, data, env_ok)
    missing = [k for k in (required or []) if _get(data, str(k)) is None]
    if missing:
        raise ConfigError("settings missing: " + ", ".join(missing) +
                          (f" (looked in {', '.join(str(f) for f in files)})" if files else ""))
    rec = _rec(data)
    _WHERE[id(rec)] = (rec, dict(sources))
    return rec


def save(path, cfg):
    data = _plain(cfg)
    low = str(path).lower()
    tmp = str(path) + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(str(path))) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        if low.endswith(".json"):
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        elif low.endswith(".toml"):
            fh.write(to_toml(data))
        elif low.endswith((".yaml", ".yml")):
            import yaml
            yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
        else:
            raise ConfigError(f"{path}: config.save writes .toml, .json or .yaml")
    os.replace(tmp, str(path))
    return str(path)


def _tv(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_tv(x) for x in v) + "]"
    if v is None:
        return '""'
    return json.dumps(str(v), ensure_ascii=False)


def to_toml(data, prefix=""):
    plain = [(k, v) for k, v in data.items() if not isinstance(v, dict)]
    tables = [(k, v) for k, v in data.items() if isinstance(v, dict)]
    out = "".join(f"{_key(k)} = {_tv(v)}\n" for k, v in plain)
    for k, v in tables:
        name = f"{prefix}{_key(k)}"
        body = to_toml(v, name + ".")
        if any(not isinstance(x, dict) for x in v.values()):
            out += f"\n[{name}]\n"
        out += body
    return out


def _key(k):
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k)


def module(interp=None):
    def read(p):
        if interp is None:
            return p
        from .security import check_path
        return check_path(interp, p, "read", "config.load()")

    def env_ok():
        if interp is None:
            return True
        from .security import _policy, ALL
        return _policy(interp).env is ALL

    def load_(*files, env=None, args=False, defaults=None, required=None):
        return load(list(files), env, args, defaults, required, None, read, env_ok())

    def save_(path, cfg):
        if interp is not None:
            from .security import check_path
            path = check_path(interp, path, "write", "config.save()")
        return save(path, cfg)

    def get(cfg, key, default=None):
        v = _get(cfg, str(key), None)
        return default if v is None else v

    def sources(cfg):
        got = _WHERE.get(id(cfg))
        return dict(got[1]) if got and got[0] is cfg else {}

    def parse(text, format="toml"):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix="." + format, delete=False, encoding="utf-8") as fh:
            fh.write(text)
        try:
            return _rec(read_file(fh.name))
        finally:
            os.remove(fh.name)

    return {"load": load_, "save": save_, "get": get, "sources": sources, "parse": parse,
            "to_toml": lambda cfg: to_toml(_plain(cfg)), "ConfigError": ConfigError}
