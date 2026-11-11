"""`use env` — configuration and secrets from .env files and the environment, without leaking them.

    use env
    env.load(".env")                         # KEY=value lines (quotes, # comments, `export`, ${VAR} expansion)
    let db_url = env.get("DATABASE_URL")     # .env first, then the real environment (needs --allow-env for that)
    let key = env.secret("OPENAI_API_KEY")   # prints as ******; key.reveal() gives the text
    let port = env.int("PORT", 8080)
    let debug = env.bool("DEBUG", false)

`apsi run app.apsi --env-file .env` loads the file before the program starts. Values read from a .env file need
only --allow-read for the file; reading the process environment needs --allow-env (or --allow-env=NAME,…).
`apsi run --venv .venv` (or a .venv folder next to the program, found automatically) makes pyimport() use the
packages installed in that virtual environment.
"""
import os
import re

from .errors import RuntimeErr

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*(.*)\s*$")


def parse(text):
    """Parse .env text into a dict (the same rules as python-dotenv / docker compose)."""
    out = {}
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        if v[:1] in ("'", '"'):
            q = v[0]
            body = v[1:]
            while not _closed(body, q) and i < len(lines):          # multi-line quoted value
                body += "\n" + lines[i]
                i += 1
            end = _close_index(body, q)
            val = body[:end] if end >= 0 else body
            if q == '"':
                val = val.encode("latin-1", "backslashreplace").decode("unicode_escape") if "\\" in val else val
                val = _expand(val, out)
        else:
            val = re.split(r"\s+#", v, 1)[0].strip()
            val = _expand(val, out)
        out[k] = val
    return out


def _close_index(body, q):
    j = 0
    while j < len(body):
        if body[j] == "\\" and q == '"':
            j += 2
            continue
        if body[j] == q:
            return j
        j += 1
    return -1


def _closed(body, q):
    return _close_index(body, q) >= 0


def _expand(val, known):
    def rep(m):
        name = m.group(1) or m.group(2)
        return known.get(name, os.environ.get(name, "") if os.environ.get("APSI_ENV_EXPAND_OS") else "")
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", rep, val)


class Secret:
    """A value that does not show itself: print, str, f-strings and logs give ******."""
    __slots__ = ("_v", "name")

    def __init__(self, value, name=""):
        self._v = value
        self.name = name

    def reveal(self):
        return self._v

    def __str__(self):
        return "******"

    __repr__ = __format__ = lambda self, *a: "******"

    def __len__(self):
        return len(self._v or "")

    def __bool__(self):
        return bool(self._v)

    def __eq__(self, o):
        import hmac
        other = o._v if isinstance(o, Secret) else o
        return isinstance(other, str) and isinstance(self._v, str) and hmac.compare_digest(self._v, other)

    def __hash__(self):
        return hash(("secret", self._v))


class Env:
    def __init__(self, interp=None):
        self.interp = interp
        self.values = {}

    def load(self, path=".env", override=True, export=False):
        """Read a .env file (needs --allow-read for it). Returns the names it defined."""
        if self.interp is not None:
            from .security import check_path
            path = check_path(self.interp, path, "read", "env.load()")
        if not os.path.exists(path):
            raise RuntimeErr(f"env.load(): no file '{path}'")
        with open(path, encoding="utf-8-sig") as fh:
            vals = parse(fh.read())
        for k, v in vals.items():
            if override or k not in self.values:
                self.values[k] = v
            if export:
                os.environ[k] = v
        return list(vals)

    def get(self, name, default=None):
        if name in self.values:
            return self.values[name]
        if self.interp is not None:
            from .security import check_env
            try:
                check_env(self.interp, name)
            except Exception:                                  # noqa: BLE001
                if default is not None:
                    return default
                raise
        return os.environ.get(name, default)

    def _soft(self, name):
        """The value, or None when it is not set (or the environment may not be read)."""
        try:
            return self.get(name, None)
        except Exception:                                      # noqa: BLE001
            return None

    def require(self, *names):
        missing = [n for n in names if self.get(n, "") in ("", None)]
        if missing:
            raise RuntimeErr(f"missing configuration: {', '.join(missing)} (set them in .env or the environment)")
        return [self.get(n) for n in names]

    def secret(self, name, default=None):
        v = self.get(name, None) if default is None else (self._soft(name) or default)
        return None if v is None else Secret(v, name)

    def int(self, name, default=None):
        v = self._soft(name) if default is not None else self.get(name, None)
        return default if v in (None, "") else int(v)

    def float(self, name, default=None):
        v = self._soft(name) if default is not None else self.get(name, None)
        return default if v in (None, "") else float(v)

    def bool(self, name, default=False):
        v = self._soft(name)
        if v in (None, ""):
            return default
        return str(v).strip().lower() in ("1", "true", "yes", "on", "y")

    def list(self, name, sep=",", default=None):
        v = self._soft(name) if default is not None else self.get(name, None)
        return default if v in (None, "") else [x.strip() for x in str(v).split(sep) if x.strip()]

    def has(self, name):
        return name in self.values or name in os.environ


def venv_mismatch(venv):
    """Why the packages of this virtual environment cannot load in the running Python, or None when they can.
    Compiled modules (.pyd / .so: numpy, pyarrow, torch …) are built for one Python version and one build
    (free-threaded "t" or not), so a venv of another version fails with errors like "No module named 'pyarrow.lib'"."""
    import sys
    try:
        with open(os.path.join(venv, "pyvenv.cfg"), encoding="utf-8") as fh:
            cfg = dict(line.split("=", 1) for line in fh.read().splitlines() if "=" in line)
    except OSError:
        return None
    cfg = {k.strip().lower(): v.strip() for k, v in cfg.items()}
    ver = cfg.get("version") or cfg.get("version_info") or ""
    m = re.match(r"(\d+)\.(\d+)", ver)
    if not m:
        return None
    exe = os.path.basename(cfg.get("executable", "") or cfg.get("command", "").split(" ")[0]).lower()
    venv_ft = bool(re.search(r"python3?\.?\d*t(\.exe)?$", exe)) or cfg.get("gil", "").lower() in ("false", "disabled")
    run_ft = not getattr(sys, "_is_gil_enabled", lambda: True)()
    here = f"{sys.version_info.major}.{sys.version_info.minor}"
    there = f"{m.group(1)}.{m.group(2)}"
    if there == here and venv_ft == run_ft:
        return None
    return (f"{venv} is Python {there}{' free-threaded' if venv_ft else ''}, Apsi runs on Python {here}"
            f"{' free-threaded' if run_ft else ''}: its compiled packages cannot load here")


def find_venv_site(venv):
    """site-packages of a virtual environment, or None."""
    import glob
    import sys
    pats = [os.path.join(venv, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"),
            os.path.join(venv, "Lib", "site-packages")]
    for p in pats:
        if os.path.isdir(p):
            return p
    found = glob.glob(os.path.join(venv, "lib", "python3*", "site-packages"))
    return found[0] if found else None


def module(interp=None):
    e = Env(interp)
    if interp is not None:
        e.values.update(getattr(interp, "env_file_values", {}))
    return {"load": e.load, "get": e.get, "secret": e.secret, "int": e.int, "float": e.float, "bool": e.bool,
            "list": e.list, "require": e.require, "has": e.has, "parse": parse, "Secret": Secret, "values": e.values}
