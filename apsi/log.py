"""`use log` — logging for Apsi programs, to the terminal and to .log files.

    use log
    log.to_file("app.log")                       # also write every record to app.log (rotated at 10 MB, 5 kept)
    log.info("server started", port=8080)
    log.warn("disk almost full", free=3.2[GB])
    let db = log.named("db")
    db.error("query failed", table="users", err=e)
    with log.timer("load model"):
        ...                                       # "load model took 1.42 s"

A record is one line: time, level, name, message, then key=value fields:

    2026-11-11 08:30:12.345  INFO   app  server started  port=8080

`log.to_file(path, format="json")` writes JSON lines instead (one object per line, for other tools). Files are
rotated by size ("10 MB") or by day ("daily"); old ones are kept as app.1.log, app.2.log … or app.2026-11-10.log.
`log.read(path, level=, grep=, last=)` reads a .log file back as records; `apsi log app.log --level warn --follow`
shows one in colour. Writing a file needs --allow-write for its folder, as any other file.
"""
import datetime as _dt
import json
import os
import re
import sys
import threading
import time

LEVELS = {"debug": 10, "info": 20, "warn": 30, "warning": 30, "error": 40, "critical": 50}
NAMES = {10: "DEBUG", 20: "INFO", 30: "WARN", 40: "ERROR", 50: "CRIT"}
COLORS = {10: "\033[2m", 20: "\033[36m", 30: "\033[33m", 40: "\033[31m", 50: "\033[1;31m"}
RESET = "\033[0m"
_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?)\s+(DEBUG|INFO|WARN|ERROR|CRIT)\s+(\S+)\s+(.*)$")


class LogError(Exception):
    pass


def _text(v):
    """A field value as compact text (Apsi's own formatting for numbers, units, lists …)."""
    try:
        from .fmt import fmt
        s = fmt(v)
    except Exception:
        s = str(v)
    if isinstance(v, BaseException):
        s = f"{type(v).__name__}: {v}"
    if not s or any(c in s for c in " \t\"=") or "\n" in s:
        s = json.dumps(s, ensure_ascii=False)
    return s


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, BaseException):
        return f"{type(v).__name__}: {v}"
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    try:
        from .fmt import fmt
        return fmt(v)
    except Exception:
        return str(v)


def _size(text):
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(b|kb|mb|gb)?\s*", str(text).lower())
    if not m:
        raise LogError(f"rotate: give a size such as \"10 MB\", or \"daily\" (not {text!r})")
    return int(float(m.group(1)) * {"b": 1, None: 1, "kb": 1 << 10, "mb": 1 << 20, "gb": 1 << 30}[m.group(2)])


class _File:
    def __init__(self, path, fmt_, rotate, keep, level):
        self.path, self.fmt, self.keep, self.level = path, fmt_, max(0, int(keep)), level
        self.daily = str(rotate).lower() == "daily"
        self.limit = None if self.daily or rotate in (None, False, "none") else _size(rotate)
        self.day = _dt.date.today()
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8", newline="\n")

    def _rotate(self):
        self.fh.close()
        stem, ext = os.path.splitext(self.path)
        if self.daily:
            os.replace(self.path, f"{stem}.{self.day.isoformat()}{ext or '.log'}")
            olds = sorted(p for p in os.listdir(os.path.dirname(os.path.abspath(self.path)))
                          if re.fullmatch(re.escape(os.path.basename(stem)) + r"\.\d{4}-\d\d-\d\d" +
                                          re.escape(ext or ".log"), p))
            for p in olds[:max(0, len(olds) - self.keep)]:
                try:
                    os.remove(os.path.join(os.path.dirname(os.path.abspath(self.path)), p))
                except OSError:
                    pass
        else:
            for i in range(self.keep, 0, -1):          # app.4.log -> app.5.log … app.log -> app.1.log
                src = self.path if i == 1 else f"{stem}.{i - 1}{ext}"
                if os.path.exists(src):
                    os.replace(src, f"{stem}.{i}{ext}")
            if self.keep == 0:
                os.remove(self.path)
        self.day = _dt.date.today()
        self.fh = open(self.path, "a", encoding="utf-8", newline="\n")

    def write(self, line):
        if self.daily and _dt.date.today() != self.day and os.path.getsize(self.path) > 0:
            self._rotate()
        elif self.limit and self.fh.tell() + len(line) > self.limit and self.fh.tell() > 0:
            self._rotate()
        self.fh.write(line)
        self.fh.flush()

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


class _Hub:
    """Where records go: the terminal (stderr) and any files, shared by every named logger."""

    def __init__(self):
        self.lock = threading.Lock()
        self.level = LEVELS.get(os.environ.get("APSI_LOG_LEVEL", "info").lower(), 20)
        self.console = True
        self.files = []
        self.color = None

    def emit(self, lvl, name, msg, fields):
        if lvl < self.level and not any(lvl >= f.level for f in self.files):
            return
        now = _dt.datetime.now()
        stamp = now.strftime("%Y-%m-%d %H:%M:%S.") + f"{now.microsecond // 1000:03d}"
        tail = "".join(f"  {k}={_text(v)}" for k, v in fields.items())
        text = f"{stamp}  {NAMES[lvl]:<5}  {name}  {msg}{tail}"
        with self.lock:
            if self.console and lvl >= self.level:
                out = sys.stderr
                color = self.color if self.color is not None else getattr(out, "isatty", lambda: False)()
                if color:
                    out.write(f"\033[2m{stamp}\033[0m  {COLORS[lvl]}{NAMES[lvl]:<5}{RESET}  \033[1m{name}\033[0m  "
                              f"{msg}\033[2m{tail}\033[0m\n")
                else:
                    out.write(text + "\n")
                out.flush()
            for f in self.files:
                if lvl >= f.level:
                    if f.fmt == "json":
                        rec = {"time": now.isoformat(timespec="milliseconds"), "level": NAMES[lvl].lower(),
                               "name": name, "message": msg}
                        rec.update({k: _jsonable(v) for k, v in fields.items()})
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    else:
                        f.write(text + "\n")


_HUB = _Hub()


class Timer:
    def __init__(self, logger, what, level):
        self.logger, self.what, self.level = logger, what, level

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, kind, err, tb):
        secs = time.perf_counter() - self.t0
        if err is None:
            self.logger._log(self.level, f"{self.what} took {secs:.3g} s", {})
        else:
            self.logger._log(40, f"{self.what} failed after {secs:.3g} s", {"err": err})
        return False


class Logger:
    def __init__(self, name="app", fields=None):
        self.name = name
        self.fields = dict(fields or {})

    def _log(self, lvl, msg, fields):
        f = dict(self.fields)
        f.update(fields)
        _HUB.emit(lvl, self.name, " ".join(_text(m) if not isinstance(m, str) else m for m in msg)
                  if isinstance(msg, tuple) else str(msg), f)

    def debug(self, *msg, **fields):
        self._log(10, msg, fields)

    def info(self, *msg, **fields):
        self._log(20, msg, fields)

    def warn(self, *msg, **fields):
        self._log(30, msg, fields)

    warning = warn

    def error(self, *msg, **fields):
        self._log(40, msg, fields)

    def critical(self, *msg, **fields):
        self._log(50, msg, fields)

    def named(self, name, **fields):
        """A logger for one part of a program: its name on every line, and fields added to each record."""
        f = dict(self.fields)
        f.update(fields)
        return Logger(f"{self.name}.{name}" if self.name != "app" else str(name), f)

    def timer(self, what, level="info"):
        return Timer(self, what, LEVELS.get(str(level).lower(), 20))

    def __repr__(self):
        return f"<logger {self.name}>"


def _rotated(path):
    """The older files of a rotated log, oldest first: app.3.log, app.2.log, app.1.log / app.2026-11-10.log."""
    stem, ext = os.path.splitext(path)
    d = os.path.dirname(os.path.abspath(path))
    base = os.path.basename(stem)
    nums, days = [], []
    for p in os.listdir(d):
        m = re.fullmatch(re.escape(base) + r"\.(\d+)" + re.escape(ext), p)
        if m:
            nums.append((int(m.group(1)), p))
        elif re.fullmatch(re.escape(base) + r"\.\d{4}-\d\d-\d\d" + re.escape(ext or ".log"), p):
            days.append(p)
    return [os.path.join(d, p) for _, p in sorted(nums, reverse=True)] + [os.path.join(d, p) for p in sorted(days)]


def read(path, level=None, grep=None, last=None, name=None, older=False):
    """Records of a .log file (text or JSON lines): [{"time", "level", "name", "message", "fields"}], filtered.
    older=true reads the rotated files too, oldest first."""
    want = LEVELS.get(str(level).lower(), 0) if level else 0
    pat = re.compile(grep, re.I) if grep else None
    out = []
    for p in (_rotated(path) if older else []) + [path]:
        _read_one(p, out, want, pat, name)
    return out[-int(last):] if last else out


def _read_one(path, out, want, pat, name):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            rec = None
            if line.startswith("{"):
                try:
                    j = json.loads(line)
                    rec = {"time": j.pop("time", ""), "level": str(j.pop("level", "info")).upper(),
                           "name": j.pop("name", ""), "message": j.pop("message", ""), "fields": j}
                except ValueError:
                    rec = None
            if rec is None:
                m = _LINE.match(line)
                if m:
                    msg, fields = _split_fields(m.group(4))
                    rec = {"time": m.group(1), "level": m.group(2), "name": m.group(3), "message": msg,
                           "fields": fields}
                elif out:                              # a continuation (a traceback, a long message)
                    out[-1]["message"] += "\n" + line
                    continue
                else:
                    rec = {"time": "", "level": "INFO", "name": "", "message": line, "fields": {}}
            lv = {"DEBUG": 10, "INFO": 20, "WARN": 30, "WARNING": 30, "ERROR": 40, "CRIT": 50,
                  "CRITICAL": 50}.get(rec["level"], 20)
            if lv < want or (name and not str(rec["name"]).startswith(name)):
                continue
            if pat and not pat.search(line):
                continue
            out.append(rec)


def _split_fields(text):
    """'message  a=1  b="x y"' -> ("message", {"a": "1", "b": "x y"})."""
    parts = re.split(r"  (?=[A-Za-z_][\w.]*=)", text)
    fields = {}
    for p in parts[1:]:
        k, _, v = p.partition("=")
        if v.startswith('"'):
            try:
                v = json.loads(v)
            except ValueError:
                pass
        fields[k] = v
    return parts[0], fields


def show(records, color=None, out=None):
    """Print records as aligned, coloured lines (what `apsi log` shows)."""
    out = out or sys.stdout
    color = getattr(out, "isatty", lambda: False)() if color is None else color
    for r in records:
        lv = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40, "CRIT": 50}.get(r["level"], 20)
        tail = "".join(f"  {k}={_text(v)}" for k, v in r["fields"].items())
        if color:
            out.write(f"\033[2m{r['time']}\033[0m  {COLORS[lv]}{r['level']:<5}{RESET}  \033[1m{r['name']}\033[0m  "
                      f"{r['message']}\033[2m{tail}\033[0m\n")
        else:
            out.write(f"{r['time']}  {r['level']:<5}  {r['name']}  {r['message']}{tail}\n")


def module(interp=None):
    root = Logger("app")

    def to_file(path, format="text", rotate="10 MB", keep=5, level="debug"):
        """Also write records to a file (format "text" or "json"; rotate "10 MB" / "daily" / none; keep old ones)."""
        if interp is not None:
            from .security import check_path
            path = check_path(interp, path, "write", "log.to_file()")
        if format not in ("text", "json"):
            raise LogError('log.to_file: format is "text" or "json"')
        f = _File(path, format, rotate, keep, LEVELS.get(str(level).lower(), 10))
        with _HUB.lock:
            _HUB.files.append(f)
        return path

    def close():
        with _HUB.lock:
            for f in _HUB.files:
                f.close()
            _HUB.files.clear()

    def level(name=None):
        """The terminal's level ("debug", "info", "warn", "error"); without an argument, the current one."""
        if name is not None:
            if str(name).lower() not in LEVELS:
                raise LogError(f"log.level: one of {', '.join(k for k in LEVELS if k != 'warning')}")
            _HUB.level = LEVELS[str(name).lower()]
        return NAMES[_HUB.level].lower()

    def console(on=True, color=None):
        _HUB.console = bool(on)
        _HUB.color = color
        return _HUB.console

    def read_(path, level=None, grep=None, last=None, name=None, older=False):
        if interp is not None:
            from .security import check_path
            path = check_path(interp, path, "read", "log.read()")
        return read(path, level, grep, last, name, older)

    return {"debug": root.debug, "info": root.info, "warn": root.warn, "warning": root.warn, "error": root.error,
            "critical": root.critical, "named": root.named, "timer": root.timer, "to_file": to_file,
            "close": close, "level": level, "console": console, "read": read_, "show": show,
            "LogError": LogError}
