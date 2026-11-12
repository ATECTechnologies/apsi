"""Interactive Apsi shell.

Everything typed at the ``apsi>`` prompt is Apsi code, except:

* ``:commands``   - REPL commands (``:help`` lists them all)
* shell words     - ``cd``, ``pwd``, ``ls``/``dir``, ``clear``/``cls``, ``run FILE``,
                    ``exit``/``quit`` work like a terminal when the word is not an Apsi variable
* ``!command``    - passed to the operating-system shell (``!git status``, ``!python -V``)

Multi-line input is detected automatically: open brackets, a trailing ``:`` or ``\\``,
an open triple-quoted string, or a following ``|>`` line keep the prompt in ``...`` mode.
A block that started with ``:`` ends with an empty line.
"""
import glob
import os
import shlex
import subprocess
import sys
import re
import time

from . import __version__
from .errors import ApsiError, CheckFailed

_APSI_TAIL = re.compile(r"^(\(|\[|==|!=|\|>|=|[+\-*/%@]=|\.[A-Za-z_]|[+*%@<>^])")
SHELL_WORDS = {"cd", "pwd", "ls", "dir", "clear", "cls", "run", "exit", "quit", "help"}


# ---------------------------------------------------------------------------- terminal
class Style:
    """ANSI colours, disabled automatically when not on a terminal or when NO_COLOR is set."""

    def __init__(self, enabled):
        self.on = enabled

    def _c(self, code, s):
        return f"\x1b[{code}m{s}\x1b[0m" if self.on else s

    def prompt(self, s, where=None):
        """The prompt; `where` (the current folder) is shown before it, like a shell: C:\\work apsi> """
        if not self.on:
            return f"{where} {s}" if where else s
        # \001/\002 tell GNU readline the escape codes take no width
        rl = "readline" in sys.modules and os.name != "nt"
        a, b = ("\001", "\002") if rl else ("", "")
        head = f"{a}\x1b[38;5;75m{b}{where}{a}\x1b[0m{b} " if where else ""
        return f"{head}{a}\x1b[38;5;43m\x1b[1m{b}{s}{a}\x1b[0m{b}"

    def err(self, s):
        return self._c("31", s)

    def warn(self, s):
        return self._c("33", s)

    def dim(self, s):
        return self._c("2", s)

    def ok(self, s):
        return self._c("32", s)

    def accent(self, s):
        return self._c("36", s)

    def bold(self, s):
        return self._c("1", s)


def _enable_vt():
    """Turn on ANSI escape handling in the Windows 10/11 console."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not k.GetConsoleMode(h, ctypes.byref(mode)):
            return False
        return bool(k.SetConsoleMode(h, mode.value | 0x0004))
    except Exception:
        return False


def color_supported(stream=None):
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR") or os.environ.get("APSI_NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return _enable_vt()


# ---------------------------------------------------------------------------- input analysis
def needs_more(src):
    """True when ``src`` is an incomplete statement (open bracket/string, trailing ':' or '\\')."""
    depth = 0
    i, n = 0, len(src)
    in_str = None
    while i < n:
        ch = src[i]
        if in_str:
            if in_str == '"""':
                if src.startswith('"""', i):
                    in_str = None
                    i += 3
                    continue
            elif ch == "\\":
                i += 2
                continue
            elif ch == '"':
                in_str = None
            elif ch == "\n":
                in_str = None            # unterminated plain string: let the lexer report it
            i += 1
            continue
        if ch == "#":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if src.startswith('"""', i):
            in_str = '"""'
            i += 3
            continue
        if ch == '"':
            in_str = '"'
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        i += 1
    if in_str == '"""' or depth > 0:
        return True
    last = src.rstrip().splitlines()[-1] if src.strip() else ""
    code = last.split("#", 1)[0].rstrip() if '"' not in last else last.rstrip()
    if code.endswith(":") or code.endswith("\\"):
        return True
    # a dangling operator means the expression continues on the next line
    if code.endswith(("=", ",", "|>", "->", "=>", "+", "-", "*", "/", "@", "**", " and", " or", " not")):
        return True
    return False


def _shell_path(arg):
    """Accept Windows and POSIX paths, with or without quotes, ~ and %VAR%/$VAR."""
    arg = arg.strip()
    if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in "\"'":
        arg = arg[1:-1]
    return os.path.expandvars(os.path.expanduser(arg))


def _split(rest):
    try:
        return shlex.split(rest, posix=(os.name != "nt"))
    except ValueError:
        return rest.split()


# ---------------------------------------------------------------------------- the REPL
class Repl:
    def __init__(self, args, make_interp, stdin=None, stdout=None):
        self.args = args
        self.make_interp = make_interp
        self.out = stdout or sys.stdout
        self.inp = stdin
        self.style = Style(color_supported(self.out) and stdin is None)
        self.history = []            # complete inputs, in order
        self.source_log = []         # inputs that ran without error (for :save)
        self.prev_dir = None
        self.show_time = False
        self.engine = "interp"
        self._readline = None
        self.reset(quiet=True)

    # ------------------------------------------------------------------ session
    def reset(self, quiet=False):
        from .interpreter import Env
        self.it = self.make_interp(self.args)
        self.it.repl_echo = True
        self.env = Env(self.it.globals)
        self.source_log = []
        if not quiet:
            self.print(self.style.dim("session reset: all variables and functions cleared"))

    def print(self, *a, **k):
        k.setdefault("file", self.out)
        print(*a, **k)

    def error(self, msg):
        self.print(self.style.err(msg))

    # ------------------------------------------------------------------ readline
    def _setup_readline(self):
        if self.inp is not None:
            return
        try:
            import readline
        except ImportError:
            try:
                import pyreadline3 as readline  # noqa: F401  (optional on Windows)
            except ImportError:
                return
        self._readline = readline
        try:
            from .security import config_dir
            self.hist_file = os.path.join(config_dir(), "repl_history")
            os.makedirs(config_dir(), exist_ok=True)
            if os.path.exists(self.hist_file):
                readline.read_history_file(self.hist_file)
            readline.set_history_length(5000)
        except Exception:
            self.hist_file = None
        try:
            readline.set_completer(self._complete)
            readline.set_completer_delims(" \t\n()[]{},+-*/%=<>!|&;:'\"")
            if "libedit" in (getattr(readline, "__doc__", "") or ""):
                readline.parse_and_bind("bind ^I rl_complete")
            else:
                readline.parse_and_bind("tab: complete")
        except Exception:
            pass

    def _save_history(self):
        if self._readline and getattr(self, "hist_file", None):
            try:
                self._readline.write_history_file(self.hist_file)
            except Exception:
                pass

    def names(self):
        from .lexer import KEYWORDS
        out = set(KEYWORDS) | {"test"}
        e = self.env
        while e is not None:
            out.update(k for k in e.vars if not k.startswith("__"))
            e = e.parent
        return out

    def _complete(self, text, state):
        if state == 0:
            line = self._readline.get_line_buffer() if self._readline else text
            stripped = line.lstrip()
            first = stripped.split(" ", 1)[0]
            if stripped.startswith(":") and " " not in stripped:
                cands = [":" + c for c in COMMANDS if (":" + c).startswith(text)]
            elif first in ("cd", "ls", "dir", "run", ":cd", ":ls", ":load", ":run", ":check", ":test",
                           ":fmt", ":save", ":edit") and " " in stripped:
                cands = [p + (os.sep if os.path.isdir(p) else "")
                         for p in glob.glob(os.path.expanduser(text) + "*")]
            elif "." in text:
                head, _, tail = text.rpartition(".")
                cands = []
                try:
                    obj = self.env.lookup(head.split(".")[0])
                    for part in head.split(".")[1:]:
                        obj = getattr(obj, part)
                    attrs = set(getattr(obj, "vars", {}) or {}) | set(getattr(obj, "fields", {}) or {}) \
                        | {a for a in dir(obj) if not a.startswith("_")}
                    cands = [f"{head}.{a}" for a in sorted(attrs) if a.startswith(tail)]
                except Exception:
                    pass
            else:
                cands = sorted(n for n in self.names() if n.startswith(text))
            self._cands = cands
        try:
            return self._cands[state]
        except IndexError:
            return None

    # ------------------------------------------------------------------ reading input
    def _input(self, prompt):
        if self.inp is not None:
            line = self.inp.readline()
            if not line:
                raise EOFError
            return line.rstrip("\n").rstrip("\r")
        return input(prompt)

    def read_statement(self):
        """Read one complete input (possibly many lines). Returns None at end of input."""
        where = _prompt_folder()
        p1 = self.style.prompt("apsi> ", where)
        p2 = self.style.prompt(" " * (len(where) + 1 if where else 0) + "...   ")
        lines = []
        block = False
        while True:
            try:
                line = self._input(p2 if lines else p1)
            except EOFError:
                if lines:
                    return "\n".join(lines)
                return None
            except KeyboardInterrupt:
                self.print("\n" + self.style.dim("(input cancelled — Ctrl+D or :quit to exit)")
                           if lines else "\n" + self.style.dim("(Ctrl+D or :quit to exit)"))
                lines, block = [], False
                continue
            if not lines:
                s = line.strip()
                if not s:
                    continue
                if s.startswith(":") or s.startswith("!") or self._is_shell(s):
                    return line
            if lines and block and not line.strip():
                return "\n".join(lines)
            if lines and not block and _dangling(lines[-1]) and not _open_brackets("\n".join(lines)):
                # `let x = 1 +` ⏎ `2`: the language has no line breaks after an operator outside brackets,
                # so join the continuation onto the same logical line
                lines[-1] = lines[-1].rstrip().rstrip("\\").rstrip() + " " + line.strip()
            else:
                lines.append(line)
            src = "\n".join(lines)
            if needs_more(src):
                if src.rstrip().endswith(":"):
                    block = True
                continue
            if block:
                continue
            return src

    def _is_shell(self, s):
        word = s.split(None, 1)[0]
        if word not in SHELL_WORDS:
            return False
        rest = s[len(word):].lstrip()
        # an Apsi expression such as `run(x)`, `dir = 3`, `ls.size`, `cd + 1` stays Apsi code
        if _APSI_TAIL.match(rest):
            return False
        return word not in self.env.vars     # a session variable with this name stays Apsi code

    # ------------------------------------------------------------------ main loop
    def banner(self):
        s = self.style
        self.print(s.bold(s.accent(f"Apsi {__version__}")) + s.dim(
            f"  ·  Python {sys.version.split()[0]}  ·  {sys.platform}"))
        self.print(s.dim("Type Apsi code, :help for commands, cd/ls/run/!cmd for the shell, :quit to exit."))

    def loop(self):
        self._setup_readline()
        if self.inp is None:
            self.banner()
        try:
            while True:
                src = self.read_statement()
                if src is None:
                    if self.inp is None:
                        self.print()
                    return 0
                self.history.append(src)
                try:
                    rc = self.dispatch(src)
                except KeyboardInterrupt:
                    self.print(self.style.warn("KeyboardInterrupt"))
                    continue
                except SystemExit as e:
                    return e.code if isinstance(e.code, int) else 0
                if rc == "quit":
                    return 0
        finally:
            self._save_history()

    def dispatch(self, src):
        s = src.strip()
        if s.startswith(":"):
            name, _, rest = s[1:].partition(" ")
            return self.command(name.strip(), rest.strip())
        if s.startswith("!"):
            return self.shell(s[1:].strip())
        if self._is_shell(s):
            word, _, rest = s.partition(" ")
            word = {"dir": "ls", "cls": "clear", "exit": "quit"}.get(word, word)
            return self.command(word, rest.strip())
        self.run_code(src)

    def run_code(self, src, filename="<repl>"):
        t0 = time.perf_counter()
        from . import runner
        try:
            runner.call(self.it.run, src, filename, check=False, env=self.env, engine="interp")
            self.source_log.append(src)
        except ApsiError as e:
            self.error(e.format(src, filename))
            hint = _hint(e)
            if hint:
                self.print(self.style.dim("hint: " + hint))
        except CheckFailed as e:
            for x in e.errors:
                self.error(x.format(src, filename))
        except RecursionError:
            self.error("RuntimeError: maximum recursion depth exceeded")
        except KeyboardInterrupt:
            self.print(self.style.warn("KeyboardInterrupt"))
        except SystemExit as e:
            self.print(self.style.dim(f"(exit({e.code}) ignored in the REPL — use :quit to leave)"))
        finally:
            sys.stdout.flush()
        if self.show_time:
            self.print(self.style.dim(f"({_dur(time.perf_counter() - t0)})"))

    # ------------------------------------------------------------------ commands
    def command(self, name, rest):
        fn = COMMANDS.get(name) or COMMANDS.get(ALIASES.get(name, ""))
        if fn is None:
            close = [c for c in COMMANDS if c.startswith(name[:2])] if name else []
            self.error(f"unknown command ':{name}'" + (f" — did you mean {', '.join(':' + c for c in close)}?"
                                                      if close else " — type :help"))
            return
        return fn[0](self, rest)

    def shell(self, cmd):
        if not cmd:
            self.error("usage: !COMMAND   e.g. !git status")
            return
        try:
            r = subprocess.run(cmd, shell=True)
            if r.returncode:
                self.print(self.style.dim(f"(exit status {r.returncode})"))
        except OSError as e:
            self.error(f"cannot run '{cmd}': {e}")

    def c_help(self, rest):
        s = self.style
        if rest:
            return self.c_doc(rest)
        self.print(s.bold("REPL commands") + s.dim("  (prefix with ':'; shell words also work without it)"))
        for name, (_, args, desc) in COMMANDS.items():
            self.print(f"  {s.accent((':' + name + ' ' + args).ljust(22))} {desc}")
        self.print(s.bold("\nShell"))
        self.print(f"  {s.accent('cd PATH'.ljust(22))} change folder — Windows paths like C:\\Users\\me work as typed")
        self.print(f"  {s.accent('ls / dir / pwd'.ljust(22))} list / print the current folder")
        self.print(f"  {s.accent('run FILE.apsi'.ljust(22))} run a program in a fresh interpreter")
        self.print(f"  {s.accent('!COMMAND'.ljust(22))} run any operating-system command")
        self.print(s.bold("\nEditing"))
        self.print("  Blocks ending in ':' continue until an empty line; open brackets, '\\' and '|>' continue too.")
        self.print("  Tab completes names, fields and paths; Up/Down recall history; Ctrl+C cancels the input.")
        self.print("  The last result is available as _ .")

    def c_quit(self, rest):
        return "quit"

    def c_clear(self, rest):
        if self.style.on:
            self.out.write("\x1b[2J\x1b[H")
            self.out.flush()
        else:
            os.system("cls" if os.name == "nt" else "clear")

    def c_reset(self, rest):
        self.reset()

    def c_cd(self, rest):
        target = _shell_path(rest) if rest else os.path.expanduser("~")
        if rest.strip() == "-":
            if not self.prev_dir:
                self.error("cd: no previous folder")
                return
            target = self.prev_dir
        if os.name == "nt" and len(target) == 2 and target[1] == ":":
            target += "\\"
        try:
            cur = os.getcwd()
            os.chdir(target)
            self.prev_dir = cur                        # the prompt now shows the new folder, as in a shell
        except FileNotFoundError:
            self.error(f"cd: no such folder: {target}")
        except NotADirectoryError:
            self.error(f"cd: not a folder: {target}")
        except PermissionError:
            self.error(f"cd: permission denied: {target}")

    def c_pwd(self, rest):
        self.print(os.getcwd())

    def c_ls(self, rest):
        """A tidy listing: folders first, then files, with modification time and size, and a summary line."""
        import time as _time
        args = [a for a in _split(rest) if not (a.startswith("-") and not os.path.exists(a))]
        target = _shell_path(args[0]) if args else "."
        s = self.style
        if os.path.isfile(target):
            st = os.stat(target)
            self.print(f"  {_time.strftime('%Y-%m-%d %H:%M', _time.localtime(st.st_mtime))}  "
                       f"{_size(st.st_size):>9}  {target}")
            return
        try:
            entries = sorted(os.scandir(target), key=lambda e: (not _is_dir(e), e.name.lower()))
        except OSError as e:
            self.error(f"ls: {e.strerror}: {target}")
            return
        show_all = "-a" in _split(rest)
        rows = []
        n_dirs = n_files = total = hidden = 0
        for e in entries:
            if e.name.startswith(".") and not show_all:
                hidden += 1
                continue
            try:
                st = e.stat()
                when = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(st.st_mtime))
            except OSError:
                st, when = None, "?"
            if _is_dir(e):
                n_dirs += 1
                rows.append((when, "<dir>", s.accent(e.name + os.sep)))
            else:
                n_files += 1
                size = st.st_size if st else 0
                total += size
                name = s.ok(e.name) if e.name.endswith(".apsi") else e.name
                rows.append((when, _size(size), name))
        self.print(s.dim(f"  Folder: {os.path.abspath(target)}"))
        if not rows:
            self.print(s.dim("  (empty)" + (f" — {hidden} hidden, ls -a shows them" if hidden else "")))
            return
        self.print("")
        self.print(s.dim(f"  {'Modified':<16}  {'Size':>9}  Name"))
        self.print(s.dim(f"  {'-' * 16}  {'-' * 9}  {'-' * 4}"))
        for when, size, name in rows:
            self.print(f"  {when:<16}  {size:>9}  {name}")
        parts = [f"{n_dirs} folder{'s' if n_dirs != 1 else ''}", f"{n_files} file{'s' if n_files != 1 else ''}"]
        summary = ", ".join(parts) + (f" ({_size(total)})" if n_files else "")
        if hidden:
            summary += f", {hidden} hidden (ls -a)"
        self.print("")
        self.print(s.dim("  " + summary))

    def c_run(self, rest):
        args = _split(rest)
        if not args:
            self.error("usage: run FILE.apsi")
            return
        path = _shell_path(args[0])
        if not os.path.exists(path) and os.path.exists(path + ".apsi"):
            path += ".apsi"
        try:
            with open(path, encoding="utf-8-sig") as fh:
                src = fh.read()
        except OSError as e:
            self.error(f"run: cannot open '{path}': {e.strerror}")
            return
        it = self.make_interp(self.args, path)
        t0 = time.perf_counter()
        from . import runner
        try:
            runner.call(it.run, src, path)
        except (ApsiError, CheckFailed) as e:
            from .cli import _report
            _report(e, src, path)
        except KeyboardInterrupt:
            self.print(self.style.warn("KeyboardInterrupt"))
        except SystemExit as e:
            self.print(self.style.dim(f"(program called exit({e.code}))"))
        self.print(self.style.dim(f"({path} finished in {_dur(time.perf_counter() - t0)})"))

    def c_load(self, rest):
        args = _split(rest)
        if not args:
            self.error("usage: :load FILE.apsi   (runs the file inside this session)")
            return
        path = _shell_path(args[0])
        if not os.path.exists(path) and os.path.exists(path + ".apsi"):
            path += ".apsi"
        try:
            with open(path, encoding="utf-8-sig") as fh:
                src = fh.read()
        except OSError as e:
            self.error(f":load: cannot open '{path}': {e.strerror}")
            return
        self.last_loaded = path
        echo, self.it.repl_echo = self.it.repl_echo, False
        try:
            self.run_code(src, path)
        finally:
            self.it.repl_echo = echo
        self.print(self.style.dim(f"loaded {path}"))

    def c_reload(self, rest):
        path = getattr(self, "last_loaded", None)
        if not path:
            self.error(":reload: nothing loaded yet — use :load FILE")
            return
        self.c_load(path)

    def c_vars(self, rest):
        from .fmt import fmt, type_name
        vs = self.env.vars
        if not vs:
            self.print(self.style.dim("(no variables yet)"))
            return
        w = max(len(k) for k in vs)
        for k in sorted(vs):
            if k.startswith("__"):
                continue
            v = vs[k]
            kind = "let" if k in self.env.consts else "var"
            try:
                shown = fmt(v)
            except Exception:
                shown = repr(v)
            shown = shown if len(shown) < 70 else shown[:67] + "..."
            self.print(f"  {self.style.dim(kind)} {self.style.accent(k.ljust(w))} : "
                       f"{type_name(v):<12} = {shown}")

    def c_type(self, rest):
        from .fmt import type_name
        v = self._eval(rest)
        if v is not _NOTHING:
            self.print(self.style.accent(type_name(v)))

    def c_time(self, rest):
        if not rest:
            self.show_time = not self.show_time
            self.print(self.style.dim(f"timing {'on' if self.show_time else 'off'}"))
            return
        t0 = time.perf_counter()
        self.run_code(rest)
        self.print(self.style.dim(f"({_dur(time.perf_counter() - t0)})"))

    def c_bench(self, rest):
        n, _, code = rest.partition(" ")
        if not n.isdigit():
            n, code = "0", rest
        if not code:
            self.error("usage: :bench [N] EXPRESSION")
            return
        echo, self.it.repl_echo = self.it.repl_echo, False
        try:
            times = []
            reps = int(n) or 0
            t_end = time.perf_counter() + 1.0
            i = 0
            while (reps and i < reps) or (not reps and (time.perf_counter() < t_end or i < 3) and i < 100000):
                t0 = time.perf_counter()
                self.it.run(code, "<bench>", check=False, env=self.env, engine="interp")
                times.append(time.perf_counter() - t0)
                i += 1
        except ApsiError as e:
            self.error(e.format(code, "<bench>"))
            return
        finally:
            self.it.repl_echo = echo
        times.sort()
        mean = sum(times) / len(times)
        self.print(f"{len(times)} runs: best {_dur(times[0])}, median {_dur(times[len(times) // 2])}, "
                   f"mean {_dur(mean)}")

    def c_doc(self, rest):
        from .fmt import type_name
        name = rest.strip()
        if not name:
            self.error("usage: :doc NAME")
            return
        v = self._eval(name, quiet=True)
        if v is _NOTHING:
            from .lexer import KEYWORDS
            if name in KEYWORDS or name == "test":
                self.print(f"{self.style.accent(name)} is a keyword — see docs/LANGUAGE.md")
            else:
                self.error(f"'{name}' is not defined")
            return
        self.print(f"{self.style.accent(name)} : {type_name(v)}")
        sig = _signature(v)
        if sig:
            self.print(f"  {sig}")
        doc = _docstring(v)
        if doc:
            for line in doc.strip().splitlines():
                self.print("  " + line.strip())

    def c_ast(self, rest):
        from .parser import parse
        from . import ast as A
        try:
            self.print(A.dump(parse(rest).body))
        except ApsiError as e:
            self.error(e.format(rest, "<repl>"))

    def c_tokens(self, rest):
        from .lexer import tokenize
        try:
            for t in tokenize(rest):
                self.print(f"  {t}")
        except ApsiError as e:
            self.error(e.format(rest, "<repl>"))

    def c_check(self, rest):
        from .cli import cmd_check
        files = [_shell_path(a) for a in _split(rest)]
        if not files:
            self.error("usage: :check FILE...")
            return
        cmd_check(_NS(files=files))

    def c_test(self, rest):
        from .cli import cmd_test
        paths = [_shell_path(a) for a in _split(rest)]
        ns = _NS(**vars(self.args))
        ns.paths = paths
        cmd_test(ns)

    def c_fmt(self, rest):
        from .cli import cmd_fmt
        files = [_shell_path(a) for a in _split(rest)]
        if not files:
            self.error("usage: :fmt FILE...")
            return
        cmd_fmt(_NS(files=files, check=False))

    def c_save(self, rest):
        path = _shell_path(rest) if rest else "session.apsi"
        if not path.endswith(".apsi"):
            path += ".apsi"
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n\n".join(self.source_log) + "\n")
        except OSError as e:
            self.error(f":save: {e.strerror}: {path}")
            return
        self.print(self.style.ok(f"saved {len(self.source_log)} inputs to {path}"))

    def c_history(self, rest):
        n = int(rest) if rest.isdigit() else 20
        start = max(0, len(self.history) - n - 1)
        for i, h in enumerate(self.history[start:-1], start + 1):
            first, *more = h.splitlines()
            self.print(f"  {self.style.dim(str(i).rjust(4))}  {first}" + (self.style.dim(" …") if more else ""))

    def c_edit(self, rest):
        import tempfile
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or ("notepad" if os.name == "nt" else "nano")
        path = _shell_path(rest) if rest else None
        tmp = None
        if path is None:
            fd, tmp = tempfile.mkstemp(suffix=".apsi")
            os.close(fd)
            path = tmp
        try:
            subprocess.call(f'{editor} "{path}"', shell=True)
            with open(path, encoding="utf-8-sig") as fh:
                src = fh.read()
        except OSError as e:
            self.error(f":edit: {e}")
            return
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        if src.strip():
            self.print(self.style.dim(src.rstrip()))
            self.run_code(src)

    def c_paste(self, rest):
        self.print(self.style.dim("paste mode: enter code, finish with a line containing only :end"))
        lines = []
        while True:
            try:
                line = self._input("")
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip() == ":end":
                break
            lines.append(line)
        if lines:
            self.run_code("\n".join(lines))

    def c_perms(self, rest):
        for k, v in self.it.policy.describe().items():
            self.print(f"  {k:7}: {v}")

    def c_env(self, rest):
        self.print(f"  apsi      {__version__}")
        self.print(f"  python    {sys.version.split()[0]}  ({sys.executable})")
        self.print(f"  platform  {sys.platform}")
        self.print(f"  folder    {os.getcwd()}")
        try:
            import numpy
            self.print(f"  numpy     {numpy.__version__}")
        except ImportError:
            pass

    def _eval(self, expr, quiet=False):
        from .parser import parse
        from . import ast as A
        try:
            prog = parse(expr)
            if len(prog.body) != 1 or not isinstance(prog.body[0], A.ExprStmt):
                raise ApsiError("expected a single expression")
            return self.it.eval(prog.body[0].expr, self.env)
        except ApsiError as e:
            if not quiet:
                self.error(e.format(expr, "<repl>"))
            return _NOTHING
        except Exception as e:
            if not quiet:
                self.error(f"{type(e).__name__}: {e}")
            return _NOTHING


_NOTHING = object()


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


COMMANDS = {
    "help": (Repl.c_help, "[NAME]", "show this help, or documentation for NAME"),
    "quit": (Repl.c_quit, "", "leave the REPL (also :q, exit, Ctrl+D)"),
    "vars": (Repl.c_vars, "", "list variables and functions defined in this session"),
    "type": (Repl.c_type, "EXPR", "show the type of an expression"),
    "doc": (Repl.c_doc, "NAME", "signature and documentation of a function or value"),
    "time": (Repl.c_time, "[CODE]", "time one input, or toggle timing for every input"),
    "bench": (Repl.c_bench, "[N] CODE", "benchmark code (N runs, or ~1 second)"),
    "load": (Repl.c_load, "FILE", "run a file inside this session (keeps its definitions)"),
    "reload": (Repl.c_reload, "", "load the last :load file again"),
    "run": (Repl.c_run, "FILE", "run a file in a fresh interpreter"),
    "check": (Repl.c_check, "FILE...", "static check without running"),
    "test": (Repl.c_test, "[PATH...]", "run test blocks"),
    "fmt": (Repl.c_fmt, "FILE...", "format files in place"),
    "edit": (Repl.c_edit, "[FILE]", "open $EDITOR (or Notepad) and run the result"),
    "paste": (Repl.c_paste, "", "paste a large block; finish with :end"),
    "save": (Repl.c_save, "[FILE]", "save every successful input of this session to a .apsi file"),
    "history": (Repl.c_history, "[N]", "show the last N inputs"),
    "ast": (Repl.c_ast, "CODE", "print the syntax tree"),
    "tokens": (Repl.c_tokens, "CODE", "print the tokens"),
    "reset": (Repl.c_reset, "", "clear all variables and functions"),
    "perms": (Repl.c_perms, "", "show the permissions this session runs with"),
    "env": (Repl.c_env, "", "versions, interpreter and current folder"),
    "cd": (Repl.c_cd, "[PATH|-]", "change the current folder"),
    "pwd": (Repl.c_pwd, "", "print the current folder"),
    "ls": (Repl.c_ls, "[PATH]", "list a folder"),
    "clear": (Repl.c_clear, "", "clear the screen"),
}
ALIASES = {"q": "quit", "exit": "quit", "h": "help", "?": "help", "t": "type", "l": "load", "r": "reload",
           "dir": "ls", "cls": "clear", "v": "vars", "hist": "history"}


_DANGLING = ("+", "-", "*", "/", "%", "@", "^", "=", "<", ">", "|>", "\\", "and", "or", "not", "±", "..")


def _dangling(line):
    """Does this line end in the middle of an expression (an operator or a backslash)?"""
    t = line.split("#", 1)[0].rstrip() if '"' not in line and "'" not in line else line.rstrip()
    if not t or t.endswith((":", "->")):
        return False
    if t.endswith(("and", "or", "not")):
        word = t.rsplit(None, 1)[-1]
        return word in ("and", "or", "not")
    return t.endswith(_DANGLING)


def _open_brackets(src):
    depth = 0
    quote = None
    for ch in src:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            continue
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
    return depth > 0


def _is_dir(e):
    try:
        return e.is_dir()
    except OSError:
        return False


def _prompt_folder(limit=48):
    """The current folder for the prompt: ~ for home on Linux/macOS, long paths shortened in the middle.

    APSI_PROMPT=short shows only `apsi>`."""
    if os.environ.get("APSI_PROMPT", "").strip().lower() in ("short", "plain", "0", "off"):
        return ""
    try:
        cwd = os.getcwd()
    except OSError:
        return ""
    home = os.path.expanduser("~")
    if os.name != "nt" and (cwd == home or cwd.startswith(home + os.sep)):
        cwd = "~" + cwd[len(home):]
    if len(cwd) <= limit:
        return cwd
    sep = os.sep
    parts = cwd.split(sep)
    head = parts[0] + sep if parts[0] else sep
    tail = []
    for p in reversed(parts[1:]):
        if len(head) + 4 + len(sep.join([p] + tail)) > limit and tail:
            break
        tail.insert(0, p)
    return head + "..." + sep + sep.join(tail)


def _size(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def _dur(s):
    if s < 1e-3:
        return f"{s * 1e6:.1f} µs"
    if s < 1:
        return f"{s * 1e3:.2f} ms"
    return f"{s:.3f} s"


def _signature(v):
    import inspect
    p = getattr(v, "params", None)
    if p is not None and hasattr(v, "name"):
        try:
            names = [getattr(x, "name", str(x)) for x in p]
            return f"fn {v.name}({', '.join(names)})"
        except Exception:
            return None
    if callable(v):
        try:
            return f"{getattr(v, '__name__', 'fn')}{inspect.signature(v)}"
        except (TypeError, ValueError):
            return None
    return None


def _docstring(v):
    d = getattr(v, "doc", None)
    if isinstance(d, str):
        return d
    if callable(v) and not isinstance(v, type):
        d = getattr(v, "__doc__", None)
        if d and not d.startswith(("Built-in", "Initialize self", "Return self")):
            return d
    return None


def _hint(e):
    m = getattr(e, "msg", "") or ""
    if e.kind == "SyntaxError" and "unexpected character '\\\\'" in m:
        return "shell paths work with shell words: cd C:\\path, ls, run file.apsi — or prefix any OS command with !"
    if e.kind == "NameError" and "not defined" in m:
        return "declare names with let (constant) or var (mutable); :vars lists what exists"
    return None


def run_repl(args, make_interp):
    return Repl(args, make_interp).loop()
