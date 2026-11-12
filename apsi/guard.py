"""Apsi Guard — malware defence for Apsi programs and the files they bring with them.

Three layers, each usable on its own:

1. Static threat scan (`apsi scan`, and automatically before `apsi run`)
   * Trojan Source: bidirectional-override and invisible characters (CVE-2021-42574 / 42694)
   * homoglyph identifiers that mix scripts (Latin "a" vs Cyrillic "а")
   * credential theft: reads of SSH keys, cloud credentials, browser profiles, wallets
   * exfiltration: credential/secret reads in a program that also talks to the network
   * persistence: Startup folder, Run keys, crontab, systemd/LaunchAgents, shell rc files
   * download-and-execute, sandbox escape (subprocess/os/ctypes/... via pyimport),
     dynamic (computed) imports and native library names
   * obfuscation: long base64/hex blobs, high-entropy strings
   * ransomware shape: directory walking + read + write + encryption in one program
   * crypto-miners (stratum pools, xmrig), exfiltration endpoints (Discord/Telegram
     webhooks, paste sites, .onion), hard-coded IP addresses
   * companion files: .py with exec/eval of decoded data, native libraries (PE/ELF)
     checked against the block list, EICAR test signature, optional YARA rules
     and the system antivirus (Microsoft Defender or ClamAV) with --av

2. Behaviour monitor (runtime): stops a program that overwrites or deletes many existing
   files in a short time (ransomware), or touches decoy files.

3. Import and native-code control: even with --allow-python, modules that escape the
   sandbox (subprocess, os, ctypes, socket, ...) need --allow-python=all; native libraries
   are refused from world-writable/temporary folders and can be pinned by sha256.
"""
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import unicodedata

SEVERITY = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}

# ------------------------------------------------------------------ indicators
BIDI = {"\u202a", "\u202b", "\u202c", "\u202d", "\u202e", "\u2066", "\u2067", "\u2068", "\u2069",
        "\u200e", "\u200f", "\u061c"}
INVISIBLE = {"\u200b", "\u200c", "\u200d", "\u2060", "\u2061", "\u2062", "\u2063", "\u2064", "\ufeff",
             "\u00ad", "\u180e", "\u3164", "\u115f", "\u1160"}

# non-Latin letters that are visually identical to Latin ones (Unicode confusables, common subset)
CONFUSABLE = set("аеорсухіјѕԁԛԝһӏғвкмнтАВЕКМНОРСТХУІЈЅԌ"      # Cyrillic
                 "ΑΒΕΖΗΙΚΜΝΟΡΤΥΧοκνι"                            # Greek capitals + omicron, kappa, nu, iota
                 "ՕՍոսց")                                       # Armenian

SECRET_HINTS = [
    r"\.ssh", r"id_rsa", r"id_ed25519", r"id_ecdsa", r"authorized_keys", r"known_hosts",
    r"\.aws", r"credentials", r"\.azure", r"\.kube", r"kubeconfig", r"\.docker[/\\]config",
    r"\.netrc", r"\.pgpass", r"\.git-credentials", r"\.npmrc", r"\.pypirc", r"gcloud",
    r"Login Data", r"Cookies", r"Local State", r"Web Data", r"key[34]\.db", r"logins\.json",
    r"cookies\.sqlite", r"wallet\.dat", r"[/\\]Exodus", r"[/\\]Electrum", r"keystore", r"MetaMask",
    r"\.gnupg", r"secring", r"/etc/shadow", r"/etc/passwd", r"SAM$", r"NTDS\.dit", r"lsass",
    r"\.env$", r"Telegram Desktop", r"[/\\]tdata\b", r"discord[/\\]Local Storage", r"FileZilla",
]
PERSISTENCE_HINTS = [
    r"Start Menu[/\\]Programs[/\\]Startup", r"CurrentVersion[/\\]Run", r"CurrentVersion\\\\Run",
    r"Winlogon", r"Image File Execution Options", r"schtasks", r"crontab", r"/etc/cron",
    r"\.config/autostart", r"systemd/user", r"/etc/systemd", r"LaunchAgents", r"LaunchDaemons",
    r"\.bashrc", r"\.bash_profile", r"\.zshrc", r"\.profile$", r"rc\.local", r"/etc/ld\.so\.preload",
    r"LD_PRELOAD", r"AppInit_DLLs",
]
NET_BAD = [
    (r"stratum\+(tcp|ssl)://", "critical", "crypto-miner pool address"),
    (r"\bxmrig\b|\bminerd\b|cryptonight|randomx|nicehash|\bmonero\b", "critical", "crypto-miner reference"),
    (r"discord(app)?\.com/api/webhooks", "high", "Discord webhook (common exfiltration channel)"),
    (r"api\.telegram\.org/bot", "high", "Telegram bot API (common exfiltration channel)"),
    (r"pastebin\.com|hastebin|paste\.ee|transfer\.sh|anonfiles|file\.io|ngrok\.io|requestbin",
     "medium", "paste/file-drop or tunnelling service"),
    (r"\.onion\b", "high", "Tor hidden service"),
    (r"(^|[^\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)([^\d.]|$)", "low",
     "hard-coded IP address"),
    (r"169\.254\.169\.254|metadata\.google\.internal|100\.100\.100\.200", "critical",
     "cloud metadata endpoint (credential theft)"),
]
SHELL_HINTS = r"(cmd(\.exe)?\s*/c|powershell(\.exe)?\s+-(e|enc|encodedcommand|nop|w\s+hidden)|/bin/(ba)?sh\s+-c|" \
              r"curl\s+[^|]+\|\s*(ba)?sh|wget\s+[^|]+\|\s*(ba)?sh|certutil\s+-urlcache|bitsadmin|mshta|rundll32|regsvr32|" \
              r"vssadmin\s+delete|wbadmin\s+delete|bcdedit|cipher\s+/w|icacls\s|takeown\s|chmod\s+[0-7]*7[0-7]*\s)"
SAFE_SUBMODULES = {"importlib.util", "importlib.metadata", "importlib.resources", "ctypes.util", "os.path",
                   "io.BytesIO", "types.SimpleNamespace", "platform"}
ESCAPE_MODULES = {"subprocess", "os", "sys", "ctypes", "cffi", "socket", "shutil", "importlib", "builtins",
                  "runpy", "code", "codeop", "marshal", "pickle", "shelve", "dill", "multiprocessing", "pty",
                  "signal", "winreg", "_winreg", "msvcrt", "posix", "nt", "pathlib", "tempfile", "glob",
                  "webbrowser", "http", "urllib", "requests", "ftplib", "smtplib", "telnetlib", "paramiko",
                  "pexpect", "asyncio", "threading", "_thread", "gc", "inspect", "types", "io", "mmap",
                  "zipimport", "pkgutil", "site", "sysconfig", "platform"}
PYCOMPAT_STANDINS = {"sys", "os", "os.path", "pathlib", "threading", "time", "urllib.parse", "html.parser", "numpy",
                     "numpy.linalg", "numpy.fft", "numpy.random", "numpy.polynomial", "numpy.ma"}
SYSTEM_LIBS = re.compile(r"(kernel32|ntdll|advapi32|user32|ws2_32|wininet|urlmon|shell32|crypt32|"
                         r"psapi|dbghelp|libc\.so|libc$|^c$|libdl|libpthread|msvcrt)", re.I)
CRYPTO_FNS = {"encrypt", "aes_encrypt", "aes_gcm_encrypt", "xor", "chacha", "fernet", "rc4", "seal"}
WALK_FNS = {"from_files", "list_dir", "walk", "glob", "listdir"}
READ_FNS = {"read_text", "load_json", "load_csv", "load_tensor", "from_csv", "from_npy", "read_bytes",
            "load_safetensors", "load_model", "load_checkpoint", "load_gguf", "model_info", "gguf_info",
            "read_dicom", "read_dicom_series", "load_runs"}
WRITE_FNS = {"write_text", "save_json", "save_csv", "save_tensor", "write_bytes", "save_safetensors",
             "save_model", "save_checkpoint", "save_gguf", "experiment"}
NET_FNS = {"http_get", "http_post_json", "fetch", "websocket"}
B64 = re.compile(r"^[A-Za-z0-9+/=_-]{160,}$")
HEX = re.compile(r"^(?:[0-9a-fA-F]{2}){100,}$")
EICAR = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"


class Finding:
    __slots__ = ("severity", "rule", "line", "message", "file")

    def __init__(self, severity, rule, line, message, file=None):
        self.severity, self.rule, self.line, self.message, self.file = severity, rule, line, message, file

    def as_dict(self):
        return {"severity": self.severity, "rule": self.rule, "line": self.line, "message": self.message,
                "file": self.file}

    def __repr__(self):
        return f"[{self.severity}] {self.rule} line {self.line}: {self.message}"


def entropy(s):
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def _script(ch):
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "UNKNOWN"
    return name.split(" ")[0]


# ------------------------------------------------------------------ source-level scan
def scan_text(src, file=None):
    """Checks that do not need a parse: Trojan Source, invisible and confusable characters."""
    out = []
    special = "".join(BIDI) + "".join(INVISIBLE)
    lines = src.splitlines() if any(ch in src for ch in special) else []
    for i, line in enumerate(lines, 1):
        bad = [c for c in line if c in BIDI]
        if bad:
            out.append(Finding("critical", "trojan-source", i,
                               "bidirectional control character U+%04X can make code look different from what "
                               "runs (CVE-2021-42574)" % ord(bad[0]), file))
        inv = [c for c in line if c in INVISIBLE and not (i == 1 and line.startswith("\ufeff") and c == "\ufeff")]
        if inv:
            out.append(Finding("high", "invisible-char", i,
                               "invisible character U+%04X inside the source (hidden identifiers)" % ord(inv[0]), file))
    # homoglyphs matter in identifiers only (text inside strings and comments is data)
    try:
        from .lexer import tokenize, NAME
        words = [] if src.isascii() else [(t.line, t.value) for t in tokenize(src)
                                           if t.type == NAME and not t.value.isascii()]
    except Exception:                                      # noqa: BLE001 - unparsable: scan the raw text
        words = [(i, w) for i, line in enumerate(src.splitlines(), 1) for w in re.findall(r"[^\W\d_][\w]*", line)]
    seen = set()
    for i, word in words:
        if word in seen or word.isascii():
            continue
        seen.add(word)
        scripts = {_script(c) for c in word if c.isalpha()}
        # Greek letters are normal in physics (ω, π, θ); only letters that look like Latin count
        spoof = any(c in CONFUSABLE for c in word)
        if "LATIN" in scripts and spoof:
            out.append(Finding("high", "homoglyph", i,
                               f"identifier '{word}' mixes {', '.join(sorted(scripts))} letters "
                               f"(look-alike characters)", file))
    return out


_ANY_HINT = None


def _any_hint():
    global _ANY_HINT
    if _ANY_HINT is None:
        pats = list(SECRET_HINTS) + list(PERSISTENCE_HINTS) + [p for p, _, _ in NET_BAD] + [SHELL_HINTS]
        _ANY_HINT = re.compile("|".join(f"(?:{p})" for p in pats), re.I)
    return _ANY_HINT


def scan_program(prog, src="", file=None):
    """AST-level behaviour analysis. Returns a list of Finding."""
    from . import ast as A
    out = scan_text(src, file) if src else []
    calls = {}           # name -> first line
    strings = []         # (line, value)
    pyimports = []       # (line, module or None when computed)
    cloads = []          # (line, name or None)
    loops_with = []      # sets of call names inside each loop body

    def callname(f):
        return f.id if isinstance(f, A.Name) else (f.name if isinstance(f, A.Attr) else None)

    def collect(n, sink):
        if isinstance(n, (list, tuple)):
            for x in n:
                collect(x, sink)
            return
        if not isinstance(n, A.AST):
            return
        if isinstance(n, A.Call):
            nm = callname(n.func)
            if nm:
                sink.add(nm)
        for fld in n._fields:
            collect(getattr(n, fld), sink)

    def walk(n, line=0):
        if isinstance(n, (list, tuple)):
            for x in n:
                walk(x, line)
            return
        if not isinstance(n, A.AST):
            return
        line = n.line or line
        if isinstance(n, A.Str):
            strings.append((line, n.value))
        elif isinstance(n, A.FStr):
            for p in n.parts:
                if isinstance(p, str):
                    strings.append((line, p))
        elif isinstance(n, A.Call):
            nm = callname(n.func)
            if nm:
                calls.setdefault(nm, line)
                first = n.args[0] if n.args else None
                lit = first.value if isinstance(first, A.Str) else None
                if nm in ("pyimport", "pyfrom"):
                    pyimports.append((line, lit))
                elif nm == "cload":
                    cloads.append((line, lit))
        elif isinstance(n, (A.For, A.While)):
            s = set()
            collect(n.body, s)
            if isinstance(n, A.For):
                collect(n.iter, s)
            loops_with.append((line, s))
        elif isinstance(n, A.Use):
            calls.setdefault(f"use {n.path}", line)
        for fld in n._fields:
            walk(getattr(n, fld), line)

    walk(prog.body)

    reads_secret = []
    any_hint = _any_hint()
    for line, s in strings:
        if not isinstance(s, str):
            continue
        if len(s) < 160 and not any_hint.search(s) and "EICAR" not in s:
            continue                       # one combined search rules out almost every string
        # prose (a docstring, a long message) is checked only in its path-like words, so a sentence that mentions
        # "credentials" is not taken for code that reads them
        text = s
        if "\n" in s or len(s) >= 160:
            text = "\n".join(w for w in re.findall(r"[^\s'\"`]+", s) if "/" in w or "\\" in w)
        for pat in SECRET_HINTS:
            if re.search(pat, text, re.I | re.M):
                reads_secret.append(line)
                out.append(Finding("high", "credential-access", line,
                                   f"refers to a credential or browser-data location ('{_short(s)}')", file))
                break
        for pat in PERSISTENCE_HINTS:
            if re.search(pat, s, re.I):
                out.append(Finding("high", "persistence", line,
                                   f"refers to an auto-start location ('{_short(s)}')", file))
                break
        for pat, sev, what in NET_BAD:
            if re.search(pat, s, re.I):
                out.append(Finding(sev, "suspicious-endpoint", line, f"{what}: '{_short(s)}'", file))
                break
        if re.search(SHELL_HINTS, s, re.I):
            out.append(Finding("high", "shell-command", line, f"embedded shell / LOLBin command ('{_short(s)}')", file))
        st = s.strip()
        if len(st) >= 160 and (B64.match(st) or HEX.match(st)):
            out.append(Finding("medium", "obfuscation", line,
                               f"encoded blob of {len(st)} characters (base64/hex payload)", file))
        elif len(st) >= 200 and entropy(st) > 5.2 and " " not in st:
            out.append(Finding("medium", "obfuscation", line, f"high-entropy string ({entropy(st):.1f} bits/char)", file))
        if EICAR.decode() in s:
            out.append(Finding("critical", "eicar", line, "EICAR antivirus test signature", file))

    from .checker import is_python_mode
    pyc = is_python_mode(prog)
    for line, mod in pyimports:
        if pyc and mod in PYCOMPAT_STANDINS:
            continue                       # `use pycompat` gives permission-checked stand-ins (pysafe.py)
        if mod is None:
            out.append(Finding("high", "dynamic-import", line,
                               "pyimport() with a computed module name hides what is loaded", file))
        elif mod.split(".")[0] in ESCAPE_MODULES and mod not in SAFE_SUBMODULES:
            out.append(Finding("high", "sandbox-escape", line,
                               f"pyimport(\"{mod}\") gives full access to the computer, bypassing Apsi permissions", file))
    for line, lib in cloads:
        if lib is None:
            out.append(Finding("medium", "dynamic-native", line, "cload() with a computed library name", file))
        elif SYSTEM_LIBS.search(os.path.basename(lib)):
            out.append(Finding("high", "system-library", line,
                               f"loads the operating-system library '{lib}' directly (process injection, "
                               f"keylogging and privilege APIs live there)", file))

    has_net = any(n in calls for n in NET_FNS) or "use net" in calls or "run" in calls
    if reads_secret and has_net:
        out.append(Finding("critical", "exfiltration", reads_secret[0],
                           "reads credential locations and also sends data over the network", file))
    if "http_get" in calls and (cloads or any(m and m.split(".")[0] in ESCAPE_MODULES for _, m in pyimports)) \
            and any(n in calls for n in WRITE_FNS):
        out.append(Finding("high", "download-execute", calls["http_get"],
                           "downloads data, writes it to disk and can run native/Python code", file))
    for line, inside in loops_with:
        if (inside & WALK_FNS or "from_files" in calls) and inside & READ_FNS and inside & WRITE_FNS \
                and (inside & CRYPTO_FNS):
            out.append(Finding("critical", "ransomware-pattern", line,
                               "a loop reads files, encrypts them and writes them back", file))
            break
    # one finding per kind and line (a line that names .env four times is one finding)
    seen = set()
    unique = []
    for f in out:
        key = (f.severity, f.rule, f.line)
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)
    return unique


def _short(s, n=60):
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n - 1] + "…"


# ------------------------------------------------------------------ companion files
PY_RULES = [
    (r"\b(exec|eval)\s*\(\s*(base64|codecs|zlib|marshal|bytes\.fromhex|__import__)", "critical",
     "executes decoded/obfuscated code"),
    (r"__import__\s*\(\s*['\"](os|subprocess|ctypes|socket)", "high", "hidden import of a system module"),
    (r"subprocess\.(Popen|call|run|check_output)\s*\([^)]*shell\s*=\s*True", "high", "runs shell commands"),
    (r"ctypes\.(windll|WinDLL|CDLL)\s*\(?[^)]*(kernel32|ntdll|advapi32)", "high", "calls Windows system APIs"),
    (r"VirtualAlloc|WriteProcessMemory|CreateRemoteThread|SetWindowsHookEx|GetAsyncKeyState", "critical",
     "process injection or keylogging API"),
    (r"pickle\.loads?\s*\(", "medium", "unpickling can execute code"),
    (r"os\.system\s*\(", "medium", "runs shell commands"),
]


def scan_file(path, blocklist=None, rules=None):
    """Scan one file of any type. Returns (sha256, [Finding])."""
    out = []
    try:
        with open(path, "rb") as fh:
            data = fh.read(64 * 1024 * 1024)
    except OSError as e:
        return None, [Finding("info", "unreadable", 0, str(e), path)]
    digest = hashlib.sha256(data).hexdigest()
    if blocklist and digest in blocklist:
        out.append(Finding("critical", "blocklisted", 0, f"file hash {digest[:16]}… is on the block list", path))
    if EICAR in data:
        out.append(Finding("critical", "eicar", 0, "EICAR antivirus test signature", path))
    low = path.lower()
    if low.endswith(".apsi"):
        from .parser import parse
        src = data.decode("utf-8", "replace")
        try:
            out += scan_program(parse(src), src, path)
        except Exception as e:  # unparsable: still run the text checks
            out += scan_text(src, path)
            out.append(Finding("info", "parse-error", getattr(e, "line", 0) or 0, str(e).splitlines()[0], path))
    elif low.endswith((".py", ".pyw")):
        txt = data.decode("utf-8", "replace")
        out += scan_text(txt, path)
        for pat, sev, what in PY_RULES:
            m = re.search(pat, txt)
            if m:
                out.append(Finding(sev, "python-" + what.split()[0], txt[:m.start()].count("\n") + 1, what, path))
    elif data[:2] == b"MZ" or data[:4] == b"\x7fELF" or data[:4] in (b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf"):
        kind = "Windows executable/DLL" if data[:2] == b"MZ" else "native binary"
        out.append(Finding("info", "native-binary", 0, f"{kind}, sha256 {digest}", path))
        if low.endswith((".apsi", ".txt", ".json", ".csv", ".md", ".png", ".jpg", ".pdf")):
            out.append(Finding("critical", "disguised-binary", 0, "executable disguised with a document extension", path))
        if re.search(rb"(VirtualAllocEx|WriteProcessMemory|CreateRemoteThread)", data) and \
                re.search(rb"(OpenProcess)", data):
            out.append(Finding("high", "injection-imports", 0, "imports process-injection APIs", path))
    if rules is not None:
        try:
            for m in rules.match(data=data):
                out.append(Finding("high", "yara:" + m.rule, 0, f"YARA rule {m.rule} matched", path))
        except Exception:
            pass
    return digest, out


def load_blocklist():
    from .security import config_dir
    path = os.path.join(config_dir(), "blocklist.txt")
    hashes = set()
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                t = line.split("#")[0].strip().lower()
                if re.fullmatch(r"[0-9a-f]{64}", t):
                    hashes.add(t)
    except OSError:
        pass
    return hashes


def load_yara():
    """Compile ~/.apsi/rules/*.yar when yara-python is installed (apsi pkg install yara-python)."""
    from .security import config_dir
    d = os.path.join(config_dir(), "rules")
    if not os.path.isdir(d):
        return None
    try:
        import yara
    except ImportError:
        return None
    files = {f: os.path.join(d, f) for f in os.listdir(d) if f.endswith((".yar", ".yara"))}
    return yara.compile(filepaths=files) if files else None


def system_antivirus(path):
    """Ask Microsoft Defender or ClamAV to scan `path`. Returns (engine, clean: bool|None, detail)."""
    import shutil
    import subprocess
    if os.name == "nt":
        root = os.environ.get("ProgramFiles", r"C:\Program Files")
        mp = os.path.join(root, "Windows Defender", "MpCmdRun.exe")
        if os.path.exists(mp):
            r = subprocess.run([mp, "-Scan", "-ScanType", "3", "-File", os.path.abspath(path), "-DisableRemediation"],
                               capture_output=True, text=True)
            return "Microsoft Defender", r.returncode == 0, (r.stdout or r.stderr).strip().splitlines()[-1:]
    for exe in ("clamdscan", "clamscan"):
        if shutil.which(exe):
            r = subprocess.run([exe, "--no-summary", "-r", os.path.abspath(path)], capture_output=True, text=True)
            return "ClamAV", r.returncode == 0, [l for l in r.stdout.splitlines() if "FOUND" in l][:20]
    return None, None, ["no system antivirus found (Microsoft Defender or ClamAV)"]


SCAN_EXT = (".apsi", ".py", ".pyw", ".dll", ".so", ".dylib", ".exe", ".pyd", ".bin", ".cmd", ".bat", ".ps1", ".sh")


def scan_paths(paths, with_av=False):
    blocklist = load_blocklist()
    rules = load_yara()
    results = []
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, names in os.walk(p):
                dirs[:] = [d for d in dirs if d not in ("node_modules", ".git", "__pycache__")]
                files += [os.path.join(root, n) for n in names if n.lower().endswith(SCAN_EXT)]
        else:
            files.append(p)
    for f in sorted(files):
        digest, found = scan_file(f, blocklist, rules)
        results.append((f, digest, found))
    av = system_antivirus(paths[0] if len(paths) == 1 else os.path.commonpath([os.path.abspath(p) for p in paths])) \
        if with_av else None
    return results, av


def worst(findings):
    return max((SEVERITY[f.severity] for f in findings), default=0)


# ------------------------------------------------------------------ pre-run gate
def gate_program(prog, src, path, policy_name, trust=False, stream=sys.stderr, python_all=False):
    """Called by `apsi run` before execution. Critical findings stop the run unless --trust;
    high findings stop it under the sandbox/strict profiles and warn otherwise."""
    mode = os.environ.get("APSI_SCAN") or _cfg().get("scan_on_run", "on")
    if mode == "off" or trust:
        return True
    if _known_clean(src):
        return True
    found = [f for f in scan_program(prog, src, path) if SEVERITY[f.severity] >= SEVERITY["high"]
             and not (python_all and f.rule == "sandbox-escape")]   # granted explicitly with --allow-python=all
    if not found:
        _remember_clean(src)
        return True
    w = worst(found)
    block = w >= SEVERITY["critical"] or (policy_name in ("sandbox", "strict") and w >= SEVERITY["high"])
    head = "Apsi Guard blocked this program" if block else "Apsi Guard warning"
    print(f"\n⛨ {head}: {path}", file=stream)
    for f in sorted(found, key=lambda x: -SEVERITY[x.severity])[:12]:
        print(f"  {f.severity.upper():8} line {f.line:<5} {f.rule}: {f.message}", file=stream)
    if block:
        print("  Review the code. If you trust it, run again with --trust "
              "(or 'apsi scan' for a full report).", file=stream)
    return not block


def _clean_path(src):
    """A signed marker per (Guard rules, source): a program that scanned clean is not scanned again (a 20 000-line
    program saves about half a second per run). Any change to the program or to these rules gives a new key."""
    import hashlib
    import hmac
    from .compile_cache import cache_dir, _secret
    with open(__file__, "rb") as fh:
        rules = hashlib.sha256(fh.read()).hexdigest()
    key = hashlib.sha256((rules + "\0" + src).encode("utf-8")).hexdigest()
    sig = hmac.new(_secret(), key.encode(), hashlib.sha256).hexdigest()
    return os.path.join(cache_dir(), "guard", key[:2], key + ".ok"), sig


def _known_clean(src):
    if len(src) < 20000:
        return False
    try:
        from .compile_cache import enabled
        if not enabled():
            return False
        path, sig = _clean_path(src)
        with open(path, encoding="ascii") as fh:
            import hmac
            return hmac.compare_digest(fh.read().strip(), sig)
    except Exception:                     # noqa: BLE001 - no marker: scan normally
        return False


def _remember_clean(src):
    if len(src) < 20000:
        return
    try:
        from .compile_cache import enabled
        if not enabled():
            return
        path, sig = _clean_path(src)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="ascii") as fh:
            fh.write(sig)
    except Exception:                     # noqa: BLE001
        pass


def _cfg():
    try:
        from .security import load_config
        return load_config()
    except Exception:
        return {}


# ------------------------------------------------------------------ behaviour monitor
class BehaviourMonitor:
    """Counts destructive file operations on files that existed before the program touched them.
    Too many in a sliding window (default 150 in 20 s) looks like ransomware or a wiper, and the
    program is stopped. Decoy files (names starting with '.apsi-decoy') stop it immediately."""

    def __init__(self, limit=None, window=None):
        cfg = _cfg()
        self.limit = int(limit or os.environ.get("APSI_GUARD_MAX_OVERWRITES") or cfg.get("guard_max_overwrites", 150))
        self.window = float(window or os.environ.get("APSI_GUARD_WINDOW") or cfg.get("guard_window_s", 20))
        self.events = []
        self.touched = set()
        self.lock = threading.Lock()
        self.tripped = None

    def destructive(self, full, what):
        from .security import PermissionErr
        name = os.path.basename(full)
        if name.startswith(".apsi-decoy"):
            self.tripped = f"touched decoy file {full}"
        elif os.path.exists(full) and full not in self.touched:
            now = time.monotonic()
            with self.lock:
                self.touched.add(full)
                self.events.append(now)
                while self.events and now - self.events[0] > self.window:
                    self.events.pop(0)
                if len(self.events) > self.limit:
                    self.tripped = (f"{len(self.events)} existing files modified or deleted in "
                                    f"{self.window:.0f} s by {what} (limit {self.limit})")
        if self.tripped:
            try:
                from .security import _record
                _record(None, "guard-stop", self.tripped)
            except Exception:
                pass
            raise PermissionErr(f"Apsi Guard stopped the program: {self.tripped}. This looks like ransomware or "
                                f"a wiper. If it is intended, raise the limit with APSI_GUARD_MAX_OVERWRITES.")


def monitor_for(interp):
    m = getattr(interp, "_guard", None)
    if m is None:
        m = BehaviourMonitor()
        try:
            interp._guard = m
        except Exception:
            pass
    return m


# ------------------------------------------------------------------ Python import & native-code control
def check_pyimport(policy, module):
    from .security import PermissionErr
    if policy.python == "all":
        return
    root = (module or "").split(".")[0]
    if module in SAFE_SUBMODULES:
        return
    if root in ESCAPE_MODULES or root.startswith("_"):
        raise PermissionErr(f"pyimport(\"{module}\") would bypass Apsi's permissions; it needs "
                            f"--allow-python=all (or the developer profile)")


def check_native(policy, path):
    """Refuse native libraries from world-writable or temporary folders, and enforce sha256 pins
    from --allow-ffi=name@sha256,... or config 'ffi_pins'."""
    from .security import PermissionErr
    import tempfile
    real = os.path.realpath(path)
    if os.path.exists(real):
        d = os.path.dirname(real)
        if os.name != "nt":
            st = os.stat(d)
            if st.st_mode & 0o002 and not st.st_mode & 0o1000:
                raise PermissionErr(f"cload: '{real}' is in a world-writable folder; anyone could replace it")
        tmp = os.path.realpath(tempfile.gettempdir())
        if real.startswith(tmp + os.sep) and not os.environ.get("APSI_ALLOW_TEMP_LIBS"):
            raise PermissionErr(f"cload: loading native code from the temporary folder is blocked ('{real}')")
        dl = os.path.join(os.path.expanduser("~"), "Downloads")
        if real.startswith(os.path.realpath(dl) + os.sep) and not _pins(policy):
            raise PermissionErr(f"cload: '{real}' is in Downloads; pin it with --allow-ffi={os.path.basename(real)}@<sha256>")
        pins = _pins(policy)
        if pins:
            with open(real, "rb") as fh:
                h = hashlib.sha256(fh.read()).hexdigest()
            want = pins.get(os.path.basename(real)) or pins.get(real)
            if want is None:
                raise PermissionErr(f"cload: '{os.path.basename(real)}' is not in the pinned libraries "
                                    f"({', '.join(sorted(pins))}); its sha256 is {h}")
            if h != want.lower():
                raise PermissionErr(f"cload: '{real}' does not match its pinned sha256 (expected {want[:16]}…, got {h[:16]}…)")
    return real


def _pins(policy):
    pins = dict(_cfg().get("ffi_pins", {}) or {})
    pins.update(getattr(policy, "ffi_pins", {}) or {})
    return pins


def report_json(results, av=None):
    return json.dumps({"files": [{"file": f, "sha256": d, "findings": [x.as_dict() for x in fs]}
                                 for f, d, fs in results],
                       "antivirus": None if not av else {"engine": av[0], "clean": av[1], "detail": av[2]}},
                      indent=2, ensure_ascii=False)
