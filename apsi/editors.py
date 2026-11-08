"""`apsi editors` — find the code editors on this computer and give each one Apsi support.

    apsi editors                     list editors, whether they were found and whether Apsi is installed in them
    apsi editors install [NAMES]     install into the named editors, or into every editor that was found
    apsi editors uninstall [NAMES]   remove it again
    apsi editors path                print the folder that holds the editor integrations

Supported automatically:
  VS Code family (one extension):  Visual Studio Code, VS Code Insiders, Cursor, Windsurf, VSCodium, Trae
  Sublime Text 3/4                 syntax, build system (Ctrl+B), comments, indentation, symbols, snippets
  Neovim and Vim                   filetype, syntax, indentation, :make, and the language server in Neovim
  Notepad++ (Windows)              user-defined language with Apsi colours
With a short manual step (shown by `apsi editors`): JetBrains IDEs, Helix, Zed, Emacs.

Nothing outside the editors' own extension/package folders is changed. `--json` prints machine-readable
results (the setup wizard uses it to record what it must remove on uninstall).
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import zipfile

IS_WIN = os.name == "nt"
IS_MAC = sys.platform == "darwin"
HOME = os.path.expanduser("~")
EXT_ID = "apsi.apsi-language"
APPDATA = os.environ.get("APPDATA") or os.path.join(HOME, "AppData", "Roaming")
LOCALAPPDATA = os.environ.get("LOCALAPPDATA") or os.path.join(HOME, "AppData", "Local")
XDG_CONFIG = os.environ.get("XDG_CONFIG_HOME") or os.path.join(HOME, ".config")
XDG_DATA = os.environ.get("XDG_DATA_HOME") or os.path.join(HOME, ".local", "share")


# ---------------------------------------------------------------------------- where the integrations live
def integrations_root(explicit=None):
    """Folder that contains editors/ (and, in an installed Apsi, vscode/*.vsix)."""
    if explicit:
        return os.path.abspath(explicit)
    env = os.environ.get("APSI_ROOT")
    if env and os.path.isdir(os.path.join(env, "editors")):
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):                      # source checkout: <root>/apsi; installed: <root>/runtime/python/…/apsi
        if os.path.isdir(os.path.join(d, "editors")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    for guess in ([os.path.join(LOCALAPPDATA, "Programs", "Apsi")] if IS_WIN else
                  [os.path.join(XDG_DATA, "apsi"), "/usr/share/apsi"]):
        if os.path.isdir(os.path.join(guess, "editors")):
            return guess
    return None


def find_vsix(root):
    for pattern in ("vscode/*.vsix", "dist/apsi-language-*.vsix", "editors/vscode/*.vsix"):
        hits = sorted(glob.glob(os.path.join(root, pattern)))
        if hits:
            return hits[-1]
    return None


def _which(name):
    p = shutil.which(name)
    return p


def _first(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------- VS Code family
VSCODE_FAMILY = [
    # id, name, CLI names, Windows install folders (relative to LOCALAPPDATA\Programs and ProgramFiles),
    # Linux/mac CLI paths, user extension folder
    ("vscode", "Visual Studio Code", ["code"], ["Microsoft VS Code"],
     ["/usr/share/code/bin/code", "/snap/bin/code", "/usr/bin/code",
      "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code"], ".vscode"),
    ("vscode-insiders", "VS Code Insiders", ["code-insiders"], ["Microsoft VS Code Insiders"],
     ["/usr/share/code-insiders/bin/code-insiders", "/snap/bin/code-insiders",
      "/Applications/Visual Studio Code - Insiders.app/Contents/Resources/app/bin/code-insiders"], ".vscode-insiders"),
    ("cursor", "Cursor", ["cursor"], ["cursor", "Cursor"],
     ["/usr/bin/cursor", "/opt/Cursor/resources/app/bin/cursor",
      "/Applications/Cursor.app/Contents/Resources/app/bin/cursor"], ".cursor"),
    ("windsurf", "Windsurf", ["windsurf"], ["Windsurf"],
     ["/usr/bin/windsurf", "/usr/share/windsurf/bin/windsurf",
      "/Applications/Windsurf.app/Contents/Resources/app/bin/windsurf"], ".windsurf"),
    ("vscodium", "VSCodium", ["codium"], ["VSCodium"],
     ["/usr/bin/codium", "/snap/bin/codium", "/usr/share/codium/bin/codium",
      "/Applications/VSCodium.app/Contents/Resources/app/bin/codium"], ".vscode-oss"),
    ("trae", "Trae", ["trae"], ["Trae"],
     ["/usr/bin/trae", "/Applications/Trae.app/Contents/Resources/app/bin/trae"], ".trae"),
]


def _vscode_cli(names, win_dirs, unix_paths):
    for n in names:
        p = _which(n + ".cmd") if IS_WIN else _which(n)
        if p:
            return p
    if IS_WIN:
        roots = [os.path.join(LOCALAPPDATA, "Programs"), os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")]
        cands = []
        for r in roots:
            for d in win_dirs:
                for n in names:
                    cands += [os.path.join(r, d, "bin", n + ".cmd"),
                              os.path.join(r, d, "resources", "app", "bin", n + ".cmd")]
        return _first(cands)
    return _first(unix_paths)


class VSCodeLike:
    kind = "extension"

    def __init__(self, ident, name, clis, win_dirs, unix_paths, data):
        self.id, self.name = ident, name
        self.cli = _vscode_cli(clis, win_dirs, unix_paths)
        self.ext_dir = os.path.join(HOME, data, "extensions")
        self.found = bool(self.cli) or os.path.isdir(os.path.join(HOME, data))

    def installed(self):
        return bool(glob.glob(os.path.join(self.ext_dir, EXT_ID + "-*")))

    def install(self, root):
        vsix = find_vsix(root)
        result = {"files": [], "dirs": [], "uninstall_cli": None}
        if self.cli and vsix:
            r = _run([self.cli, "--install-extension", vsix, "--force"], timeout=180)
            if r.returncode == 0:
                result["uninstall_cli"] = self.cli
                result["detail"] = f"installed {os.path.basename(vsix)} with {os.path.basename(self.cli)}"
                return result
        # no command-line tool (or it failed): unpack the extension into the editor's extension folder
        version = _ext_version(root, vsix)
        target = os.path.join(self.ext_dir, f"{EXT_ID}-{version}")
        for old in glob.glob(os.path.join(self.ext_dir, EXT_ID + "-*")):
            shutil.rmtree(old, ignore_errors=True)
        os.makedirs(self.ext_dir, exist_ok=True)
        if vsix:
            _unpack_vsix(vsix, target)
        else:
            src = os.path.join(root, "editors", "vscode")
            if not os.path.isfile(os.path.join(src, "package.json")):
                raise RuntimeError("no Apsi extension (.vsix) found to install")
            shutil.copytree(src, target, ignore=shutil.ignore_patterns("*.vsix", ".vscode-test"))
        _register_extension(self.ext_dir, target, version)
        result["dirs"].append(target)
        result["detail"] = f"copied the extension into {target}"
        return result

    def uninstall(self):
        if self.cli:
            _run([self.cli, "--uninstall-extension", EXT_ID], timeout=120)
        for d in glob.glob(os.path.join(self.ext_dir, EXT_ID + "-*")):
            shutil.rmtree(d, ignore_errors=True)
        _register_extension(self.ext_dir, None, None)


def _ext_version(root, vsix):
    if vsix:
        try:
            with zipfile.ZipFile(vsix) as z:
                return json.loads(z.read("extension/package.json"))["version"]
        except (KeyError, ValueError, zipfile.BadZipFile, OSError):
            pass
    try:
        with open(os.path.join(root, "editors", "vscode", "package.json"), encoding="utf-8") as fh:
            return json.load(fh)["version"]
    except (OSError, ValueError, KeyError):
        from . import __version__
        return __version__


def _unpack_vsix(vsix, target):
    tmp = target + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(vsix) as z:
        for m in z.infolist():
            if not m.filename.startswith("extension/") or m.is_dir():
                continue
            rel = m.filename[len("extension/"):]
            dest = os.path.normpath(os.path.join(tmp, rel))
            if not dest.startswith(os.path.normpath(tmp) + os.sep):
                raise RuntimeError(f"unsafe path in extension package: {m.filename}")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with z.open(m) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
    shutil.rmtree(target, ignore_errors=True)
    os.replace(tmp, target)


def _register_extension(ext_dir, target, version):
    """Add the extension to extensions.json, the list newer VS Code builds read at start-up."""
    index = os.path.join(ext_dir, "extensions.json")
    if not os.path.isfile(index):
        return
    try:
        with open(index, encoding="utf-8") as fh:
            items = json.load(fh)
        if not isinstance(items, list):
            return
    except (OSError, ValueError):
        return
    items = [i for i in items if (i.get("identifier") or {}).get("id", "").lower() != EXT_ID]
    if target is None:                        # unregister
        _write_json(index, items)
        return
    p = target.replace("\\", "/")
    if IS_WIN:
        p = "/" + p[0].lower() + p[1:]
    items.append({"identifier": {"id": EXT_ID}, "version": version,
                  "location": {"$mid": 1, "path": p, "scheme": "file"},
                  "relativeLocation": os.path.basename(target),
                  "metadata": {"installedTimestamp": 0, "source": "vsix"}})
    _write_json(index, items)


def _write_json(path, value):
    tmp = path + ".apsi-tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(value, fh)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------- folder-copy editors
class FolderEditor:
    kind = "package"

    def __init__(self, ident, name, found, target, source, note="", owned=None):
        self.id, self.name, self.found, self.target, self.source, self.note = ident, name, found, target, source, note
        self.owned = owned or target          # what uninstall removes (e.g. the whole pack/apsi folder)

    def installed(self):
        return bool(self.target) and os.path.exists(self.target)

    def install(self, root):
        src = os.path.join(root, "editors", *self.source.split("/"))
        if not os.path.exists(src):
            raise RuntimeError(f"missing integration files: {src}")
        if os.path.isdir(src):
            shutil.rmtree(self.target, ignore_errors=True)
            shutil.copytree(src, self.target)
            return {"files": [], "dirs": [self.owned], "detail": f"copied into {self.target}"}
        os.makedirs(os.path.dirname(self.target), exist_ok=True)
        shutil.copyfile(src, self.target)
        return {"files": [self.target], "dirs": [], "detail": f"copied to {self.target}" + (f" — {self.note}" if self.note else "")}

    def uninstall(self):
        if not self.owned:
            return
        if os.path.isdir(self.owned):
            shutil.rmtree(self.owned, ignore_errors=True)
        elif os.path.exists(self.owned):
            os.remove(self.owned)


def _sublime():
    if IS_WIN:
        bases = [os.path.join(APPDATA, "Sublime Text"), os.path.join(APPDATA, "Sublime Text 3")]
    elif IS_MAC:
        lib = os.path.join(HOME, "Library", "Application Support")
        bases = [os.path.join(lib, "Sublime Text"), os.path.join(lib, "Sublime Text 3")]
    else:
        bases = [os.path.join(XDG_CONFIG, "sublime-text"), os.path.join(XDG_CONFIG, "sublime-text-3")]
    base = _first(bases)
    exe = _which("subl") or (_first([os.path.join(os.environ.get("ProgramFiles", ""), "Sublime Text", "sublime_text.exe")])
                             if IS_WIN else None)
    base = base or (bases[0] if exe else None)
    target = os.path.join(base, "Packages", "Apsi") if base else None
    return FolderEditor("sublime", "Sublime Text", bool(base), target, "sublime/Apsi")


def _neovim():
    cfg = os.path.join(LOCALAPPDATA, "nvim") if IS_WIN else os.path.join(XDG_CONFIG, "nvim")
    data = os.path.join(LOCALAPPDATA, "nvim-data") if IS_WIN else os.path.join(XDG_DATA, "nvim")
    found = bool(_which("nvim")) or os.path.isdir(cfg)
    pack = os.path.join(data, "site", "pack", "apsi")
    return FolderEditor("neovim", "Neovim", found, os.path.join(pack, "start", "apsi"), "vim", owned=pack)


def _vim():
    base = os.path.join(HOME, "vimfiles") if IS_WIN else os.path.join(HOME, ".vim")
    found = bool(_which("vim") or _which("gvim")) or os.path.isdir(base)
    if IS_WIN and not found:
        found = bool(glob.glob(os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Vim", "vim*", "gvim.exe")))
    pack = os.path.join(base, "pack", "apsi")
    return FolderEditor("vim", "Vim", found, os.path.join(pack, "start", "apsi"), "vim", owned=pack)


def _notepadpp():
    if not IS_WIN:
        return None
    cfg = os.path.join(APPDATA, "Notepad++")
    exe = _first([os.path.join(os.environ.get("ProgramFiles", ""), "Notepad++", "notepad++.exe"),
                  os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Notepad++", "notepad++.exe")])
    found = os.path.isdir(cfg) or bool(exe)
    return FolderEditor("notepadpp", "Notepad++", found, os.path.join(cfg, "userDefineLangs", "Apsi.xml"),
                        "notepad++/Apsi.xml", note="restart Notepad++ to load it")


class ManualEditor:
    kind = "manual"

    def __init__(self, ident, name, found, steps):
        self.id, self.name, self.found, self.steps = ident, name, found, steps

    def installed(self):
        return False


def _jetbrains(root):
    bases = [os.path.join(APPDATA, "JetBrains")] if IS_WIN else \
        [os.path.join(HOME, "Library", "Application Support", "JetBrains")] if IS_MAC else [os.path.join(XDG_CONFIG, "JetBrains")]
    found = any(os.path.isdir(b) and os.listdir(b) for b in bases)
    bundle = os.path.join(root or "<apsi>", "editors", "textmate", "Apsi.tmbundle")
    return ManualEditor("jetbrains", "JetBrains IDEs (IntelliJ, PyCharm, CLion, …)", found, [
        f"Settings → Editor → TextMate Bundles → + → {bundle}",
        "language server (optional): install the LSP4IJ plugin, add a server with command `apsi lsp` for *.apsi",
    ])


def _helix(root):
    cfg = os.path.join(APPDATA, "helix") if IS_WIN else os.path.join(XDG_CONFIG, "helix")
    return ManualEditor("helix", "Helix", bool(_which("hx") or _which("helix")) or os.path.isdir(cfg), [
        f"append {os.path.join(root or '<apsi>', 'editors', 'helix', 'languages.toml')} to {os.path.join(cfg, 'languages.toml')}",
    ])


def _zed(root):
    cfg = os.path.join(APPDATA, "Zed") if IS_WIN else os.path.join(XDG_CONFIG, "zed")
    return ManualEditor("zed", "Zed", bool(_which("zed")) or os.path.isdir(cfg), [
        f"merge {os.path.join(root or '<apsi>', 'editors', 'zed', 'settings.json')} into {os.path.join(cfg, 'settings.json')}",
    ])


def _emacs(root):
    return ManualEditor("emacs", "Emacs", bool(_which("emacs")) or os.path.isdir(os.path.join(HOME, ".emacs.d")), [
        f"(load \"{os.path.join(root or '<apsi>', 'editors', 'emacs', 'apsi-mode.el').replace(os.sep, '/')}\") in your init file",
    ])


def all_editors(root=None):
    eds = [VSCodeLike(*spec) for spec in VSCODE_FAMILY]
    eds += [e for e in (_sublime(), _neovim(), _vim(), _notepadpp()) if e is not None]
    eds += [_jetbrains(root), _helix(root), _zed(root), _emacs(root)]
    return eds


# ---------------------------------------------------------------------------- helpers
def _run(cmd, timeout=120):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              shell=IS_WIN and cmd[0].lower().endswith((".cmd", ".bat")))
    except (OSError, subprocess.TimeoutExpired) as e:
        return subprocess.CompletedProcess(cmd, 1, "", str(e))


def _select(eds, names):
    if not names:
        return [e for e in eds if e.found and e.kind != "manual"]
    by = {e.id: e for e in eds}
    alias = {"code": "vscode", "vs-code": "vscode", "nvim": "neovim", "notepad++": "notepadpp", "npp": "notepadpp",
             "sublime-text": "sublime", "subl": "sublime", "codium": "vscodium", "insiders": "vscode-insiders"}
    out = []
    for n in names:
        key = alias.get(n.lower(), n.lower())
        if key == "all":
            return [e for e in eds if e.kind != "manual"]
        if key not in by:
            raise SystemExit(f"apsi editors: unknown editor '{n}' (known: {', '.join(by)})")
        out.append(by[key])
    return out


# ---------------------------------------------------------------------------- commands
def cmd_editors(args):
    action = args.action or "list"
    root = integrations_root(getattr(args, "root", None))
    eds = all_editors(root)
    if action == "path":
        print(root or "(not found)")
        return 0 if root else 1
    if action == "list":
        if args.json:
            print(json.dumps([{"id": e.id, "name": e.name, "kind": e.kind, "found": e.found,
                               "installed": e.installed()} for e in eds]))
            return 0
        print(f"{'editor':<46} {'found':<7} Apsi support")
        for e in eds:
            if e.kind == "manual":
                status = "manual step:" if e.found else "—"
            else:
                status = "installed" if e.installed() else ("not installed" if e.found else "—")
            print(f"  {e.name:<44} {'yes' if e.found else 'no':<7} {status}")
            if e.kind == "manual" and e.found:
                for s in e.steps:
                    print(f"  {'':<52}{s}")
        print("\nInstall into every editor found:  apsi editors install"
              "\nOnly some:                         apsi editors install cursor vscode sublime")
        return 0
    if root is None and action == "install":
        print("apsi editors: cannot find the editor integration files (set APSI_ROOT to the Apsi folder)",
              file=sys.stderr)
        return 2
    targets = _select(eds, args.names)
    results, status = [], 0
    for e in targets:
        if e.kind == "manual":
            results.append({"id": e.id, "name": e.name, "ok": False, "manual": e.steps})
            continue
        try:
            if action == "install":
                r = e.install(root)
                results.append({"id": e.id, "name": e.name, "ok": True, **r})
            else:
                had = e.installed()
                e.uninstall()
                results.append({"id": e.id, "name": e.name, "ok": True,
                                "detail": "removed" if had else "was not installed"})
        except Exception as ex:                      # one broken editor must not stop the others
            status = 1
            results.append({"id": e.id, "name": e.name, "ok": False, "error": str(ex)})
    if args.json:
        print(json.dumps({"results": results}))
        return status
    if not targets:
        print("No supported editor was found. `apsi editors` lists what is supported.")
        return 0
    for r in results:
        if r.get("manual"):
            print(f"• {r['name']}: needs a manual step")
            for s in r["manual"]:
                print(f"    {s}")
        elif r["ok"]:
            print(f"✓ {r['name']}: " + (r.get("detail") or ("removed" if action == "uninstall" else "done")))
        else:
            print(f"✗ {r['name']}: {r['error']}")
    if action == "install":
        print("\nRestart the editors that were open to load Apsi.")
    return status


def add_parser(sub):
    p = sub.add_parser("editors", help="install Apsi support into VS Code, Cursor, Windsurf, Sublime, Vim/Neovim, …")
    p.add_argument("action", nargs="?", choices=["list", "install", "uninstall", "path"])
    p.add_argument("names", nargs="*", help="editors (default: every editor found), or 'all'")
    p.add_argument("--root", help="folder that contains editors/ (default: found automatically)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(fn=cmd_editors)
