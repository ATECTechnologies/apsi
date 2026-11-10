"""Build `ApsiSetup-<v>-linux.sh`: one self-extracting installer for every Linux distribution (works offline).

    python3 packaging/make_linux_installer.py [OUT_DIR]          (default: dist/linux)

The installer is a POSIX shell script followed by a tar.gz archive. It installs for the current user (no sudo):
Apsi into ~/.local/share/apsi, the commands apsi / apsi-repl / apsi-run into ~/.local/bin, the .apsi file type with
its icon, "Run with Apsi", a ~/Templates entry, and Apsi support in the code editors it finds. It uses the
system's python3 (3.12+) and numpy; when numpy is missing it creates a private environment and installs it with pip.
`sh ApsiSetup-<v>-linux.sh --help` lists the options; `~/.local/share/apsi/uninstall.sh` removes everything again.
"""
import io
import os
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEADER = r'''#!/bin/sh
# Apsi @V@ installer for Linux (all distributions, x86-64 and ARM). No sudo needed.
#   sh ApsiSetup-@V@-linux.sh                 install for the current user
#   sh ApsiSetup-@V@-linux.sh --prefix DIR    install somewhere else (default: ~/.local/share/apsi)
#   sh ApsiSetup-@V@-linux.sh --no-editors    do not add Apsi to code editors
#   sh ApsiSetup-@V@-linux.sh --uninstall     remove Apsi
set -e
V="@V@"
PREFIX="${APSI_PREFIX:-$HOME/.local/share/apsi}"
BIN="${APSI_BIN:-$HOME/.local/bin}"
EDITORS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) shift; PREFIX="$1" ;;
    --prefix=*) PREFIX="${1#--prefix=}" ;;
    --no-editors) EDITORS=0 ;;
    --uninstall) [ -x "$PREFIX/uninstall.sh" ] && exec "$PREFIX/uninstall.sh"; echo "Apsi is not installed in $PREFIX"; exit 1 ;;
    -h|--help) sed -n 2,7p "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
  shift
done
say() { printf '\033[36m==>\033[0m %s\n' "$*"; }

say "Apsi $V installer"
PY=""
for c in python3.15 python3.14 python3.13 python3.12 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
    PY="$(command -v "$c")"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.12 or newer is required (Ubuntu 24.04+, Debian 13+, Fedora 39+, or: uv python install 3.12)."
  echo "  Ubuntu/Debian: sudo apt install python3 python3-numpy    Fedora: sudo dnf install python3 python3-numpy"
  echo "  Arch: sudo pacman -S python python-numpy                  openSUSE: sudo zypper install python3 python3-numpy"
  exit 1
fi
say "using $("$PY" --version 2>&1) ($PY)"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT INT TERM
SKIP=$(awk '/^__APSI_ARCHIVE__$/ { print NR + 1; exit }' "$0")
tail -n +"$SKIP" "$0" | tar -xz -C "$TMP"

if [ -x "$PREFIX/uninstall.sh" ]; then
  say "replacing the Apsi already installed in $PREFIX"
  APSI_KEEP_SETTINGS=1 "$PREFIX/uninstall.sh" --quiet || true
fi
mkdir -p "$PREFIX" "$BIN"
cp -R "$TMP/apsi/." "$PREFIX/"

RUNPY="$PY"
if "$PY" -c 'import numpy' 2>/dev/null && ! "$PY" -c 'import llvmlite' 2>/dev/null; then
  # numpy is there but the native compiler's llvmlite is not: a private environment that still sees the system's
  # packages, plus llvmlite (a prebuilt wheel). Without it Apsi still works, on the bytecode engine only.
  say "installing llvmlite (Apsi's native compiler) into a private Python environment"
  if "$PY" -m venv --system-site-packages "$PREFIX/venv" 2>/dev/null && \
     "$PREFIX/venv/bin/python" -m pip install --quiet "llvmlite>=0.42" "ml_dtypes>=0.5" 2>/dev/null; then
    RUNPY="$PREFIX/venv/bin/python"
  else
    rm -rf "$PREFIX/venv"
    say "could not install llvmlite: @native functions will run on the bytecode engine"
    say "  (later: $PY -m pip install --user llvmlite)"
  fi
fi
if ! "$PY" -c 'import numpy' 2>/dev/null; then
  say "numpy is missing: creating a private Python environment and installing numpy"
  if "$PY" -m venv "$PREFIX/venv" 2>/dev/null && "$PREFIX/venv/bin/python" -m pip install --quiet "numpy>=1.22" && \
     { "$PREFIX/venv/bin/python" -m pip install --quiet "llvmlite>=0.42" "ml_dtypes>=0.5" || say "llvmlite not installed: no native compiler"; }; then
    RUNPY="$PREFIX/venv/bin/python"
  else
    rm -rf "$PREFIX/venv"
    echo "Could not install numpy. Install it with your package manager and run this installer again:"
    echo "  sudo apt install python3-numpy   (or: dnf/pacman/zypper)   or: $PY -m pip install --user numpy"
    exit 1
  fi
fi
"$RUNPY" -m compileall -q "$PREFIX/lib/apsi" >/dev/null 2>&1 || true

for cmd in apsi apsi-repl apsi-run; do
  case "$cmd" in
    apsi) body='exec "@PY@" -X utf8 -m apsi "$@"' ;;
    apsi-repl) body='exec "@PY@" -X utf8 -m apsi repl "$@"' ;;
    apsi-run) body='"@PY@" -X utf8 -m apsi run "$@"; s=$?; [ -t 0 ] && { printf "\n[exit %s] press Enter to close " "$s"; read _; }; exit $s' ;;
  esac
  printf '#!/bin/sh\n# Apsi %s\nPYTHONPATH="%s/lib${PYTHONPATH:+:$PYTHONPATH}"\nexport PYTHONPATH\n%s\n' \
    "$V" "$PREFIX" "$(printf '%s' "$body" | sed "s|@PY@|$RUNPY|g")" > "$PREFIX/bin-$cmd"
  chmod 755 "$PREFIX/bin-$cmd"
  ln -sf "$PREFIX/bin-$cmd" "$BIN/$cmd"
done
say "commands: $BIN/apsi, apsi-repl, apsi-run"

# .apsi file type, icons, "Run with Apsi", New Document template
SHARE="${XDG_DATA_HOME:-$HOME/.local/share}"
mkdir -p "$SHARE/mime/packages" "$SHARE/applications"
cp "$PREFIX/share/apsi.xml" "$SHARE/mime/packages/apsi.xml"
for s in 16 32 48 64 128 256; do
  mkdir -p "$SHARE/icons/hicolor/${s}x${s}/apps" "$SHARE/icons/hicolor/${s}x${s}/mimetypes"
  cp "$PREFIX/brand/png/apsi-$s.png" "$SHARE/icons/hicolor/${s}x${s}/apps/apsi.png"
  cp "$PREFIX/brand/png/apsi-file-$s.png" "$SHARE/icons/hicolor/${s}x${s}/mimetypes/text-x-apsi.png"
done
cat > "$SHARE/applications/apsi-run.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Apsi
Comment=Run Apsi programs
Exec="$BIN/apsi-run" %f
Icon=apsi
Terminal=true
MimeType=text/x-apsi;
NoDisplay=true
DESK
cat > "$SHARE/applications/apsi-repl.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Apsi REPL
Comment=Interactive Apsi
Exec="$BIN/apsi-repl"
Icon=apsi
Terminal=true
Categories=Development;
DESK
command -v update-mime-database >/dev/null 2>&1 && update-mime-database "$SHARE/mime" >/dev/null 2>&1 || true
command -v xdg-mime >/dev/null 2>&1 && xdg-mime default apsi-run.desktop text/x-apsi >/dev/null 2>&1 || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q "$SHARE/icons/hicolor" >/dev/null 2>&1 || true
TEMPLATES=""
[ -f "$HOME/.config/user-dirs.dirs" ] && TEMPLATES="$(sed -n 's/^XDG_TEMPLATES_DIR="\(.*\)"/\1/p' "$HOME/.config/user-dirs.dirs" | sed "s|\$HOME|$HOME|")"
[ -z "$TEMPLATES" ] && TEMPLATES="$HOME/Templates"
TEMPLATE_MADE=0
if [ -d "$TEMPLATES" ] && [ ! -e "$TEMPLATES/Apsi file.apsi" ]; then
  printf '# New Apsi program\n\nprint("Hello from Apsi!")\n' > "$TEMPLATES/Apsi file.apsi"; TEMPLATE_MADE=1
fi
say "registered .apsi files (icon, Run with Apsi, New Document)"

if [ "$EDITORS" = 1 ]; then
  PYTHONPATH="$PREFIX/lib" "$RUNPY" -m apsi editors install --root "$PREFIX" 2>/dev/null | sed 's/^/    /' || true
fi

cat > "$PREFIX/uninstall.sh" <<UNI
#!/bin/sh
# removes Apsi $V installed by ApsiSetup-$V-linux.sh
[ "\$1" = "--quiet" ] || echo "Removing Apsi $V from $PREFIX"
PYTHONPATH="$PREFIX/lib" "$RUNPY" -m apsi editors uninstall --root "$PREFIX" >/dev/null 2>&1 || true
for cmd in apsi apsi-repl apsi-run; do
  [ "\$(readlink "$BIN/\$cmd" 2>/dev/null)" = "$PREFIX/bin-\$cmd" ] && rm -f "$BIN/\$cmd"
done
rm -f "$SHARE/mime/packages/apsi.xml" "$SHARE/applications/apsi-run.desktop" "$SHARE/applications/apsi-repl.desktop"
for s in 16 32 48 64 128 256; do
  rm -f "$SHARE/icons/hicolor/\${s}x\${s}/apps/apsi.png" "$SHARE/icons/hicolor/\${s}x\${s}/mimetypes/text-x-apsi.png"
done
[ "$TEMPLATE_MADE" = 1 ] && rm -f "$TEMPLATES/Apsi file.apsi"
command -v update-mime-database >/dev/null 2>&1 && update-mime-database "$SHARE/mime" >/dev/null 2>&1 || true
rm -rf "$PREFIX"
[ "\$1" = "--quiet" ] || echo "Apsi removed (your settings in ~/.apsi are kept)."
UNI
chmod 755 "$PREFIX/uninstall.sh"

echo ""
"$BIN/apsi" version || true
case ":$PATH:" in
  *":$BIN:"*) say "ready — try: apsi   (REPL)   or   apsi run $PREFIX/examples/01_hello.apsi"
              say "terminals and editors that were already open keep the old PATH: open a new one" ;;
  *) say "$BIN is not on your PATH yet. Make it permanent (then open a new terminal):"
     echo "    echo 'export PATH=\"$BIN:\$PATH\"' >> ~/.bashrc"
     echo "  or use Apsi in this terminal right away:"
     echo "    export PATH=\"$BIN:\$PATH\"; hash -r" ;;
esac
echo "Uninstall: $PREFIX/uninstall.sh"
exit 0
__APSI_ARCHIVE__
'''

MIME = '''<?xml version="1.0" encoding="UTF-8"?>
<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
  <mime-type type="text/x-apsi">
    <comment>Apsi source file</comment>
    <sub-class-of type="text/plain"/>
    <glob pattern="*.apsi"/>
    <icon name="text-x-apsi"/>
  </mime-type>
</mime-info>
'''


def _add(tar, src, arc, skip=()):
    def filt(ti):
        name = ti.name.replace("\\", "/")
        if "__pycache__" in name or name.endswith((".pyc", ".so", ".dll")) or any(s in name for s in skip):
            return None
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = "root"
        ti.mode = 0o755 if ti.isdir() or (ti.mode & 0o100) else 0o644
        return ti
    tar.add(src, arc, filter=filt)


def build(out_dir):
    sys.path.insert(0, ROOT)
    from apsi import __version__ as v
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
        _add(tar, os.path.join(ROOT, "apsi"), "apsi/lib/apsi")
        _add(tar, os.path.join(ROOT, "editors"), "apsi/editors", skip=("node_modules",))
        _add(tar, os.path.join(ROOT, "docs"), "apsi/docs")
        _add(tar, os.path.join(ROOT, "examples"), "apsi/examples")
        _add(tar, os.path.join(ROOT, "assets", "brand", "png"), "apsi/brand/png")
        for f in ("README.md", "CHANGELOG.md"):
            _add(tar, os.path.join(ROOT, f), f"apsi/{f}")
        data = MIME.encode()
        ti = tarfile.TarInfo("apsi/share/apsi.xml")
        ti.size, ti.mode = len(data), 0o644
        tar.addfile(ti, io.BytesIO(data))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ApsiSetup-{v}-linux.sh")
    with open(path, "wb") as fh:
        fh.write(HEADER.replace("@V@", v).encode("utf-8"))
        fh.write(buf.getvalue())
    os.chmod(path, 0o755)
    return path


if __name__ == "__main__":
    print(build(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "dist", "linux"))))
