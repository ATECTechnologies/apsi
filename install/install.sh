#!/usr/bin/env sh
# Apsi installer for Linux (and macOS)
#   sh install/install.sh            install for the current user (no sudo needed)
#   sh install/install.sh --uninstall
#
# installs into ~/.apsi (its own virtual environment) and creates the command ~/.local/bin/apsi
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${APSI_HOME:-$HOME/.apsi}"
BIN="${APSI_BIN:-$HOME/.local/bin}"

if [ "$1" = "--uninstall" ]; then
  rm -rf "$PREFIX" "$BIN/apsi"
  echo "Apsi uninstalled."
  exit 0
fi

PY=""
for c in python3.14 python3.13 python3.12 python3.11 python3.10 python3.9 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
      PY="$c"; break
    fi
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.9 or newer is required  (Ubuntu/Debian: sudo apt install python3 python3-venv)"
  exit 1
fi
echo "using $($PY --version)"

if ! "$PY" -m venv "$PREFIX/venv" 2>/dev/null; then
  echo "could not create a venv  (Ubuntu/Debian: sudo apt install python3-venv)"
  exit 1
fi
"$PREFIX/venv/bin/python" -m pip install --quiet --upgrade pip
"$PREFIX/venv/bin/python" -m pip install --quiet "$ROOT"

mkdir -p "$BIN"
cat > "$BIN/apsi" <<SH
#!/usr/bin/env sh
exec "$PREFIX/venv/bin/python" -m apsi "\$@"
SH
chmod +x "$BIN/apsi"

echo ""
"$BIN/apsi" version
case ":$PATH:" in
  *":$BIN:"*) echo "ready: apsi run hello.apsi" ;;
  *) echo "add this line to ~/.bashrc or ~/.zshrc, then open a new terminal:"
     echo "  export PATH=\"$BIN:\$PATH\"" ;;
esac
