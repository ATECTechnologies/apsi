#!/usr/bin/env bash
# Apsi 2.6.0 — one-shot setup for Linux (run from the source folder).
#
#   ./setup.sh                      install Apsi for development + run tests
#   ./setup.sh --installer          also build the setup wizard (AppImage) and the Windows payload
#   ./setup.sh --release            build every release file (wheel, .deb, Windows zip, NSIS exe, wizard)
#   ./setup.sh --libs ai            also install libraries: gpu onnx torch transformers sci data | ai ai-gpu science all
#
# Needs: python3 (3.12+) with venv.  --installer/--release: Node.js 20+;  --release: nsis, zip, dpkg-deb.
#   Ubuntu/Debian: sudo apt install python3 python3-venv nodejs npm nsis zip
set -euo pipefail
cd "$(dirname "$0")"

INSTALLER=0; RELEASE=0; TESTS=1; LIBS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --installer) INSTALLER=1 ;;
    --release) RELEASE=1 ;;
    --skip-tests) TESTS=0 ;;
    --libs) shift; IFS=',' read -r -a more <<< "$1"; LIBS+=("${more[@]}") ;;
    -h|--help) sed -n 2,11p "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done

step() { printf '\n\033[36m==> %s\033[0m\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1. $2"; exit 1; }; }

step "Checking Python"
need python3 "sudo apt install python3 python3-venv"
python3 -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12+ required"'

step "Creating a private environment (.venv)"
[ -d .venv ] || python3 -m venv .venv
VPY="$PWD/.venv/bin/python"
"$VPY" -m pip install --disable-pip-version-check -q --upgrade pip
"$VPY" -m pip install --disable-pip-version-check -q --only-binary=:all: -r requirements.txt
"$VPY" -m pip install --disable-pip-version-check -q -e .
"$VPY" -c "from apsi.security import write_manifest; write_manifest()"

if [ ${#LIBS[@]} -gt 0 ]; then
  step "Installing libraries: ${LIBS[*]}"
  "$VPY" -m apsi pkg install "${LIBS[@]}"
fi

if [ "$TESTS" = 1 ]; then
  step "Running the test suite"
  "$VPY" -m apsi test tests/apsi examples --allow-all
  "$VPY" -m unittest discover -s tests -q
  "$VPY" -m apsi scan examples
fi

if [ "$INSTALLER" = 1 ] || [ "$RELEASE" = 1 ]; then
  need node "Install Node.js 20+ (https://nodejs.org or: sudo apt install nodejs npm)"
  step "Building the Linux payload and the setup wizard (AppImage)"
  (cd installer && rm -rf payload && "$VPY" ../packaging/build_payload.py --platform linux --out payload \
     && npm ci --no-audit --no-fund && npm test && npm run build:ui \
     && npx electron-builder --linux AppImage --publish never)
  ls -lh installer/release/*.AppImage
fi

if [ "$RELEASE" = 1 ]; then
  step "Building all release files into dist/"
  PATH="$PWD/.venv/bin:$PATH" sh packaging/build_all.sh
fi

step "Done"
echo "Activate:   source .venv/bin/activate"
echo "Try:        apsi run examples/01_hello.apsi"
echo "Libraries:  apsi pkg list   |   apsi pkg install gpu onnx torch transformers"
echo "Security:   apsi scan <folder>   |   apsi audit <file>   |   apsi config --profile strict"
