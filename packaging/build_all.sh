#!/usr/bin/env sh
# Build every Apsi release artefact into dist/:
#   apsi-<v>-py3-none-any.whl     pip wheel (all platforms)
#   apsi_<v>_all.deb              Debian / Ubuntu package (uses apt's python3 + python3-numpy)
#   apsi-<v>-windows-x64.zip      portable Windows build with its own Python 3.12 + numpy
#   ApsiSetup-<v>-x64.exe         Windows 10/11 installer (needs NSIS: apt install nsis)
#   apsi-language-<v>.vsix        VS Code extension (needs Node.js)
#
# Usage: sh packaging/build_all.sh [path/to/cpython-3.12-x86_64-pc-windows-msvc-install_only.tar.gz]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
V="$(python3 -c 'import apsi; print(apsi.__version__)')"
DIST="$ROOT/dist"
BUILD="$ROOT/build"
WINPY_TGZ="${1:-$BUILD/cpython-windows.tar.gz}"
WINPY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20250612/cpython-3.12.11+20250612-x86_64-pc-windows-msvc-install_only.tar.gz"
rm -rf "$BUILD/wheel" "$BUILD/deb" "$BUILD/windows"
python3 -c "from apsi.security import write_manifest; write_manifest()"
mkdir -p "$DIST" "$BUILD"
echo "== Apsi $V"

# ---------------------------------------------------------------- wheel
# standard library only: works offline and with any setuptools version
python3 packaging/make_wheel.py "$BUILD/wheel" >/dev/null
cp "$BUILD/wheel/apsi-$V-py3-none-any.whl" "$DIST/"
echo "built dist/apsi-$V-py3-none-any.whl"

# ---------------------------------------------------------------- .deb
P="$BUILD/deb/apsi_${V}_all"
mkdir -p "$P/DEBIAN" "$P/usr/bin" "$P/usr/lib/python3/dist-packages/apsi" "$P/usr/share/doc/apsi" "$P/usr/share/apsi"
cp apsi/*.py apsi/MANIFEST.json "$P/usr/lib/python3/dist-packages/apsi/"
for sub in apsi/*/; do                                   # subpackages (apsi/native), without caches
  case "$sub" in *__pycache__*) continue ;; esac
  mkdir -p "$P/usr/lib/python3/dist-packages/$sub"
  cp "$sub"*.py "$P/usr/lib/python3/dist-packages/$sub"
done
cp README.md CHANGELOG.md "$P/usr/share/doc/apsi/"
cp -r examples docs "$P/usr/share/apsi/"          # /usr/share/doc may be stripped on minimal systems
rm -f "$P"/usr/share/apsi/examples/native/*.so
cp editors/vscode/syntaxes/apsi.tmLanguage.json "$P/usr/share/apsi/"
[ -f "$DIST/apsi-language-$V.vsix" ] && cp "$DIST/apsi-language-$V.vsix" "$P/usr/share/apsi/" || true
cat > "$P/usr/bin/apsi" <<'SH'
#!/bin/sh
exec /usr/bin/python3 -m apsi "$@"
SH
chmod 755 "$P/usr/bin/apsi"
SIZE="$(du -sk "$P/usr" | cut -f1)"
cat > "$P/DEBIAN/control" <<CTL
Package: apsi
Version: $V
Section: devel
Priority: optional
Architecture: all
Depends: python3 (>= 3.12), python3-numpy (>= 1:1.26)
Recommends: python3-llvmlite
Suggests: code
Installed-Size: $SIZE
Maintainer: Apsi maintainers <apsi@localhost>
Homepage: https://example.invalid/apsi
Description: Apsi programming language for AI, physics and servers
 Apsi is a natively compiled (LLVM) language with physical units, tensor shape
 checking, automatic differentiation (any order), FP64-FP4 numeric formats,
 streaming data pipelines and a hardened high-concurrency web server.
CTL
cat > "$P/DEBIAN/postinst" <<'SH'
#!/bin/sh
set -e
python3 -m compileall -q /usr/lib/python3/dist-packages/apsi >/dev/null 2>&1 || true
SH
cat > "$P/DEBIAN/prerm" <<'SH'
#!/bin/sh
set -e
rm -rf /usr/lib/python3/dist-packages/apsi/__pycache__
SH
chmod 755 "$P/DEBIAN/postinst" "$P/DEBIAN/prerm"
find "$P" -type d -exec chmod 755 {} +
dpkg-deb --root-owner-group --build "$P" "$DIST/apsi_${V}_all.deb" >/dev/null
echo "built dist/apsi_${V}_all.deb"

if [ "${SKIP_WINDOWS:-0}" != "1" ]; then
# ---------------------------------------------------------------- Windows payload (shared with the Electron installer)
W="$BUILD/windows/Apsi"
mkdir -p "$BUILD/windows"
if [ -f "$WINPY_TGZ" ]; then RT="--runtime-archive $WINPY_TGZ"; else RT=""; fi
python3 packaging/build_payload.py --platform windows --out "$W" $RT
crlf() { sed 's/\r$//; s/$/\r/' > "$1"; }
crlf "$W/apsi.cmd" <<'CMD'
@echo off
rem Apsi launcher (Windows 10/11)
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONSAFEPATH=1"
"%~dp0runtime\python\python.exe" -I -X utf8 -m apsi %*
endlocal & exit /b %ERRORLEVEL%
CMD
crlf "$W/apsi-repl.cmd" <<'CMD'
@echo off
title Apsi REPL
chcp 65001 >nul
set "PYTHONUTF8=1"
"%~dp0runtime\python\python.exe" -I -X utf8 -m apsi repl
CMD
crlf "$W/apsi-run.cmd" <<'CMD'
@echo off
rem Used by the .apsi file association: runs (or tests) a file and keeps the window open.
chcp 65001 >nul
set "PYTHONUTF8=1"
if "%~1"=="--test" (
    "%~dp0runtime\python\python.exe" -I -X utf8 -m apsi test "%~2"
) else (
    pushd "%~dp1"
    "%~dp0runtime\python\python.exe" -I -X utf8 -m apsi run "%~1"
    popd
)
echo.
pause
CMD
(cd "$BUILD/windows" && rm -f "$DIST/apsi-$V-windows-x64.zip" && zip -qr "$DIST/apsi-$V-windows-x64.zip" Apsi)
echo "built dist/apsi-$V-windows-x64.zip"

# ---------------------------------------------------------------- Windows installer (.exe)
if command -v makensis >/dev/null 2>&1; then
  makensis -V2 -DVERSION="$V" -DPAYLOAD="$W" -DOUTFILE="$DIST/ApsiSetup-$V-x64.exe" packaging/windows/apsi.nsi
  echo "built dist/ApsiSetup-$V-x64.exe"
else
  echo "NSIS not found: skipped ApsiSetup-$V-x64.exe (apt install nsis)"
fi

# ---------------------------------------------------------------- Electron installer (modern wizard)
if command -v npm >/dev/null 2>&1 && [ "${SKIP_ELECTRON:-0}" != "1" ]; then
  rm -rf installer/payload && cp -r "$W" installer/payload
  (cd installer && npm ci --no-audit --no-fund && npm run build:ui && npx electron-builder --win portable --x64 --publish never) \
    && cp installer/release/ApsiSetup-*-x64.exe "$DIST/ApsiSetup-$V-x64-wizard.exe" \
    && echo "built dist/ApsiSetup-$V-x64-wizard.exe" || echo "Electron installer build failed (needs wine on Linux; build on Windows with: cd installer && npm run dist:win)"
  rm -rf installer/payload
fi

fi  # SKIP_WINDOWS

# ---------------------------------------------------------------- checksums
(cd "$DIST" && sha256sum $(ls apsi-$V-py3-none-any.whl apsi_${V}_all.deb apsi-$V-windows-x64.zip ApsiSetup-$V-x64.exe \
   apsi-language-$V.vsix ApsiSetup-$V-x64-wizard.exe 2>/dev/null) > SHA256SUMS.txt)
echo "== done"; ls -lh "$DIST"
