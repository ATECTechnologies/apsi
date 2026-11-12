#!/usr/bin/env bash
# Build every Linux release file of Apsi into dist/linux/ and zip them:
#   ApsiSetup-<v>-linux-x86_64.AppImage   setup wizard (Express/Custom, security, Apsi Guard)
#   apsi_<v>_all.deb                      Debian/Ubuntu package (system python3 + numpy)
#   apsi-<v>-linux-x86_64.tar.gz          portable: own Python 3.12 + numpy, unpack and run
#   apsi-<v>-py3-none-any.whl             pip package for any Python 3.12+
#   apsi-language-<v>.vsix                VS Code extension
#   SHA256SUMS.txt, INSTALL-LINUX.md
# Needs: python3, node 20+, npm, dpkg-deb, zip.   Usage: bash packaging/build_linux_release.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
V="$(python3 -c 'import apsi; print(apsi.__version__)')"
OUT="$ROOT/dist/linux"
rm -rf "$OUT" && mkdir -p "$OUT"
python3 -c "from apsi.security import write_manifest; write_manifest()"

echo "== VS Code extension"
if [ ! -f "dist/apsi-language-$V.vsix" ]; then
  (cd editors/vscode && npm install --no-audit --no-fund --silent \
     && npx --yes @vscode/vsce@2 package --allow-missing-repository --skip-license -o "$ROOT/dist/apsi-language-$V.vsix") \
     || echo "vsix skipped"
fi

echo "== wheel + .deb"
SKIP_ELECTRON=1 SKIP_WINDOWS=1 sh packaging/build_all.sh >/dev/null
cp "dist/apsi-$V-py3-none-any.whl" "dist/apsi_${V}_all.deb" "$OUT/"
[ -f "dist/apsi-language-$V.vsix" ] && cp "dist/apsi-language-$V.vsix" "$OUT/"

echo "== payload (Python 3.12 + numpy + Apsi)"
P="$ROOT/build/payload-linux"
python3 packaging/build_payload.py --platform linux --out "$P"

echo "== portable tar.gz"
T="$ROOT/build/portable/apsi-$V-linux-x86_64"
rm -rf "$ROOT/build/portable" && mkdir -p "$T/bin"
cp -r "$P"/. "$T/"
rm -f "$T/payload.json" "$T/payload-files.json"
cat > "$T/bin/apsi" <<'SH'
#!/bin/sh
# Apsi launcher (portable). Works from any folder; symlink it into your PATH if you like.
HERE="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
exec "$HERE/runtime/python/bin/python3" -I -X utf8 -m apsi "$@"
SH
cat > "$T/bin/apsi-repl" <<'SH'
#!/bin/sh
exec "$(dirname "$(readlink -f "$0")")/apsi" repl "$@"
SH
cat > "$T/install.sh" <<'SH'
#!/bin/sh
# Optional: put `apsi` on your PATH (~/.local/bin) without moving this folder.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HOME/.local/bin"
ln -sf "$HERE/bin/apsi" "$HOME/.local/bin/apsi"
ln -sf "$HERE/bin/apsi-repl" "$HOME/.local/bin/apsi-repl"
echo "Linked apsi into ~/.local/bin. Open a new terminal (or: export PATH=\"\$HOME/.local/bin:\$PATH\") and run: apsi version"
SH
chmod 755 "$T/bin/apsi" "$T/bin/apsi-repl" "$T/install.sh"
tar -C "$ROOT/build/portable" -czf "$OUT/apsi-$V-linux-x86_64.tar.gz" "apsi-$V-linux-x86_64"

echo "== setup wizard (AppImage)"
(cd installer && rm -rf payload release && cp -r "$P" payload && npm install --no-audit --no-fund --silent \
   && npx vite build >/dev/null && npx electron-builder --linux AppImage --x64 --publish never)
cp installer/release/*.AppImage "$OUT/"
rm -rf installer/payload

cp docs/INSTALL-LINUX.md "$OUT/INSTALL-LINUX.md"
(cd "$OUT" && sha256sum *.AppImage *.deb *.tar.gz *.whl $(ls *.vsix 2>/dev/null) > SHA256SUMS.txt)
(cd "$ROOT/dist" && rm -f "apsi-$V-linux.zip" && mv linux "apsi-$V-linux" && zip -qr "apsi-$V-linux.zip" "apsi-$V-linux" \
   && mv "apsi-$V-linux" linux)
echo "== done: dist/apsi-$V-linux.zip"; ls -lh "$OUT"
