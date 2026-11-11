#!/bin/sh
# The Linux release zip, offline, from a clean copy of the committed tree:
#   dist/apsi-<v>-linux.zip  = ApsiSetup-<v>-linux.sh, apsi_<v>_all.deb, the wheel, source/apsi-<v>-source.zip,
#                              gpu-tuner/ (Apply / Restore for Windows and Linux), README, INSTALL, CHANGELOG, SHA256SUMS
# Usage (repository root): sh packaging/make_release_zip.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
V=$(cd "$ROOT" && python3 -c 'import apsi; print(apsi.__version__)')
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
git -C "$ROOT" archive HEAD | tar -x -C "$W"
cd "$W"
SKIP_ELECTRON=1 SKIP_WINDOWS=1 sh packaging/build_all.sh >/dev/null
R="dist/apsi-$V-linux"
mkdir -p "$R/source" "$R/gpu-tuner"
python3 packaging/make_linux_installer.py "$R" >/dev/null
git -C "$ROOT" archive --format=zip --prefix="apsi-$V/" -o "$W/$R/source/apsi-$V-source.zip" HEAD
python3 tools/gpu-tuner/build.py "$R/gpu-tuner" >/dev/null
cp "dist/apsi-$V-py3-none-any.whl" "dist/apsi_${V}_all.deb" LICENSE CHANGELOG.md docs/TROUBLESHOOTING.md "$R/"
cp docs/INSTALL-LINUX.md "$R/INSTALL-LINUX.md"
sed "s/@V@/$V/g" packaging/linux-README.txt > "$R/README.txt"
(cd "$R" && sha256sum "ApsiSetup-$V-linux.sh" "apsi_${V}_all.deb" "apsi-$V-py3-none-any.whl" "source/apsi-$V-source.zip" \
   gpu-tuner/* > SHA256SUMS.txt)
mkdir -p "$ROOT/dist"
(cd dist && rm -f "apsi-$V-linux.zip" && zip -qr "$ROOT/dist/apsi-$V-linux.zip" "apsi-$V-linux")
cp "$R/source/apsi-$V-source.zip" "$ROOT/dist/"
ls -l "$ROOT/dist/apsi-$V-linux.zip" "$ROOT/dist/apsi-$V-source.zip"
