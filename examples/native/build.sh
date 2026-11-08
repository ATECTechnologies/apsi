#!/usr/bin/env sh
# compile vecmath.c into libvecmath.so (Linux) / libvecmath.dylib (macOS)
# targets x86-64-v3 (AVX2+FMA), Apsi's hardware baseline, when the compiler supports it
set -e
cd "$(dirname "$0")"
CC="${CC:-cc}"
ARCH=""
case "$(uname -m)" in
  x86_64|amd64) ARCH="-march=x86-64-v3" ;;
esac
if [ "$(uname -s)" = "Darwin" ]; then
  $CC -O3 -shared -fPIC vecmath.c -o libvecmath.dylib
  echo "built $(pwd)/libvecmath.dylib"
else
  $CC -O3 $ARCH -shared -fPIC vecmath.c -o libvecmath.so 2>/dev/null || \
  $CC -O3 -shared -fPIC vecmath.c -o libvecmath.so
  echo "built $(pwd)/libvecmath.so"
fi
