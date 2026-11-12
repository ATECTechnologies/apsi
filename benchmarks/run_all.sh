#!/bin/sh
# Build and run the same benchmarks in Apsi, C, C++, Rust and Go (best of 5 runs each; times in ms).
# Needs: apsi, gcc, g++, clang (optional), rustc, go.   Usage: sh benchmarks/run_all.sh
set -e
cd "$(dirname "$0")"
sed 's/^@native$/@native("fast")/' bench.apsi > /tmp/apsi_bench_fast.apsi
echo "== Apsi @native";            apsi bench.apsi | grep " ms"
echo "== Apsi @native(\"fast\")";  apsi /tmp/apsi_bench_fast.apsi | grep " ms"
gcc -O3 -march=native -o /tmp/b_c bench.c -lm && echo "== C (gcc)" && /tmp/b_c | grep " ms"
g++ -O3 -march=native -o /tmp/b_cpp bench.cpp && echo "== C++ (g++)" && /tmp/b_cpp | grep " ms"
if command -v clang >/dev/null; then clang -O3 -march=native -o /tmp/b_clang bench.c -lm && echo "== C (clang)" && /tmp/b_clang | grep " ms"; fi
rustc -C opt-level=3 -C target-cpu=native -o /tmp/b_rs bench.rs && echo "== Rust" && /tmp/b_rs | grep " ms"
go build -o /tmp/b_go bench.go && echo "== Go" && /tmp/b_go | grep " ms"
