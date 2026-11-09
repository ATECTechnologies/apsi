# Heavy CPU benchmarks

Eight demanding programs, each written the same way — same algorithm, sizes and random-number generator — in Apsi,
C, C++, Rust, Go and Mojo:

| Program | What it stresses |
|---|---|
| `nbody-dense 2000x10` | all-pairs gravity, 40 M interactions with square roots and divisions |
| `jacobi-2d 1024x200` | 2-D heat-equation stencil, memory bandwidth |
| `matmul 1024` | dense matrix product, 2.1 GFLOP |
| `lu-pivot 800` | LU factorisation with partial pivoting (log-determinant) |
| `ising 512x50` | Metropolis Monte Carlo, random numbers and unpredictable branches |
| `levenshtein 10k` | 100 M-cell dynamic programme with a three-way minimum |
| `ode-rk4 20M` | Runge–Kutta 4 on the Van der Pol oscillator, long dependency chains |
| `kmeans 200k k16 x20` | clustering, nearest-centre search |

```bash
python3 benchmarks/heavy/run.py            # builds and runs every language that is installed
python3 benchmarks/heavy/run.py --skip Mojo
```

The runner builds each program with release flags for the current machine (`-O3 -march=native`,
`-C target-cpu=native`, `go build`, `mojo build`), runs it, checks that all languages printed the same checksum,
and writes `results.json` and `results.md`. Mojo is installed with `pip install mojo`.
