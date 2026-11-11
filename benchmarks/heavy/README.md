# Heavy CPU benchmarks

Eight demanding programs, each written the same way — same algorithm, sizes and random-number generator — in Apsi,
C, C++, Rust, Go, Java, JavaScript (Node.js), C# (.NET), Julia and Mojo:

| Program | What it stresses |
|---|---|
| `nbody-dense 2000x10` | all-pairs gravity, 40 M interactions with square roots and divisions |
| `jacobi-2d 1024x200` | 2-D heat-equation stencil, memory bandwidth |
| `matmul 1024` | dense matrix product, 2.1 GFLOP (loops in every language; `run.py` sets `APSI_IDIOMS=0` so Apsi runs the loops too — with the default, Apsi runs them on BLAS, see [`../matmul`](../matmul/README.md)) |
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
`-C target-cpu=native`, `go build`, `mojo build`, `javac`, `dotnet build -c Release`, `julia -O3`; JIT languages are
timed after a warm-up run and every result is kept, so no compiler can drop a run), runs it, checks that all languages printed the same checksum,
and writes `results.json` and `results.md`. Mojo is installed with `pip install mojo`.

## Results (November 2026, AMD EPYC 7763 GitHub runner, one core, best of 3, ms)

| Program | Apsi 2.4.4 | C | C++ | Rust | Go | Java | JavaScript | C# | Julia |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `nbody-dense 2000x10` | 47 | **44** | **44** | 182 | 318 | 76 | 184 | 186 | 182 |
| `jacobi-2d 1024x200` | **151** | 164 | 159 | 172 | 705 | 402 | 978 | 769 | 152 |
| `matmul 1024` | 199 | 195 | 186 | 190 | 480 | 418 | 1446 | 1373 | **165** |
| `lu-pivot 800` | 41 | **41** | 41 | 42 | 118 | 77 | 261 | 177 | 42 |
| `ising 512x50` | **82** | 91 | 90 | 84 | 146 | 131 | 281 | 132 | 90 |
| `levenshtein 10k` | **195** | 812 | 306 | 269 | 1742 | 479 | 1046 | 1085 | 923 |
| `ode-rk4 20M` | 355 | 289 | **289** | 355 | 355 | 363 | 355 | 355 | 355 |
| `kmeans 200k k16 x20` | **51** | 61 | 61 | 65 | 214 | 197 | 276 | 208 | 66 |
| time relative to C (geometric mean) | 0.83× | 1.00× | 0.88× | 1.07× | 2.75× | 1.66× | 3.56× | 2.89× | 1.22× |

Every language printed the same checksum for every program. Apsi 2.0.0 on the same runner: see the website's benchmark page.
