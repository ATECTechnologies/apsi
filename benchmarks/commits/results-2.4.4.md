# Every commit of the 2.4.4 speed work, on one machine

`python3 benchmarks/commits/measure.py --commits 1187386 7810294 040e971 --rounds 3` on a GitHub Actions runner
(Actions → commits). 1187386 is 2.4.4 before the work on number formats, GPU and outer-loop vectorisation; 7810294 adds
them; 040e971 adds the SIMD lanes written by Apsi itself. Each round runs every tree once, in rotating order; the
table keeps the best of all rounds. Ratio = first tree / this tree (above 1: faster).

Machine: AMD EPYC 7763 64-Core Processor, 4 threads, Python 3.13.18; best of 3 rounds; times in ms (lower is better); ratio = first tree / this tree.

| Row | 1187386 | 7810294 | 040e971 |
|---|---:|---:|---:|
| heavy: nbody-dense 2000x10 | 184.4 (1.00×) | 127.7 (1.44×) | 46.0 (4.01×) |
| heavy: jacobi-2d 1024x200 | 156.4 (1.00×) | 156.9 (1.00×) | 152.5 (1.03×) |
| heavy: matmul 1024 | 189.3 (1.00×) | 182.6 (1.04×) | 179.0 (1.06×) |
| heavy: lu-pivot 800 | 41.4 (1.00×) | 41.2 (1.00×) | 41.4 (1.00×) |
| heavy: ising 512x50 | 81.5 (1.00×) | 81.6 (1.00×) | 81.5 (1.00×) |
| heavy: levenshtein 10k | 194.8 (1.00×) | 195.1 (1.00×) | 195.1 (1.00×) |
| heavy: ode-rk4 20M | 533.4 (1.00×) | 355.0 (1.50×) | 355.0 (1.50×) |
| heavy: kmeans 200k k16 x20 | 65.0 (1.00×) | 51.3 (1.27×) | 51.3 (1.27×) |
| programs: matmul 300, loops (compiled engine) | 38,245.6 (1.00×) | 38,307.7 (1.00×) | 38,642.7 (0.99×) |
| programs: matmul 512, @native loops | 25.9 (1.00×) | 24.8 (1.04×) | 26.0 (0.99×) |
| programs: harmonic sum 50M, @native | 70.0 (1.00×) | 70.0 (1.00×) | 70.1 (1.00×) |
| programs: fib(25), recursive calls | 24.8 (1.00×) | 24.6 (1.01×) | 24.1 (1.03×) |
| programs: 200k objects: create and sum | 431.8 (1.00×) | 434.8 (0.99×) | 435.1 (0.99×) |
| programs: word count, 260k words | 201.7 (1.00×) | 201.2 (1.00×) | 203.7 (0.99×) |
| programs: 200k f-strings and join | 193.9 (1.00×) | 190.4 (1.02×) | 188.2 (1.03×) |
| programs: integer loop, 3M iterations | 168.6 (1.00×) | 172.3 (0.98×) | 171.7 (0.98×) |
| start-up: apsi run hello.apsi | 173.0 (1.00×) | 173.3 (1.00×) | 172.3 (1.00×) |
| import apsi | 133.9 (1.00×) | 133.6 (1.00×) | 133.4 (1.00×) |
| cold heavy.apsi: compile + run, whole process | 7,290.7 (1.00×) | 6,617.3 (1.10×) | 6,031.9 (1.21×) |
| apsi check, 43 examples, empty cache | 9,900.8 (1.00×) | 9,788.9 (1.01×) | 9,628.6 (1.03×) |

Slower in the last tree than in the first: none
