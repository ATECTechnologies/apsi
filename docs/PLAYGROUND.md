# Apsi Playground (runs in the browser)

The `playground/` folder is a web page that runs Apsi in the browser with no server doing the work. It uses
Pyodide (CPython compiled to WebAssembly) and numpy entirely on the visitor's machine, so it can be put on the
official website as is.

## Use

```bash
python tools/build_playground.py          # bundle the apsi sources + examples into playground/apsi_bundle.js
python -m http.server -d playground 8000  # open http://localhost:8000
```

To publish, upload `playground/index.html` and `playground/apsi_bundle.js` to any static host such as
GitHub Pages, Netlify, Cloudflare Pages or Vercel.

## Features

- pick an example from the menu (examples that need Python, C or the network are left out)
- run with `Ctrl+Enter`; output, checker errors (with their location) and `test` results are shown
- follows the system's dark/light mode and works on phones
- **safe**: user code runs sandboxed (`permissions = {}`) and cannot read files, call Python/C or use the network

## What runs in the browser

| Works | Not available in the browser |
|---|---|
| the whole language, both engines, the checker and `test` blocks | files, the network, Python libraries and C code (the sandbox has no permissions) |
| units, uncertainty, exact fractions, number formats (FP64 … FP4) | the PyTorch backend and GPUs (`backend="auto"` stays on NumPy) |
| `use nn`, `use layers` (every layer, `dag`, transformers, GNNs, `fit`), `use sci`, `use linalg`, `evolve` | `use web` servers, `use parallel` worker threads |
| autodiff of any order | `apsi build`, the compile cache on disk (the in-memory cache still works) |

Speed: Pyodide runs Python and NumPy as WebAssembly, so programs run slower than on the desktop. The first visit
downloads Pyodide and NumPy (several MB); the browser caches them for later visits.

## Updating

Run `python tools/build_playground.py` again after changing the Apsi sources or the examples; it rebuilds
`playground/apsi_bundle.js` (45 modules and every example that runs in the browser).

## Embedding in another page

Call `apsi.playground.run(code)` through Pyodide; it returns a dict `{output, error, ok, tests}`.
