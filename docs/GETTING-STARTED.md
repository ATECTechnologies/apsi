# Getting started

This guide takes you from installation to a first program, a first trained model and a first library in about ten minutes.

## 1. Install

Apsi 1.10.0 runs on **Windows 10 or newer** and **Linux** (x86-64) and needs **Python 3.12 or newer** as its host runtime. The installers check for Python and offer to install it.

- Windows: download the installer from the download page and run it. Details: [Install on Windows](INSTALL.md).
- Linux: unpack the Linux package and run `sh install.sh`, or install the `.deb`. Details: [Install on Linux](INSTALL-LINUX.md).

Check the installation:

```bash
apsi version
apsi doctor          # checks Python and your hardware
```

## 2. Your first program

```bash
apsi new hello
cd hello
apsi run
```

`apsi new` creates a project with `apsi.toml`, `src/main.apsi`, tests and editor settings. Open `src/main.apsi`:

```apsi
fn greet(who: Str) -> Str = f"Hello, {who}!"

let g = 9.81[m/s^2]
let t = 2s
print(greet("Apsi"))
print(f"a stone falls {0.5 * g * t ** 2} in {t}")
```

Physical units are part of the language. Change `t` to `2kg` and run again: the checker reports the unit error before anything runs.


## 3. Train a model

```apsi
use nn

let X, y = spiral_data(n_per_class=100, classes=3, noise=0.2, seed=0)
var params = mlp_init([2, 64, 64, 3], seed=42)

fn forward(params, x):
    var h = x
    for i, layer in enumerate(params):
        h = h @ layer.W + layer.b
        if i < len(params) - 1:
            h = relu(h)
    return h

fn loss_fn(params, x, labels) = cross_entropy(forward(params, x), labels)

let opt = adam(lr=0.01)
for step in 0..400:
    let loss, g = value_and_grad(loss_fn)(params, X, y)
    params = opt.step(params, g)
    if step % 100 == 0:
        print(f"step {step}: loss {loss:.4f}")
print("accuracy:", accuracy(forward(params, X), y))
```

Already have PyTorch code? Convert it:

```bash
apsi convert train.py -o train.apsi
apsi run train.apsi
```

See [From Python and PyTorch](PYTHON.md).

## 4. Use and publish libraries

```bash
apsi install physics-units        # newest version from the registry
```

```apsi
use physics_units
```

To publish your own, create a library project, build it into `dist/` and upload it with an API token:

```bash
apsi new my-lib --template lib
cd my-lib
apsi pack
apsi publish --token apsi_…
```

See [Libraries](LIBRARIES.md).

## 5. Where to go next

- [Language reference](LANGUAGE.md) — syntax, types, units, modules.
- [AI and machine learning](AI.md) and [layers](LAYERS.md).
- [Everything on one page](REFERENCE.md) and [maximum performance](BEST-PERFORMANCE.md).
- [Versions](VERSIONS.md): what 2.0.0 adds.
