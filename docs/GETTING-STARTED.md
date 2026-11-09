# Getting started

This guide takes you from installation to a first program, a first trained model and a first library in about ten minutes.

## 1. Install

Apsi 2.4.4 runs on **Windows 10 or newer** and **Linux** (x86-64) and needs **Python 3.12 or newer** as its host runtime. The installers check for Python and offer to install it.

- Windows: download the installer from the download page and run it. Details: [Install on Windows](INSTALL.md).
- Linux: unpack the Linux package and run `sh install.sh`, or install the `.deb`. Details: [Install on Linux](INSTALL-LINUX.md).

Check the installation:

```bash
apsi version
apsi doctor          # checks Python, the native compiler and your hardware
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

## 3. Make it fast

Mark a function `@native` and it is compiled to machine code for your processor the first time it is called. The machine code is cached on disk, so later runs start immediately.

```apsi
@native
fn mandel(n: Int, maxit: Int) -> Int:
    var count = 0
    for y in 0..n:
        for x in 0..n:
            let cr = 2.0 * x / n - 1.5
            let ci = 2.0 * y / n - 1.0
            var zr = 0.0
            var zi = 0.0
            var it = 0
            while it < maxit and zr * zr + zi * zi <= 4.0:
                let t = zr * zr - zi * zi + cr
                zi = 2.0 * zr * zi + ci
                zr = t
                it += 1
            if it == maxit:
                count += 1
    return count

print(mandel(1000, 50))
```

See the [native compiler guide](NATIVE.md) for what can be compiled and how to use all cores with `parallel for`.

## 4. Train a model

```apsi
use torch

let model = torch.Sequential(torch.Linear(2, 32), torch.ReLU(), torch.Linear(32, 1))
var params = model.init(0)
let opt = torch.AdamW(lr=1e-2)

let x = randn(256, 2, seed=1)
let y = sum(x * x, axis=1, keepdims=true)

for step in 0..500:
    fn loss_of(p) = torch.mse_loss(model(p, x), y)
    let loss, grads = value_and_grad(loss_of)(params)
    params = opt.step(params, grads)
    if step % 100 == 0:
        print(f"step {step}: loss {loss}")
```

Already have PyTorch code? Convert it:

```bash
apsi convert train.py -o train.apsi
apsi run train.apsi
```

See [From Python and PyTorch](PYTHON.md).

## 5. Use and publish libraries

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

## 6. Where to go next

- [What's new in 2.4.4](WHATS-NEW.md) — exact Python conversion, real time, memory, tokens, tables, databases.
- [From Python](PYTHON.md) — `apsi convert` an existing program and run it unchanged.
- [Language reference](LANGUAGE.md) — syntax, types, units, modules.
- [AI and machine learning](AI.md) and [Reinforcement learning](RL.md).
- [Science, physics and medicine](SCIENCE.md).
- [Troubleshooting](TROUBLESHOOTING.md) if something does not work.
