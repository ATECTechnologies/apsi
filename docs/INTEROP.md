# Using Apsi with other languages

Apsi keeps a clear border between its own world and the outside: values that cross it are converted by fixed
rules (quantities with units always travel as `{"value": ..., "unit": "..."}`), and every cross-language
call needs a permission first.

## Python

```apsi
# apsi run app.apsi --allow-python
let np = pyimport("numpy")
let scipy_special = pyimport("scipy.special")
print(scipy_special.gamma(5.5))
```

- Python objects behave like normal values: call methods and read attributes — except attributes that start
  with `_`, which are blocked for safety
- `to_py(x)` converts an Apsi value into a Python/JSON value; `from_py(v)` converts back

### Calling Apsi from Python

```python
import apsi
it = apsi.Interpreter(permissions={"fs"})
env = it.run(open("physics.apsi", encoding="utf-8").read(), "physics.apsi")
kinetic = env.vars["kinetic"]          # Apsi functions can be called from Python directly
```

## PyTorch

```apsi
# apsi run train.apsi --allow-python
let t = to_torch(randn(4, 4), device="cuda")    # Apsi tensor -> torch.Tensor
let back = from_torch(t @ t)                   # torch.Tensor -> Apsi tensor
# whole models: `use layers` then torch_module(model, params) -> torch.nn.Module (see BACKENDS.md)
```

## C / C++ / Rust (FFI)

```c
// vecmath.c   ->   gcc -O3 -shared -fPIC vecmath.c -o libvecmath.so
double dot(const double *a, const double *b, int64_t n);
```

```apsi
# apsi run app.apsi --allow-ffi
let lib = cload("libvecmath.so")
let c_dot = lib.fn("dot", [ptr_f64, ptr_f64, i64], f64)
print(c_dot(a, b, len(a)))
```

- Types: `f64 f32 i32 i64 void cstr ptr_f64 ptr_f32 ptr_i32 ptr_i64`
- Tensors passed as `ptr_*` are **not copied**, so C code can write its results straight into an Apsi tensor
- C++: wrap the functions in `extern "C" { ... }`
- Rust: `#[no_mangle] pub extern "C" fn ...`, built as a `cdylib`
- Quantities must be passed as `x.magnitude`, so you choose the unit on purpose
- `library_path("vecmath")` shows the file `cload` would open (`vecmath.dll` on Windows, `libvecmath.so` on Linux);
  `--allow-ffi=vecmath.so@<sha256>` pins a library to its hash

## JavaScript / Next.js / Java / Go / C# (HTTP)

```bash
apsi serve model.apsi --port 8787 --only predict kinetic
```

- `GET /functions` lists the functions with their parameter types
- `POST /call/<name>` with the body `{"args": [...], "kwargs": {...}}`
- arguments are checked for type, unit and shape on every call; a mismatch returns HTTP 400 with `{"error", "kind"}`
- CORS is enabled and the body size is limited to 10 MB

**Next.js (App Router, server action or route handler):**

```ts
export async function kinetic(massKg: number, speed: number) {
  const r = await fetch("http://127.0.0.1:8787/call/kinetic", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ args: [{ value: massKg, unit: "kg" }, { value: speed, unit: "m/s" }] }),
  });
  const { result, error } = await r.json();
  if (error) throw new Error(error);
  return result; // { value: 9, unit: "J" }
}
```

**Java 11+:**

```java
var client = java.net.http.HttpClient.newHttpClient();
var req = java.net.http.HttpRequest.newBuilder(java.net.URI.create("http://127.0.0.1:8787/call/predict"))
    .header("Content-Type", "application/json")
    .POST(java.net.http.HttpRequest.BodyPublishers.ofString("{\"args\": [[1, 2, 3]]}"))
    .build();
System.out.println(client.send(req, java.net.http.HttpResponse.BodyHandlers.ofString()).body());
```

For a full server with routes, middleware and authentication use `use web` — see [`SERVER.md`](SERVER.md).

## Files and network

- `load_csv`, `save_csv`, `load_json`, `save_json`, `read_text`, `write_text` need the `fs` permission;
  relative paths are resolved against the `.apsi` file that is running
- `http_get(url)` and `http_post_json(url, data)` need the `net` permission
