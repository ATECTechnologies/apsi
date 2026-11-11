# Configuration, secrets and virtual environments

## `.env` files (`use env`)

```apsi
use env
env.load(".env")                         # KEY=value lines
let db_url = env.get("DATABASE_URL")     # the .env file first, then the real environment
let key = env.secret("OPENAI_API_KEY")   # prints as ******; key.reveal() gives the text
let port = env.int("PORT", 8080)
let debug = env.bool("DEBUG", false)     # true/false, 1/0, yes/no, on/off
let hosts = env.list("ALLOWED_HOSTS")    # comma-separated
env.require("DATABASE_URL", "SECRET_KEY")   # one clear error listing everything that is missing
```

The file format is the one python-dotenv and Docker Compose read: `KEY=value`, `export KEY=value`, single and double
quotes, multi-line double-quoted values, `\n` escapes in double quotes, `# comments`, and `${OTHER}` expansion.

`apsi run app.apsi --env-file .env` loads the file before the program starts.

**Permissions:** values from a `.env` file need only `--allow-read` for that file. Reading the process environment
needs `--allow-env` (or `--allow-env=PORT,DEBUG` for named variables). A default (`env.int("PORT", 8080)`) is used
when the permission is missing, so a program still starts.

**Secrets stay secret:** a `Secret` prints and logs as `******`, is left out of error messages, and compares in
constant time. Pass it to `db.postgres(...)`, `web` clients or `crypto` functions directly; call `.reveal()` only
where the text itself is needed.

## Settings files (`use config`)

```apsi
use config
let cfg = config.load("config/default.toml", "config/local.toml?", env="APP", args=true,
                      defaults=record(train=record(lr=3e-4, epochs=10)), required=["data.path"])
print(cfg.train.lr, config.get(cfg, "server.port", 8080))
config.save("runs/42/config.toml", cfg)          # exactly what this run used
```

- **Formats:** `.toml`, `.json`, `.yaml` / `.yml` (with PyYAML), `.ini` / `.cfg`, `.env`. A name ending in `?` is
  optional (a local file only some machines have).
- **Layers:** later files override earlier ones key by key; tables are merged, not replaced. Then the environment
  (`env="APP"`: `APP_SERVER__PORT=9000` sets `server.port`; needs `--allow-env`), then the command line
  (`args=true`: `--set train.lr=0.01` or `train.lr=0.01` after `--`).
- **Checked:** the types of `defaults` are the schema; a file that gives text where a number belongs is an error
  that names the file and the key. `required=[...]` lists keys that must be set.
- **References:** `"${name}-run"` uses another setting, `"${ENV:HOME}"` an environment variable.
- `config.sources(cfg)` says which file, variable or argument set each key; `config.parse(text, "toml")` reads text.

## Python virtual environments

```bash
apsi run app.apsi --venv .venv          # pyimport() uses the packages installed in that venv
apsi run app.apsi                        # a .venv or venv folder next to the program is found automatically
```

The venv's `site-packages` are put first on the import path for `pyimport`/`pyfrom` and converted programs. The
venv's Python version must match the one Apsi runs on (Apsi says so when it does not).

## Reference (`use env`)

| Function | Returns | Behaviour |
|---|---|---|
| `env.load(path=".env", override=true, export=false)` | names defined | Reads a `.env` file. A missing file is an error. With `override=false`, names already loaded keep their value. With `export=true`, the values are also put into the process environment for child processes. Needs `--allow-read` for the file. |
| `env.get(name, default=none)` | `Str` or default | The loaded `.env` value first, then the process environment. Without a permission to read the environment, returns `default` when one is given and raises a permission error otherwise. |
| `env.has(name)` | `Bool` | Whether the name is set in the loaded files or the environment. |
| `env.require(name, …)` | the values | Raises one error listing **every** missing or empty name, so a deployment shows all problems at once. |
| `env.secret(name, default=none)` | `Secret` or `none` | The value wrapped so it never prints. `.reveal()` returns the text. |
| `env.int(name, default=none)`, `env.float(…)` | number | Converts the value. An empty or missing value gives `default`. A value that is not a number is an error (never silently 0). |
| `env.bool(name, default=false)` | `Bool` | `1`, `true`, `yes`, `on`, `y` (any case) are true; any other set value is false. |
| `env.list(name, sep=",", default=none)` | `List[Str]` | Splits and trims; empty items are dropped. |
| `env.parse(text)` | `Dict` | Parses `.env` text without touching any state — useful in tests. |
| `env.values` | `Dict` | Everything loaded from files so far. |

### File format in detail

```dotenv
# comments start with #
PORT=8080
export DEBUG=true                     # "export " is allowed and ignored
NAME='single quotes: taken literally, no ${EXPANSION}'
GREETING="double quotes: \n becomes a line break"
KEY="-----BEGIN KEY-----
spans several lines
-----END KEY-----"
DATABASE_URL=postgres://${DB_USER}:${DB_PASS}@db/app   # ${NAME} from this file or the environment
EMPTY=
TRAILING=value   # an unquoted value ends before " #"
```

A byte-order mark at the start of the file (common with Windows editors) is ignored.

### Secrets in practice

```apsi
use env
use db
env.load()
let pg = db.postgres(host=env.get("DB_HOST"), user="app", password=env.secret("DB_PASSWORD"))
print(env.secret("DB_PASSWORD"))          # ******
log(f"connecting with {env.secret("DB_PASSWORD")}")   # connecting with ******
```

* Keep `.env` out of version control (add it to `.gitignore`) and commit a `.env.example` with the
  names and harmless values instead.
* On servers, prefer real environment variables or your platform's secret store, and run with
  `--allow-env=NAME1,NAME2` so the program can read only what it needs.
* `Secret` values compare in constant time, so checking an API key with `==` does not leak it through timing.
