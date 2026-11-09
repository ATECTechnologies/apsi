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

## Python virtual environments

```bash
apsi run app.apsi --venv .venv          # pyimport() uses the packages installed in that venv
apsi run app.apsi                        # a .venv or venv folder next to the program is found automatically
```

The venv's `site-packages` are put first on the import path for `pyimport`/`pyfrom` and converted programs. The
venv's Python version must match the one Apsi runs on (Apsi says so when it does not).
