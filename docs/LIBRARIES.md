# Libraries (.apsil)

Apsi libraries are distributed as **`.apsil`** files through the official registry. One command installs the newest version, every file is verified, and a library is used exactly like a standard module.

## Installing

```bash
apsi install NAME                   # newest stable version
apsi install NAME@1.4.2             # exactly this version
apsi install NAME@^1.4              # newest 1.x that is at least 1.4
apsi install NAME@~1.4.2            # newest 1.4.x that is at least 1.4.2
apsi install "NAME>=1.2,<2"         # any range
apsi install ./dist/NAME-1.0.0.apsil   # a local file
apsi install                        # everything in [dependencies] of apsi.toml
apsi install -U NAME                # upgrade to the newest matching version
apsi install -g NAME                # for your user account instead of the project
apsi uninstall NAME
apsi lib list                       # installed libraries
apsi search WORDS                   # search the registry
apsi lib info NAME                  # versions, sizes and dates
```

Inside a project (a folder with `apsi.toml`), libraries go into `apsi_packages/` and a lock file, `apsi-packages.lock`, records the exact versions and file hashes. Outside a project, or with `-g`, they go into your user folder (`~/.apsi/packages`, or `%USERPROFILE%\.apsi\packages` on Windows). Project libraries take precedence.

Dependencies of a library are installed automatically. If two libraries need incompatible versions of the same dependency, installation stops with a clear message instead of picking one silently.

Pre-release versions (`2.0.0-rc.1`) are installed only when a requirement names a pre-release.

### Version requirements

| Requirement | Allows |
|---|---|
| `*` or nothing | the newest stable version |
| `1.4.2` or `==1.4.2` | exactly 1.4.2 |
| `1` | any 1.x.y |
| `1.4` | any 1.4.y |
| `^1.4.2` | ≥ 1.4.2 and < 2.0.0 (for 0.x: `^0.3.1` allows < 0.4.0) |
| `~1.4.2` | ≥ 1.4.2 and < 1.5.0 |
| `>=1.2,<2` | a combination of `>`, `>=`, `<`, `<=`, `==`, `!=` |

## Using a library

```apsi
use physics_units          # dashes in the library name become underscores
print(physics_units.version)

use transformer_kit as tk
let model = tk.gpt(layers=12, width=768)
```

`use NAME` makes the library’s public names available directly and as `NAME.member`. Names starting with `_` are private. The checker loads the library before your program runs, so unit and type errors across library boundaries are reported as usual.

## What is verified on installation

1. The SHA-256 of the downloaded file must match the registry’s record.
2. The archive must contain exactly the files listed in its manifest, with the listed sizes; paths may not leave the package folder; encrypted entries are refused.
3. Every file’s SHA-256 must match the manifest.
4. The library must accept your Apsi version (`apsi = ">=2.0.0"` in its manifest).
5. **Apsi Guard** scans the code. Critical findings remove the library again; high findings are shown.

## Creating a library

```bash
apsi new my-physics --template lib
```

The `[package]` table of `apsi.toml` describes the library:

```toml
[project]
name = "my-physics"              # lowercase letters, digits, "-" or "_"; 2–64 characters; starts with a letter
version = "1.0.0"                # semantic versioning
description = "Orbital mechanics helpers"

[package]
entry = "src/main.apsi"          # what `use my_physics` loads
summary = "Orbital mechanics helpers"
license = "MIT"
authors = ["Your Name"]
keywords = ["physics", "orbits"]
homepage = "https://example.org"
repository = "https://github.com/you/my-physics"
apsi = ">=2.0.0"                 # Apsi versions the library works with
include = ["src/**", "README*", "LICENSE*", "examples/**", "data/**"]
exclude = ["data/raw/**"]

[dependencies]
linalg-extra = "^1.2"
```

Libraries may be of any size and contain data files (weights, tables, datasets) next to the code; files larger than 2 GB are stored with ZIP64.

## Building: `apsi pack`

```bash
apsi pack                  # writes dist/my-physics-1.0.0.apsil
apsi pack --out build/     # another folder
```

`apsi pack` first checks every `.apsi` file (syntax, names, types and units) and stops if anything is wrong. It then writes the archive deterministically — sorted entries, fixed timestamps — so the same sources always give the same bytes and the same SHA-256.

## Publishing: `apsi publish`

Libraries are published only from the command line, with an API token:

1. On the website, open **Settings → Developer** and create a token. Give it a name, choose its permissions (`publish:new` to create new libraries, `publish:update` to add versions, `yank` to hide versions) and optionally limit it to one library. Copy it — it is shown once.
2. Store the token (the file is readable only by you). The official registry, `https://apsi.dev`, is used by
   default; `apsi lib registry URL` (or `APSI_REGISTRY`) switches to a private one.

   ```bash
   apsi lib login --token apsi_…
   ```

3. Publish:

   ```bash
   apsi pack
   apsi publish               # uploads the newest file in dist/
   apsi publish dist/my-physics-1.0.0.apsil --token apsi_…
   ```

In continuous integration, set `APSI_TOKEN` (and `APSI_REGISTRY`) as secret environment variables instead of storing them.

Rules enforced by the registry:

- Files larger than 64 MB are uploaded automatically in resumable 32 MB pieces, so libraries of any size get
  through proxies that limit request sizes; an interrupted upload continues where it stopped.
- A version can be published once and never replaced. Publish a new version to fix a mistake.
- Only the owner can publish new versions of a library. Names that differ only in `-` and `_` are the same name.
- A broken version can be **yanked**: new installations skip it, while lock files that already name it keep working.

## The .apsil format

An `.apsil` file is a ZIP archive (format version 1):

```text
apsil.json        manifest
src/…             the library's source files
README.md, LICENSE, examples/, data/ …
```

`apsil.json` holds `format` (`"apsil"`), `format_version`, `name`, `version`, `summary`, `authors`, `license`, `homepage`, `repository`, `keywords`, `apsi` (supported Apsi versions), `entry`, `dependencies`, `built_with`, and `files` — the path, size and SHA-256 of every other file in the archive.

## Registry API

The registry speaks a small JSON API, so other tools can use it too:

| Request | Answer |
|---|---|
| `GET /api/v1/packages/NAME` | name, summary, owner, latest, downloads, and every version with size, SHA-256, publish time and download URL |
| `GET /api/v1/packages/NAME/VERSION` | one version including its manifest |
| `GET /api/v1/packages/NAME/VERSION/download` | the file (supports `Range`) |
| `GET /api/v1/search?q=WORDS` | matching libraries |
| `POST /api/v1/upload` | multipart form with `file`, `name`, `version`, `sha256` — or JSON `{"upload_id": …}` after a chunked upload; header `Authorization: Bearer TOKEN` |
| `POST /api/v1/uploads` · `PUT /api/v1/uploads/ID` · `GET /api/v1/uploads/ID` | resumable chunked upload: create with `{"purpose": "library", "size", "sha256"}`, send pieces with `Content-Range: bytes START-END/TOTAL`, ask how much arrived |
| `POST /api/v1/packages/NAME/VERSION/yank` | JSON `{"yanked": true}` or `false`; needs the `yank` permission |

## Python packages

`apsi pkg` (described in [Interop](INTEROP.md)) manages *Python* packages used through `use python`. It is separate from Apsi libraries.
