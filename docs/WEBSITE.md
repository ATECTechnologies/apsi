# Publishing to apsi.dev — libraries, releases and benchmarks

Everything on the website can be added and removed **from the command line on any computer**, or with the server
command when you are on the server itself.

## Libraries (every member)

1. On the site: **Settings → Developer → New token**, permissions `publish:new`, `publish:update` (and `yank` if
   you want to remove versions later).
2. Once per computer: `apsi lib login --token apsi_…`
3. In your library folder:

```bash
apsi pack                       # checks the sources, builds dist/NAME-VERSION.apsil
apsi publish                    # uploads it (large files go up in resumable 32 MB pieces)
apsi search physics             # find libraries
apsi install geom-tools         # use one: then `use geom_tools`
```

Removing a version:

```bash
apsi yank NAME 1.2.0            # hidden from new installs; projects that use it keep working (undo: --undo)
apsi unpublish NAME 1.2.0       # deleted for good — allowed in the first 72 hours or while nobody downloaded it
```

Versions can never be replaced: publish `1.2.1` instead. That guarantee is what keeps everyone's builds
reproducible.

## Apsi releases (administrators)

Create a token with the `release` permission (only administrators see it), store it with `apsi lib login`, then:

```bash
apsi release upload ApsiSetup-2.0.0-x64.exe apsi-2.0.0-linux.zip apsi-2.0.0-source.zip --version 2.0.0
apsi release upload apsi-1.10.0-linux.zip --version 1.10.0 --date 2026-11-08T09:00:00Z   # an older version
apsi release list               # ids, sizes, downloads
apsi release hide 7             # / unhide 7
apsi release delete 7           # removes the file for good
```

The platform is recognised from the file name (`.exe` → Windows, `linux`/`.deb`/`.sh` → Linux, `source` →
source code); pass `--platform` to override and `--title`/`--notes NOTES.md` for the download page. The newest
stable version is what `/download` shows; every version stays on `/download/all`.

On the server itself the same works without a token:

```bash
node scripts/admin.js release add FILE... --version 2.0.0
node scripts/admin.js release list
node scripts/admin.js release delete 7 8 9
node scripts/admin.js release delete-version 1.10.0       # every file of a version
```

## Benchmarks (administrators)

```bash
python3 benchmarks/heavy/run.py --publish 2.0.0                       # numbers appear on /benchmarks live
python3 benchmarks/heavy/run.py --apsi-root ../apsi-1.10.0 --only Apsi --label "Apsi 1.10.0" \
        --merge benchmarks/heavy/results.json --publish 1.10.0        # measure an older version
apsi release bench results.json --version 2.0.0                       # publish an existing results file
apsi release bench --delete 1.10.0                                    # remove a version's results
```

The benchmarks page shows the newest version first and lets visitors pick any other version and compare two
versions side by side.
