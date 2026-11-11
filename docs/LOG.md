# Logging (`use log`)

`use log` writes what a program does to the terminal and to `.log` files: one line per record, with the time,
the level, the part of the program it comes from, the message and any fields.

```
use log
log.to_file("logs/app.log")                    # every record also goes to logs/app.log
log.info("server started", port=8080)
log.warn("disk almost full", free=3.2)
let db = log.named("db", table="users")        # its name on each line, and fields added to each record
db.error("query failed", err=e)
with log.timer("load model"):
    let lm = hub.model("Qwen/Qwen2.5-0.5B-Instruct")   # "load model took 0.86 s" (or "failed after …")
```

```
2026-11-11 08:30:12.345  INFO   app  server started  port=8080
2026-11-11 08:30:12.346  WARN   app  disk almost full  free=3.2
2026-11-11 08:30:12.351  ERROR  db  query failed  table=users  err="TimeoutError: no answer in 5 s"
```

On a terminal the level is coloured (grey debug, cyan info, yellow warn, red error). A file gets plain text, so it
reads the same in any editor, `grep` and `tail`. Fields with spaces or quotes are written in double quotes.

`use log` names the module `log`, which hides the math function `log()` in that file; `use log as lg` keeps both.

## Functions

| | |
|---|---|
| `log.debug / info / warn / error / critical(message, **fields)` | one record; several message parts are joined with spaces |
| `log.named(name, **fields)` | a logger for one part of a program (`db`, `web`, `train`); `.named()` again gives `db.pool` |
| `log.timer(what, level="info")` | `with log.timer("step"):` records how long the block took, or that it failed and after how long |
| `log.to_file(path, format="text", rotate="10 MB", keep=5, level="debug")` | also write records to a file; returns the path |
| `log.level(name)` | the terminal's level: `"debug"`, `"info"` (default; `APSI_LOG_LEVEL` sets it), `"warn"`, `"error"` |
| `log.console(on=true, color=)` | the terminal on or off; `color=true/false` overrides the automatic choice |
| `log.read(path, level=, grep=, last=, name=, older=false)` | records back as `{"time", "level", "name", "message", "fields"}` |
| `log.close()` | close the files (done by itself at exit) |

**Files.** `format="json"` writes JSON lines (one object per record: `time`, `level`, `name`, `message` and the
fields) for tools that collect logs. `rotate="10 MB"` starts a new file at that size and keeps `keep` older ones as
`app.1.log` (newest) … `app.5.log`; `rotate="daily"` keeps one file per day as `app.2026-11-10.log`;
`rotate="none"` never rotates. A file's own `level` lets it keep debug records the terminal does not show.
Writing a file needs `--allow-write` for its folder, reading one `--allow-read`, as for any file.

**Threads.** Records from several threads never mix within a line; each line is written and flushed at once, so a
crash loses nothing that was logged.

## `apsi log`

```
apsi log logs/app.log                       # the file, coloured
apsi log logs/app.log --level warn --older  # warnings and errors, rotated files included, oldest first
apsi log logs/app.log --name db --grep timeout --tail 20
apsi log logs/app.log -f                    # keep showing new records (also across a rotation); Ctrl+C stops
```

It reads text and JSON-lines files alike.
