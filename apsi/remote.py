"""Managing what is on the website from any computer: library versions, Apsi releases and benchmark results.

    apsi yank NAME VERSION            hide a version of your library from new installs (projects that use it keep working)
    apsi yank NAME VERSION --undo     show it again
    apsi unpublish NAME VERSION       delete a version for good (allowed in the first 72 hours, or while nobody
                                      downloaded it; administrators always)

    apsi release upload FILE... --version 2.4.4 [--platform auto|windows|linux|macos|source|any]
                        [--title T] [--notes NOTES.md] [--channel stable|beta|nightly] [--date ISO]
    apsi release list [--version V] [--all]
    apsi release delete ID... [--yes]
    apsi release hide ID / apsi release unhide ID
    apsi release bench results.json --version 2.4.4     publish benchmark results (shown on /benchmarks)
    apsi release bench --delete 1.10.0                  remove a version's benchmark results

The release commands need an API token with the `release` permission. Only administrators can create one
(Settings → Developer). Store it once with `apsi lib login --token apsi_…` or pass --token / set APSI_TOKEN.
Large files go up in resumable 32 MB pieces, so a broken connection only repeats the last piece.
"""
import json
import os
import re
import sys
import urllib.parse
import urllib.request

from .apsil import (PackageError, _chunked_upload, _human, _multipart, _send, _sha256_file, registry_url,
                    stored_token)

PLATFORMS = ("windows", "linux", "macos", "source", "any")


def _auth(registry, token):
    url = registry_url(registry)
    tok = token or os.environ.get("APSI_TOKEN") or stored_token(url)
    if not tok:
        raise PackageError("no API token: create one on the site (Settings → Developer), then run "
                           "`apsi lib login --token apsi_…`, set APSI_TOKEN or pass --token")
    return url, {"Authorization": f"Bearer {tok}", "User-Agent": "apsi", "Accept": "application/json"}


def _json(url, auth, method="GET", body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    headers = dict(auth)
    if data is not None:
        headers["Content-Type"] = "application/json"
    return _send(urllib.request.Request(url, data=data, method=method, headers=headers), timeout=timeout)


def guess_platform(filename):
    f = os.path.basename(filename).lower()
    if f.endswith((".exe", ".msi", ".msix")) or "windows" in f or "win64" in f or "-win" in f:
        return "windows"
    if "source" in f or f.endswith((".tar.gz", ".tgz")) and "linux" not in f:
        return "source"
    if f.endswith((".deb", ".rpm", ".appimage", ".sh")) or "linux" in f:
        return "linux"
    if f.endswith((".dmg", ".pkg")) or "macos" in f or "darwin" in f:
        return "macos"
    return "any"


def guess_title(filename, platform):
    f = os.path.basename(filename).lower()
    if f.endswith(".exe") or f.endswith(".msi"):
        return "Windows installer"
    if f.endswith(".deb"):
        return "Debian / Ubuntu package (.deb)"
    if f.endswith(".whl"):
        return "Python wheel (any platform)"
    if f.endswith(".sh"):
        return "Linux installer script"
    if f.endswith(".vsix"):
        return "VS Code extension"
    return {"windows": "Windows package", "linux": "Linux package", "macos": "macOS package",
            "source": "Source code", "any": "Package"}[platform]


# ----------------------------------------------------------------------------------------------- libraries
def yank(name, version, undo=False, token=None, registry=None):
    url, auth = _auth(registry, token)
    res = _json(f"{url}/api/v1/packages/{urllib.parse.quote(name)}/{urllib.parse.quote(version)}/yank", auth, "POST",
                {"yanked": not undo})
    print(f"{name} {version}: {'yanked' if res.get('yanked') else 'available again'} (latest is now {res.get('latest')})")
    return res


def unpublish(name, version, token=None, registry=None):
    url, auth = _auth(registry, token)
    res = _json(f"{url}/api/v1/packages/{urllib.parse.quote(name)}/{urllib.parse.quote(version)}", auth, "DELETE")
    print(f"deleted {name} {version}" + (f" (latest is now {res['latest']})" if res.get("latest") else ""))
    return res


# ----------------------------------------------------------------------------------------------- releases
def release_upload(path, version, platform="auto", title=None, notes=None, channel="stable", date=None, arch="x64",
                   token=None, registry=None, quiet=False):
    if not os.path.isfile(path):
        raise PackageError(f"no such file: {path}")
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", version or ""):
        raise PackageError("--version must look like 2.0.0")
    platform = guess_platform(path) if platform in (None, "", "auto") else platform
    if platform not in PLATFORMS:
        raise PackageError(f"--platform must be one of {', '.join(PLATFORMS)} or auto")
    if notes and os.path.isfile(notes):
        with open(notes, encoding="utf-8") as fh:
            notes = fh.read()
    url, auth = _auth(registry, token)
    me = _json(f"{url}/api/v1/token", auth)
    if "release" not in me.get("scopes", []):
        raise PackageError("this token has no `release` permission (an administrator creates one in Settings → Developer)")
    size = os.path.getsize(path)
    sha = _sha256_file(path)
    meta = {"version": version, "platform": platform, "arch": arch, "channel": channel or "stable",
            "title": title or guess_title(path, platform), "notes": notes or "", "date": date or "", "sha256": sha,
            "filename": os.path.basename(path)}
    if not quiet:
        print(f"uploading {os.path.basename(path)} ({_human(size)}) → {url}  [{version} · {platform}]")
    if size <= 48 << 20:
        body, boundary, total = _multipart(meta, "file", path)
        req = urllib.request.Request(f"{url}/api/v1/releases", data=body, method="POST", headers={
            **auth, "Content-Type": f"multipart/form-data; boundary={boundary}", "Content-Length": str(total)})
        res = _send(req)
    else:
        uid = _chunked_upload(url, auth, path, size, sha, quiet, purpose="release")
        res = _json(f"{url}/api/v1/releases", auth, "POST", {**meta, "upload_id": uid}, timeout=3600)
    if not quiet:
        print(f"published #{res['id']}  {res['filename']}  sha256 {res['sha256'][:16]}…\n  {res['url']}")
    return res


def release_list(version=None, everything=False, token=None, registry=None):
    url, auth = _auth(registry, token)
    q = {"all": "1" if everything else "0"}
    if version:
        q["version"] = version
    res = _json(f"{url}/api/v1/releases?{urllib.parse.urlencode(q)}", auth)
    rows = res.get("releases", [])
    if not rows:
        print("no releases")
    for r in rows:
        flags = (" hidden" if r.get("hidden") else "") + ("" if r.get("channel") == "stable" else f" {r.get('channel')}")
        print(f"#{r['id']:<5} {r['version']:<10} {r['platform']:<8} {_human(int(r['size'])):>9}  {r['downloads']:>6} dl  "
              f"{r['published_at'][:16]}  {r['filename']}{flags}")
    return rows


def release_action(ids, action, token=None, registry=None):
    url, auth = _auth(registry, token)
    for i in ids:
        if action == "delete":
            res = _json(f"{url}/api/v1/releases/{int(i)}", auth, "DELETE")
            print(f"deleted #{i} {res.get('filename', '')}")
        else:
            res = _json(f"{url}/api/v1/releases/{int(i)}/{action}", auth, "POST", {})
            print(f"#{i} {res.get('filename', '')}: {'hidden' if res.get('hidden') else 'visible'}")


def bench_publish(path, version, token=None, registry=None):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data.get("benchmarks"), dict) or not data["benchmarks"]:
        raise PackageError("that file has no benchmark results (expected the results.json written by run.py)")
    url, auth = _auth(registry, token)
    res = _json(f"{url}/api/v1/benchmarks/{urllib.parse.quote(version)}", auth, "PUT", data)
    print(f"benchmarks for {version} published ({res.get('programs')} programs, {res.get('languages')} languages) "
          f"→ {url}/benchmarks?v={version}")


def bench_delete(version, token=None, registry=None):
    url, auth = _auth(registry, token)
    _json(f"{url}/api/v1/benchmarks/{urllib.parse.quote(version)}", auth, "DELETE")
    print(f"benchmarks for {version} removed")


# ----------------------------------------------------------------------------------------------- command line
def _run(fn):
    try:
        fn()
        return 0
    except PackageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted — run the same command again to resume", file=sys.stderr)
        return 130


def cmd_yank(a):
    return _run(lambda: yank(a.name, a.version, a.undo, a.token, a.registry))


def cmd_unpublish(a):
    def go():
        if not a.yes and sys.stdin.isatty():
            if input(f"Delete {a.name} {a.version} for good? Type the name to confirm: ").strip() != a.name:
                raise PackageError("not confirmed")
        unpublish(a.name, a.version, a.token, a.registry)
    return _run(go)


def cmd_release(a):
    def go():
        c = a.release_cmd
        if c == "upload":
            for f in a.files:
                release_upload(f, a.version, a.platform, a.title, a.notes, a.channel, a.date, a.arch, a.token, a.registry)
        elif c == "list":
            release_list(a.version, a.all, a.token, a.registry)
        elif c == "delete":
            if not a.yes and sys.stdin.isatty():
                if input(f"Delete release file(s) {', '.join('#' + str(i) for i in a.ids)} for good? [y/N] ").strip().lower() != "y":
                    raise PackageError("not confirmed")
            release_action(a.ids, "delete", a.token, a.registry)
        elif c in ("hide", "unhide"):
            release_action(a.ids, c, a.token, a.registry)
        elif c == "bench":
            if a.delete:
                bench_delete(a.delete, a.token, a.registry)
            else:
                if not a.file or not a.version:
                    raise PackageError("give the results file and --version")
                bench_publish(a.file, a.version, a.token, a.registry)
    return _run(go)


def add_parser(sub):
    def common(p):
        p.add_argument("--token", help="API token (default: the one stored by `apsi lib login`, or APSI_TOKEN)")
        p.add_argument("--registry", help="site address (default https://apsi.dev)")
    p = sub.add_parser("yank", help="hide a version of your library from new installs (--undo to show it again)")
    p.add_argument("name")
    p.add_argument("version")
    p.add_argument("--undo", action="store_true")
    common(p)
    p.set_defaults(fn=cmd_yank)
    p = sub.add_parser("unpublish", help="delete a version of your library from the registry")
    p.add_argument("name")
    p.add_argument("version")
    p.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    common(p)
    p.set_defaults(fn=cmd_unpublish)
    p = sub.add_parser("release", help="upload, list or delete Apsi release files and benchmark results on the website")
    rs = p.add_subparsers(dest="release_cmd", required=True)
    q = rs.add_parser("upload", help="upload installers or packages (FILE...)")
    q.add_argument("files", nargs="+")
    q.add_argument("--version", required=True)
    q.add_argument("--platform", default="auto", help="auto (from the file name), windows, linux, macos, source, any")
    q.add_argument("--title")
    q.add_argument("--notes", help="release notes: a Markdown file or text")
    q.add_argument("--channel", default="stable", choices=["stable", "beta", "nightly"])
    q.add_argument("--arch", default="x64")
    q.add_argument("--date", help="publication time (ISO), for adding older versions")
    common(q)
    q = rs.add_parser("list", help="list release files with their ids")
    q.add_argument("--version")
    q.add_argument("--all", action="store_true", help="include hidden files")
    common(q)
    for name, hlp in (("delete", "delete release files for good"), ("hide", "hide release files"),
                      ("unhide", "show hidden release files again")):
        q = rs.add_parser(name, help=hlp)
        q.add_argument("ids", nargs="+", type=int)
        if name == "delete":
            q.add_argument("-y", "--yes", action="store_true")
        common(q)
    q = rs.add_parser("bench", help="publish benchmark results (results.json from benchmarks/heavy/run.py)")
    q.add_argument("file", nargs="?")
    q.add_argument("--version")
    q.add_argument("--delete", metavar="VERSION", help="remove the results of a version")
    common(q)
    p.set_defaults(fn=cmd_release)
