"""`use hub` — models, datasets and tokenizers from the Hugging Face Hub, without Python libraries.

    use hub
    print(hub.files("sentence-transformers/all-MiniLM-L6-v2"))
    let w = hub.load("sentence-transformers/all-MiniLM-L6-v2")           # safetensors → {name: tensor}
    let cfg = hub.config("sentence-transformers/all-MiniLM-L6-v2")
    let tok = hub.tokenizer("sentence-transformers/all-MiniLM-L6-v2")    # tokenizer.json, natively
    print(tok.encode("Hello world").ids)
    let path = hub.download("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")

Needs --allow-net=huggingface.co (downloads of large files are redirected to the Hub's storage hosts, which
`--allow-net=huggingface.co` covers automatically). Private and gated models: hub.login("hf_…"), the HF_TOKEN
environment variable, or token= on each call. Files are cached in ~/.apsi/hub (APSI_HUB_CACHE) by commit, resumed
after an interruption, and checked against the Hub's SHA-256; with hub.offline(true) or APSI_HUB_OFFLINE=1 only the
cache is used.
"""
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

from .errors import RuntimeErr

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
PARALLEL_MIN = 64 << 20          # files at least this large download over several connections
CHUNK = 16 << 20                 # bytes per piece
CONNECTIONS = max(1, int(os.environ.get("APSI_HUB_CONNECTIONS", "8") or 8))   # at most; added while they help
CONN_BEST = [1]
CHECK_HOURS = float(os.environ.get("APSI_HUB_CHECK_HOURS", "24") or 24)
_STATE = {"token": None, "offline": os.environ.get("APSI_HUB_OFFLINE", "") in ("1", "true", "yes")}
_LOCK = threading.Lock()
_STORAGE_HOSTS = ("huggingface.co", "hf.co", "cdn-lfs.huggingface.co", "cdn-lfs-us-1.huggingface.co",
                  "cdn-lfs-eu-1.huggingface.co", "cas-bridge.xethub.hf.co", "cdn-lfs.hf.co", "cas-server.xethub.hf.co")


class HubError(RuntimeErr):
    kind = "HubError"


def cache_dir():
    d = os.environ.get("APSI_HUB_CACHE") or os.path.join(os.path.expanduser("~"), ".apsi", "hub")
    os.makedirs(d, exist_ok=True)
    return d


def _repo_ok(repo):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?", repo or "") or ".." in repo:
        raise HubError(f"not a Hub repository name: {repo!r} (expected owner/name)")
    return repo


def _file_ok(name):
    parts = (name or "").replace("\\", "/").split("/")
    if not name or any(p in ("", ".", "..") for p in parts):
        raise HubError(f"not a file name inside a repository: {name!r}")
    return "/".join(parts)


def _token(token=None):
    return token or _STATE["token"] or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Hub:
    """One configured client (endpoint, token, permissions)."""

    def __init__(self, interp=None):
        self.interp = interp

    # ---- HTTP
    def _check(self, url):
        if self.interp is None:
            return
        from .security import check_net
        host = urllib.parse.urlparse(url).hostname or ""
        try:
            check_net(self.interp, host, 443, "hub")
        except Exception:
            # storage hosts behind the Hub follow the permission given for huggingface.co
            base = urllib.parse.urlparse(ENDPOINT).hostname
            if host.endswith((".huggingface.co", ".hf.co")) or host in _STORAGE_HOSTS:
                check_net(self.interp, base, 443, "hub")
            else:
                raise

    def _open(self, url, token=None, headers=None, method="GET", follow=True):
        """urlopen with the token only on the Hub's own host (never sent to a redirect target)."""
        hops = 0
        while True:
            self._check(url)
            h = {"User-Agent": "apsi-hub/2.4.4"}
            if headers:
                h.update(headers)
            tok = _token(token)
            if tok and urllib.parse.urlparse(url).hostname == urllib.parse.urlparse(ENDPOINT).hostname:
                h["Authorization"] = f"Bearer {tok}"
            req = urllib.request.Request(url, headers=h, method=method)
            opener = urllib.request.build_opener(_NoRedirect())
            try:
                return opener.open(req, timeout=60)
            except urllib.error.HTTPError as e:
                if e.code in (301, 302, 303, 307, 308) and follow and hops < 8:
                    url = urllib.parse.urljoin(url, e.headers.get("Location"))
                    hops += 1
                    continue
                if e.code in (301, 302, 303, 307, 308):
                    return e
                raise self._http_error(e, url)
            except urllib.error.URLError as e:
                raise HubError(f"cannot reach {urllib.parse.urlparse(url).hostname}: {e.reason}")

    @staticmethod
    def _http_error(e, url):
        what = {401: "needs a token (hub.login(\"hf_…\") or HF_TOKEN), or the repository does not exist",
                403: "the token has no access — for a gated model, accept its terms on the model page first",
                404: "not found (check the repository name, file name and revision)",
                429: "the Hub is rate-limiting requests; wait a minute or log in with a token"}.get(e.code, e.reason)
        return HubError(f"Hugging Face Hub: {what} ({e.code}, {url.split('?')[0]})")

    def _json(self, url, token=None):
        if _STATE["offline"]:
            raise HubError("hub is offline (hub.offline(false) or unset APSI_HUB_OFFLINE)")
        with self._open(url, token) as r:
            return json.loads(r.read().decode())

    # ---- metadata
    def info(self, repo, revision=None, token=None, kind="model"):
        _repo_ok(repo)
        base = {"model": "models", "dataset": "datasets", "space": "spaces"}[kind]
        url = f"{ENDPOINT}/api/{base}/{repo}" + (f"/revision/{urllib.parse.quote(revision, safe='')}" if revision else "")
        return self._json(url, token)

    def files(self, repo, revision=None, token=None, kind="model"):
        if _STATE["offline"]:
            return sorted(self._cached_files(repo, kind))
        return [s["rfilename"] for s in self.info(repo, revision, token, kind).get("siblings", [])]

    def search(self, query="", task=None, limit=20, sort="downloads"):
        q = {"search": query, "limit": int(limit), "sort": sort, "direction": "-1"}
        if task:
            q["pipeline_tag"] = task
        rows = self._json(f"{ENDPOINT}/api/models?{urllib.parse.urlencode(q)}")
        return [{"id": r.get("id") or r.get("modelId"), "task": r.get("pipeline_tag"), "downloads": r.get("downloads"),
                 "likes": r.get("likes")} for r in rows]

    # ---- files
    def _local(self, repo, kind, commit, name):
        return os.path.join(cache_dir(), kind + "s", repo.replace("/", "--"), commit, *name.split("/"))

    def _cached_files(self, repo, kind):
        root = os.path.join(cache_dir(), kind + "s", repo.replace("/", "--"))
        out = set()
        for d, _, fs in os.walk(root):
            for f in fs:
                if not f.endswith((".part", ".lock")):
                    rel = os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/")
                    out.add(rel.split("/", 1)[1] if "/" in rel else rel)
        return out

    def download(self, repo, filename, revision="main", token=None, kind="model", progress=True):
        """One file → its local path (cached by commit; resumed; SHA-256 checked for large files). A connection
        that breaks part way is resumed by itself, several times, before giving up."""
        for attempt in range(8):
            try:
                return self._download_once(repo, filename, revision, token, kind, progress)
            except (HubError, OSError, http.client.HTTPException) as e:
                text = str(e)
                broken = isinstance(e, (OSError, http.client.HTTPException)) or "download ended at" in text or "cannot reach" in text \
                    or "ended early" in text or "timed out" in text.lower()
                if not broken or attempt == 7 or _STATE["offline"]:
                    raise
                if progress:
                    print(f"  (the connection broke; continuing where it stopped, try {attempt + 2} of 8)",
                          flush=True)
                time.sleep(min(30, 2 + 4 * attempt))

    def _download_once(self, repo, filename, revision="main", token=None, kind="model", progress=True):
        _repo_ok(repo)
        name = _file_ok(filename)
        rev = revision or "main"
        refs = os.path.join(cache_dir(), kind + "s", repo.replace("/", "--"), "refs.json")
        if _STATE["offline"]:
            try:
                with open(refs, encoding="utf-8") as fh:
                    commit = json.load(fh).get(rev, rev)
            except OSError:
                commit = rev
            p = self._local(repo, kind, commit, name)
            if os.path.exists(p):
                return p
            raise HubError(f"{repo}/{name} is not in the cache and the hub is offline")
        prefix = {"model": "", "dataset": "datasets/", "space": "spaces/"}[kind]
        url = f"{ENDPOINT}/{prefix}{repo}/resolve/{urllib.parse.quote(rev, safe='')}/{urllib.parse.quote(name)}"
        head = self._open(url, token, method="HEAD", follow=False)
        hdr = head.headers
        commit = hdr.get("X-Repo-Commit") or rev
        etag = (hdr.get("X-Linked-Etag") or hdr.get("ETag") or "").strip('"').removeprefix("W/").strip('"')
        size = int(hdr.get("X-Linked-Size") or hdr.get("Content-Length") or 0)
        target = hdr.get("Location")
        head.close()
        path = self._local(repo, kind, commit, name)
        if os.path.exists(path) and (not size or os.path.getsize(path) == size):
            self._remember_ref(refs, rev, commit)
            return path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        part = path + ".part"
        if size >= PARALLEL_MIN and target and CONNECTIONS > 1:
            with _LOCK:
                self._download_parallel(urllib.parse.urljoin(url, target), url, token, part, size, etag, name,
                                        progress)
                os.replace(part, path)
            self._remember_ref(refs, rev, commit)
            return path
        with _LOCK:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            src = urllib.parse.urljoin(url, target) if target else url
            r = self._open(src, token, headers={"Range": f"bytes={have}-"} if have else None)
            if have and r.status != 206:
                have = 0
            h = hashlib.sha256()
            if have:
                with open(part, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
            t0, done, shown = time.time(), have, 0.0
            tty = getattr(sys.stdout, "isatty", lambda: False)()
            with open(part, "ab" if have else "wb") as out:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if progress and size > (32 << 20) and time.time() - shown > (1 if tty else 30):
                        shown = time.time()
                        rate = (done - have) / max(1e-6, shown - t0)
                        print(("\r" if tty else "") + f"  {name}: {done / 2**20:,.0f} / {size / 2**20:,.0f} MB  "
                              f"{rate / 2**20:,.1f} MB/s", end="" if tty else "\n", flush=True)
            r.close()
            if progress and size > (32 << 20) and tty:
                print()
            if size and os.path.getsize(part) != size:
                raise HubError(f"{name}: download ended at {os.path.getsize(part):,} of {size:,} bytes; "
                               "run again to resume")
            if re.fullmatch(r"[0-9a-f]{64}", etag) and h.hexdigest() != etag:
                os.remove(part)
                raise HubError(f"{name}: the downloaded file does not match the Hub's SHA-256; it was deleted")
            os.replace(part, path)
        self._remember_ref(refs, rev, commit)
        return path

    def _download_parallel(self, src, url, token, part, size, etag, name, progress):
        """A large file over several connections at once (each a byte range of the file), resumable: finished
        pieces are listed next to the .part file. The SHA-256 of the whole file is checked at the end."""
        import concurrent.futures as cf
        ranges = part + ".ranges"
        n = (size + CHUNK - 1) // CHUNK
        done = set()
        try:
            with open(ranges, encoding="utf-8") as fh:
                d = json.load(fh)
            if d.get("size") == size and d.get("chunk") == CHUNK:
                done = {int(i) for i in d.get("done", []) if 0 <= int(i) < n}
        except (OSError, ValueError, TypeError):
            if os.path.exists(part):                   # an earlier one-connection download: its prefix is good
                have = os.path.getsize(part)
                done = {i for i in range(n) if min(size, (i + 1) * CHUNK) <= have}
        mode = "r+b" if os.path.exists(part) else "w+b"
        with open(part, mode) as fh:
            fh.truncate(size)
        lock = threading.Lock()
        state = {"src": src, "got": sum(min(CHUNK, size - i * CHUNK) for i in done)}
        start_got, t0 = state["got"], time.time()

        def save():
            tmp = ranges + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"size": size, "chunk": CHUNK, "done": sorted(done)}, fh)
            os.replace(tmp, ranges)

        def fetch(i):
            a, b = i * CHUNK, min(size, (i + 1) * CHUNK) - 1
            for attempt in range(6):
                pos = a
                try:
                    r = self._open(state["src"], token, headers={"Range": f"bytes={a}-{b}"})
                    if r.status != 206:
                        r.close()
                        raise HubError(f"the server ignored the byte range (status {r.status})")
                    with open(part, "r+b") as out:
                        out.seek(a)
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            out.write(chunk)
                            pos += len(chunk)
                            with lock:
                                state["got"] += len(chunk)
                    r.close()
                    if pos != b + 1:
                        raise HubError(f"piece {i} ended early")
                    with lock:
                        done.add(i)
                        save()
                    return
                except (HubError, OSError, http.client.HTTPException) as e:
                    with lock:
                        state["got"] -= pos - a                 # this piece starts again
                    if attempt == 5:
                        raise
                    if "403" in str(e) or "410" in str(e):  # the signed storage link expired: ask the Hub again
                        try:
                            h = self._open(url, token, method="HEAD", follow=False)
                            loc = h.headers.get("Location")
                            h.close()
                            if loc:
                                state["src"] = urllib.parse.urljoin(url, loc)
                        except (HubError, OSError):
                            pass
                    time.sleep(min(8, 1 + attempt * 2))

        todo = [i for i in range(n) if i not in done]
        # Connections are added while they help: 1, then 2, 4, 8 … as long as each step raises the speed by a
        # fifth; a network (or server) that does not gain from more goes back to the best number.
        queue = list(reversed(todo))
        cond = threading.Condition()
        ctl = {"limit": 1, "active": 0, "error": None}

        def worker():
            while True:
                with cond:
                    while ctl["active"] >= ctl["limit"] and queue and ctl["error"] is None:
                        cond.wait(0.5)
                    if not queue or ctl["error"] is not None:
                        return
                    i = queue.pop()
                    ctl["active"] += 1
                try:
                    fetch(i)
                except BaseException as e:             # the finished pieces stay; the next run resumes
                    with cond:
                        ctl["error"] = e
                        queue.append(i)
                finally:
                    with cond:
                        ctl["active"] -= 1
                        cond.notify_all()

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(CONNECTIONS, max(1, len(todo))))]
        for t in threads:
            t.start()
        tty = progress and size > (32 << 20) and getattr(sys.stdout, "isatty", lambda: False)()
        shown, step_t, step_got, best, last_pct = 0.0, time.time(), state["got"], (0.0, 1), -1
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
            now = time.time()
            if now - step_t >= 8 and ctl["limit"] < CONNECTIONS:      # try more connections
                rate = (state["got"] - step_got) / (now - step_t)
                with cond:
                    if rate > best[0] * 1.2:
                        best = (rate, ctl["limit"])
                        ctl["limit"] = min(CONNECTIONS, ctl["limit"] * 2)
                    else:
                        ctl["limit"] = best[1]
                        CONN_BEST[0] = best[1]
                        step_t = float("inf")              # settled
                    cond.notify_all()
                if step_t != float("inf"):
                    step_t, step_got = now, state["got"]
            if progress and size > (32 << 20):
                rate = (state["got"] - start_got) / max(1e-6, now - t0)
                line = (f"  {name}: {state['got'] / 2**20:,.0f} / {size / 2**20:,.0f} MB  {rate / 2**20:,.1f} MB/s"
                        f"  ({ctl['limit']} connection{'s' if ctl['limit'] > 1 else ''})")
                if tty and now - shown >= 1:
                    shown = now
                    print("\r" + line + "   ", end="", flush=True)
                elif not tty and int(100 * state["got"] / size) // 10 != last_pct:
                    last_pct = int(100 * state["got"] / size) // 10   # into a file: one line per 10 %
                    print(line, flush=True)
        if ctl["error"] is not None:
            if tty:
                print()
            raise ctl["error"]
        if progress and size > (32 << 20) and getattr(sys.stdout, "isatty", lambda: False)():
            print()
        if re.fullmatch(r"[0-9a-f]{64}", etag):
            h = hashlib.sha256()
            with open(part, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
            if h.hexdigest() != etag:
                os.remove(part)
                try:
                    os.remove(ranges)
                except OSError:
                    pass
                raise HubError(f"{name}: the downloaded file does not match the Hub's SHA-256; it was deleted")
        try:
            os.remove(ranges)
        except OSError:
            pass

    @staticmethod
    def _remember_ref(refs, rev, commit):
        try:
            d = {}
            if os.path.exists(refs):
                with open(refs, encoding="utf-8") as fh:
                    d = json.load(fh)
            d[rev] = commit
            checked = d.get("_checked") if isinstance(d.get("_checked"), dict) else {}
            checked[rev] = time.time()                 # when the Hub was last asked (see Hub.model)
            d["_checked"] = checked
            tmp = f"{refs}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(d, fh)
            os.replace(tmp, refs)
        except (OSError, ValueError):
            pass

    def snapshot(self, repo, patterns=("*.json", "*.safetensors", "*.txt", "*.model"), revision="main", token=None,
                 kind="model"):
        """Download the files that match the patterns; returns the folder."""
        import fnmatch
        names = [f for f in self.files(repo, revision, token, kind)
                 if any(fnmatch.fnmatch(f, p) for p in ([patterns] if isinstance(patterns, str) else patterns))]
        if not names:
            raise HubError(f"no files in {repo} match {', '.join(patterns)}")
        last = None
        for n in names:
            last = self.download(repo, n, revision, token, kind)
        return os.path.dirname(last)[: -len(os.path.dirname(names[-1]))].rstrip(os.sep) \
            if os.path.dirname(names[-1]) else os.path.dirname(last)

    # ---- loading
    def config(self, repo, revision="main", token=None):
        with open(self.download(repo, "config.json", revision, token), encoding="utf-8") as fh:
            return json.load(fh)

    def load(self, repo, filename=None, revision="main", token=None, dtype=None):
        """Weights as {name: tensor}: model.safetensors (also sharded), or a named .safetensors / .gguf file."""
        from . import models, modelio
        names = None if filename else self.files(repo, revision, token)
        if filename is None:
            if "model.safetensors" in names:
                filename = "model.safetensors"
            elif "model.safetensors.index.json" in names:
                idx = self.download(repo, "model.safetensors.index.json", revision, token)
                with open(idx, encoding="utf-8") as fh:
                    shards = sorted(set(json.load(fh)["weight_map"].values()))
                from .hub_models import read_safetensors
                out = {}
                for s in shards:
                    out.update(read_safetensors(self.download(repo, s, revision, token)))
                return out
            else:
                st = [n for n in names if n.endswith(".safetensors")]
                gg = [n for n in names if n.endswith(".gguf")]
                if st:
                    filename = st[0]
                elif gg:
                    raise HubError(f"{repo} has several GGUF files; choose one: hub.load(repo, \"{gg[0]}\")")
                else:
                    raise HubError(f"{repo} has no safetensors or GGUF weights (only PyTorch .bin files, which "
                                   "need pickle: use hub.download and torch_load with --allow-python)")
        path = self.download(repo, filename, revision, token)
        if filename.endswith(".gguf"):
            return modelio.load_gguf(path, nested=False)
        if filename.endswith(".safetensors"):
            from .hub_models import read_safetensors
            return read_safetensors(path)
        raise HubError(f"hub.load reads .safetensors and .gguf files; use hub.download for {filename}")

    def tokenizer(self, repo, revision="main", token=None):
        return Tokenizer.from_file(self.download(repo, "tokenizer.json", revision, token))

    def model(self, repo, revision="main", token=None, device="auto", update=False, gpu_layers=None, memory="auto"):
        """Download what a model needs and load it to run natively (see Model).
        device: "auto" (an NVIDIA GPU when there is one, else the processor), "cpu", "cuda" or "cuda:N".

        A model downloaded completely before starts from the cache with no network request; the Hub is asked for a
        newer version at most once a day (APSI_HUB_CHECK_HOURS), or now with update=true. Without a connection, the
        cached copy is used."""
        _repo_ok(repo)
        rev = revision or "main"
        folder = None if update else self._ready(repo, rev, None if _STATE["offline"] else CHECK_HOURS * 3600)
        if folder is None:
            try:
                folder = self._fetch_model(repo, rev, token)
            except (HubError, OSError):
                folder = self._ready(repo, rev, None)
                if folder is None:
                    raise
        return Model(folder, repo, device, gpu_layers=gpu_layers, memory=memory)

    def _ready(self, repo, rev, max_age):
        """The folder of a complete cached copy of the model, checked against the Hub within max_age seconds."""
        try:
            with open(os.path.join(cache_dir(), "models", repo.replace("/", "--"), "refs.json"), encoding="utf-8") as fh:
                refs = json.load(fh)
            commit = refs.get(rev) if re.fullmatch(r"[0-9a-f]{40}", rev) is None else rev
            if not isinstance(commit, str) or not commit:
                return None
            if max_age is not None and time.time() - float((refs.get("_checked") or {}).get(rev, 0)) > max_age:
                return None
            folder = os.path.dirname(self._local(repo, "model", commit, "config.json"))
            with open(os.path.join(folder, ".apsi-complete"), encoding="utf-8") as fh:
                names = json.load(fh)
            if all(os.path.isfile(os.path.join(folder, *n.split("/"))) for n in names):
                return folder
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        return None

    def _fetch_model(self, repo, revision, token):
        names = self.files(repo, revision, token)
        want = {"config.json", "tokenizer.json", "modules.json", "1_Pooling/config.json", "generation_config.json",
                "model.safetensors", "model.safetensors.index.json", "tokenizer_config.json"}
        get = [n for n in names if n in want]
        if "model.safetensors" not in names and "model.safetensors.index.json" in names:
            idx = self.download(repo, "model.safetensors.index.json", revision, token)
            with open(idx, encoding="utf-8") as fh:
                get += sorted(set(json.load(fh)["weight_map"].values()))
        if "model.safetensors" not in get and not any(n.endswith(".safetensors") for n in get):
            raise HubError(f"{repo} has no safetensors weights to run natively")
        for n in get:
            self.download(repo, n, revision, token)
        folder = os.path.dirname(self.download(repo, "config.json", revision, token))
        try:                                           # mark the copy complete: the next start needs no network
            with open(os.path.join(folder, ".apsi-complete"), "w", encoding="utf-8") as fh:
                json.dump(sorted(set(get) | {"config.json"}), fh)
        except OSError:
            pass
        return folder

    def pipeline(self, task, repo, revision="main", token=None):
        """A function for one task (sentiment-analysis, feature-extraction, ner, fill-mask, text-generation)."""
        return Pipeline(task, self.model(repo, revision, token))


# ======================================================================================== running models
class Model:
    """A Hub model running natively: embed / classify / fill_mask / tokens / generate, depending on the model."""

    @property
    def plan(self):
        """Where the model runs: {"gpu": {layers, of, bytes, tensor_cores, ...}, "cpu": {layers, weights, bytes, in,
        ram_available, note}} — what was placed on the GPU, what runs on the processor and from where."""
        return dict(getattr(self.net, "plan", {}) or {})

    def __init__(self, folder, name=None, device="auto", gpu_layers=None, memory="auto"):
        """memory: "ram" reads the weights into memory, "disk" maps them from the file (SSD) and decodes them as they
        are used, "auto" maps them only when the model is larger than half the free memory."""
        from . import hub_models, hub_gpu
        self.folder = folder
        self.name = name or folder
        pending = None
        try:
            with open(os.path.join(folder, "config.json"), encoding="utf-8") as fh:
                kind = json.load(fh).get("model_type", "")
        except (OSError, ValueError):
            kind = ""
        if kind in hub_models.DECODERS and str(device or "auto").lower() != "cpu":
            pending = hub_gpu.prepare(device)              # the GPU starts up while the weights are read
        self.config, weights = hub_models.load_folder(folder, mmap=_use_mmap(folder, memory))
        self.net = hub_models.build(self.config, weights)
        tj = os.path.join(folder, "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tj) if os.path.exists(tj) else None
        self.kind = "encoder" if isinstance(self.net, hub_models.Encoder) else "decoder"
        self.device, self.device_note = "cpu", ""
        self.last_tokens = 0
        if self.kind == "decoder":
            self.device, self.device_note = self.net.to_device(device, pending, gpu_layers)
            if self.net.gpu is not None and getattr(self.net, "_gl", None) is not None and self.net._ng < self.net.L:
                self.net._cm_for = self.net.gpu            # split: the processor's layers are made ready now, with
                self.net._prepare_cpu()                    # progress lines, not silently at the first token
        if self.device == "cpu" and isinstance(weights, hub_models.Weights) and not weights.mapped:
            if not (self.kind == "decoder" and self.net._prepare_cpu()):
                weights.decode_all(drop_raw=True)      # the processor works in float32: decode once, keep one copy
        self.labels = {int(k): v for k, v in (self.config.get("id2label") or {}).items()}
        self.pooling = "mean"
        self.normalize = False
        mj = os.path.join(folder, "modules.json")
        if os.path.exists(mj):                                 # a sentence-transformers model
            with open(mj, encoding="utf-8") as fh:
                mods = json.load(fh)
            self.normalize = any("Normalize" in m.get("type", "") for m in mods)
            pj = os.path.join(folder, "1_Pooling", "config.json")
            if os.path.exists(pj):
                with open(pj, encoding="utf-8") as fh:
                    pc = json.load(fh)
                self.pooling = "cls" if pc.get("pooling_mode_cls_token") else (
                    "max" if pc.get("pooling_mode_max_tokens") else "mean")
        gc = os.path.join(folder, "generation_config.json")
        self.gen = {}
        if os.path.exists(gc):
            with open(gc, encoding="utf-8") as fh:
                self.gen = json.load(fh)

    def _need_tok(self):
        if self.tokenizer is None:
            raise HubError(f"{self.name} has no tokenizer.json; pass token ids instead of text")
        return self.tokenizer

    def _batch(self, texts, max_length=None):
        import numpy as np
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        tok = self._need_tok()
        encs = [tok.encode(t) for t in texts]
        limit = max_length or self.config.get("max_position_embeddings", 512)
        if self.config.get("model_type") in ("roberta", "xlm-roberta", "camembert"):
            limit -= 2                                         # positions start after the padding index
        ids = [e.ids[:limit] for e in encs]
        T = max(len(i) for i in ids)
        pad = self.config.get("pad_token_id") or 0
        arr = np.full((len(ids), T), pad, np.int64)
        mask = np.zeros((len(ids), T), np.float32)
        for k, i in enumerate(ids):
            arr[k, :len(i)] = i
            mask[k, :len(i)] = 1
        return single, arr, mask, encs

    def hidden(self, texts):
        """Last hidden states [B, T, H] (encoders)."""
        single, ids, mask, _ = self._batch(texts)
        h = self.net.hidden(ids, mask)
        return h[0] if single else h

    def embed(self, texts, normalize=None):
        """Sentence embeddings: the model's own pooling (sentence-transformers) or mean pooling."""
        import numpy as np
        if self.kind != "encoder":
            raise HubError("embed() needs an encoder model (BERT, RoBERTa, MiniLM, E5, BGE …)")
        single, ids, mask, _ = self._batch(texts)
        h = self.net.hidden(ids, mask)
        if self.pooling == "cls":
            v = h[:, 0]
        elif self.pooling == "max":
            v = np.where(mask[..., None] > 0, h, -1e9).max(1)
        else:
            v = (h * mask[..., None]).sum(1) / np.maximum(mask.sum(1, keepdims=True), 1e-9)
        if self.normalize if normalize is None else normalize:
            v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
        return v[0] if single else v

    def similarity(self, a, b):
        import numpy as np
        va, vb = self.embed([a, b], normalize=True)
        return float(np.dot(va, vb))

    def classify(self, texts, top_k=1):
        import numpy as np
        single, ids, mask, _ = self._batch(texts)
        logits = self.net.classify(self.net.hidden(ids, mask))
        multi = self.config.get("problem_type") == "multi_label_classification"
        probs = 1 / (1 + np.exp(-logits)) if multi else np.exp(logits - logits.max(-1, keepdims=True))
        if not multi:
            probs = probs / probs.sum(-1, keepdims=True)
        out = []
        for p in probs:
            order = np.argsort(-p)[:max(1, int(top_k))]
            rows = [{"label": self.labels.get(int(i), f"LABEL_{int(i)}"), "score": float(p[i])} for i in order]
            out.append(rows[0] if top_k == 1 else rows)
        return out[0] if single else out

    def tokens(self, text):
        """Token classification (named entities …): one label per token."""
        import numpy as np
        _, ids, mask, encs = self._batch(text)
        logits = self.net.token_logits(self.net.hidden(ids, mask))[0]
        p = np.exp(logits - logits.max(-1, keepdims=True))
        p = p / p.sum(-1, keepdims=True)
        out = []
        for i, tok in enumerate(encs[0].tokens[:len(p)]):
            if tok in self.tokenizer.special:
                continue
            j = int(np.argmax(p[i]))
            out.append({"token": tok, "label": self.labels.get(j, f"LABEL_{j}"), "score": float(p[i, j])})
        return out

    def fill_mask(self, text, top_k=5):
        import numpy as np
        tok = self._need_tok()
        _, ids, mask, encs = self._batch(text)
        logits = self.net.mlm_logits(self.net.hidden(ids, mask))[0]
        mask_tokens = {t for t in tok.added if "mask" in t.lower()}
        pos = [i for i, t in enumerate(encs[0].tokens) if t in mask_tokens]
        if not pos:
            raise HubError(f"the text has no mask token ({', '.join(sorted(mask_tokens)) or '[MASK]'})")
        lg = logits[pos[0]]
        p = np.exp(lg - lg.max())
        p = p / p.sum()
        top = np.argsort(-p)[:int(top_k)]
        return [{"token": tok.decode([int(i)]).strip(), "id": int(i), "score": float(p[i])} for i in top]

    def generate(self, prompt, max_new_tokens=64, temperature=0.0, top_k=0, top_p=1.0, seed=None,
                 repetition_penalty=1.0, stream=None, add_special_tokens=True):
        """Continue a prompt (decoders). stream=fn(text) receives each piece as it is produced."""
        if self.kind != "decoder":
            raise HubError("generate() needs a decoder model (GPT-2, Llama, Mistral, Qwen, Gemma)")
        tok = self._need_tok()
        ids = tok.encode(prompt, add_special_tokens=add_special_tokens).ids if isinstance(prompt, str) else list(prompt)
        return self._run(ids, max_new_tokens, temperature, top_k, top_p, seed, repetition_penalty, stream)

    def _run(self, ids, max_new_tokens, temperature, top_k, top_p, seed, repetition_penalty, stream, extra_stops=()):
        tok = self._need_tok()
        eos = self.gen.get("eos_token_id", self.config.get("eos_token_id"))
        eos = eos if isinstance(eos, list) else ([eos] if eos is not None else [])
        eos = list(eos) + [t for t in extra_stops if t not in eos]
        done = []
        at = [0, 0]                                    # decode only the last few tokens: [context start, shown up to]

        def emit(t):
            if stream is not None:
                done.append(t)
                before = tok.decode(done[at[0]:at[1]], skip_special_tokens=True)
                text = tok.decode(done[at[0]:], skip_special_tokens=True)
                if len(text) > len(before) and not text.endswith("\ufffd"):   # not half of a character
                    at[0], at[1] = at[1], len(done)
                    stream(text[len(before):])
        self.last_prompt_tokens = len(ids)             # the prompt's length, for usage reports
        out = self.net.generate(ids, max_new_tokens, temperature, top_k, top_p, eos, seed, repetition_penalty,
                                emit if stream is not None else None)
        self.last_tokens = len(out)                    # how many tokens the last generate() produced
        return tok.decode(out, skip_special_tokens=True)

    # ---------------------------------------------------------------- chat
    def chat_format(self):
        """The prompt format of a chat model, from its own chat template: chatml (Qwen, SmolLM, …), llama3, gemma,
        mistral ([INST]) or none (not a chat model)."""
        if getattr(self, "_fmt", None) is None:
            t = ""
            try:
                with open(os.path.join(self.folder, "tokenizer_config.json"), encoding="utf-8") as fh:
                    t = json.load(fh).get("chat_template") or ""
                if isinstance(t, list):                    # several named templates
                    t = " ".join(str(x.get("template", "")) for x in t if isinstance(x, dict))
            except (OSError, ValueError, AttributeError):
                t = ""
            added = getattr(self.tokenizer, "added", {}) or {}
            if "<|start_header_id|>" in t or ("<|start_header_id|>" in added and not t):
                f = "llama3"
            elif "<start_of_turn>" in t or ("<start_of_turn>" in added and not t):
                f = "gemma"
            elif "<|im_start|>" in t or ("<|im_start|>" in added and not t):
                f = "chatml"
            elif "[INST]" in t:
                f = "mistral"
            else:
                f = "none"
            self._fmt = f
        return self._fmt

    def chat_prompt(self, messages, system=None):
        """The text the model sees for a conversation: messages is a text (one user message) or a list of
        {"role": "user" | "assistant" | "system", "content": text}."""
        msgs = _messages(messages, system)
        f = self.chat_format()
        if f == "llama3":
            s = "<|begin_of_text|>"
            for m in msgs:
                s += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
            return s + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if f == "gemma":                               # no system role: it goes in front of the first user turn
            s, pre = "<bos>", ""
            for m in msgs:
                if m["role"] == "system":
                    pre += m["content"] + "\n\n"
                    continue
                role = "model" if m["role"] == "assistant" else "user"
                s += f"<start_of_turn>{role}\n{(pre if role == 'user' else '') + m['content']}<end_of_turn>\n"
                if role == "user":
                    pre = ""
            return s + "<start_of_turn>model\n"
        if f == "mistral":
            s, pre = "<s>", ""
            for m in msgs:
                if m["role"] == "system":
                    pre += m["content"] + "\n\n"
                elif m["role"] == "user":
                    s += f"[INST] {pre}{m['content']} [/INST]"
                    pre = ""
                else:
                    s += f" {m['content']}</s>"
            return s
        if f == "chatml":
            s = ""
            for m in msgs:
                s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            return s + "<|im_start|>assistant\n"
        s = ""                                         # a plain language model: a readable transcript
        for m in msgs:
            s += {"system": "", "user": "User: ", "assistant": "Assistant: "}[m["role"]] + m["content"] + "\n"
        return s + "Assistant:"

    def _stops(self):
        """End-of-turn tokens of the chat formats (besides the model's own end-of-sequence)."""
        added = getattr(self.tokenizer, "added", {}) or {}
        return [added[t] for t in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|endoftext|>", "</s>",
                                   "<|end_of_text|>") if t in added]

    def chat(self, messages, system=None, max_new_tokens=512, temperature=0.7, top_p=0.9, top_k=0,
             repetition_penalty=1.1, seed=None, stream=None):
        """The model's reply in a conversation (see chat_prompt): the right prompt format, stops at the end of the
        turn, and the oldest turns are left out when the conversation no longer fits the model's window."""
        if self.kind != "decoder":
            raise HubError("chat() needs a decoder model (Llama, Mistral, Qwen, Gemma, …)")
        tok = self._need_tok()
        msgs = _messages(messages, system)
        window = int(self.config.get("max_position_embeddings") or self.config.get("n_positions") or 4096)
        budget = max(64, min(window, 32768) - int(max_new_tokens) - 8)
        sys_msgs = [m for m in msgs if m["role"] == "system"]
        rest = [m for m in msgs if m["role"] != "system"]
        ids = tok.encode(self.chat_prompt(sys_msgs + rest), add_special_tokens=False).ids
        while len(ids) > budget and len(rest) > 1:     # drop the oldest exchange, keep the newest message
            rest = rest[2:] if len(rest) > 2 else rest[1:]
            ids = tok.encode(self.chat_prompt(sys_msgs + rest), add_special_tokens=False).ids
        if len(ids) > budget:
            ids = ids[-budget:]
        return self._run(ids, max_new_tokens, temperature, top_k, top_p, seed, repetition_penalty, stream,
                         extra_stops=self._stops()).strip()

    def conversation(self, system=None, **options):
        """A conversation that remembers its turns: c.say(text, stream=fn) -> reply; c.reset(); c.history."""
        return Conversation(self, system, options)

    def to(self, device):
        """Move a decoder to "cpu", "cuda", "cuda:N" or "auto"; returns the device it runs on."""
        if self.kind == "decoder":
            self.device, self.device_note = self.net.to_device(device)
        return self.device

    def __repr__(self):
        return f"<model {self.name}: {self.config.get('model_type')} {self.kind} on {self.device}>"


def _available_ram():
    """Bytes of memory free for new allocations, or None when unknown."""
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return int(m.ullAvailPhys)
            return None
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _use_mmap(folder, memory):
    memory = str(memory or "auto").lower()
    if memory in ("disk", "ssd", "mmap"):
        return True
    if memory in ("ram", "memory"):
        return False
    if memory != "auto":
        raise HubError(f"memory must be \"auto\", \"ram\" or \"disk\", not \"{memory}\"")
    size = sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(".safetensors"))
    free = _available_ram()
    return free is not None and size > free / 2


def _messages(messages, system=None):
    """Normalise a conversation to [{"role", "content"}], with an optional system message first."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    out = []
    for m in messages or []:
        if isinstance(m, str):
            m = {"role": "user", "content": m}
        try:
            role = str(m.get("role", "user")).lower()
            content = m.get("content", m.get("text", ""))
        except AttributeError:
            raise HubError("a chat message is a text or {\"role\": ..., \"content\": ...}") from None
        if role in ("bot", "model", "ai"):
            role = "assistant"
        if role not in ("system", "user", "assistant"):
            raise HubError(f"unknown chat role '{role}' (use system, user or assistant)")
        out.append({"role": role, "content": str(content)})
    if system is not None:
        out = [{"role": "system", "content": str(system)}] + [m for m in out if m["role"] != "system"]
    if not out or out[-1]["role"] == "assistant":
        if not out:
            raise HubError("chat() needs at least one message")
    return out


class Conversation:
    """Turns remembered between calls: say() adds the user's text and the model's reply."""

    def __init__(self, model, system=None, options=None):
        self.model = model
        self.system = system
        self.options = dict(options or {})
        self.history = []

    def say(self, text, stream=None, **options):
        o = dict(self.options)
        o.update(options)
        self.history.append({"role": "user", "content": str(text)})
        try:
            reply = self.model.chat(self.history, system=self.system, stream=stream, **o)
        except BaseException:
            self.history.pop()                         # a failed or interrupted turn leaves no half entry
            raise
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self, system=None):
        self.history = []
        if system is not None:
            self.system = system

    def transcript(self):
        return "\n\n".join(("You" if m["role"] == "user" else "Assistant") + ": " + m["content"] for m in self.history)

    def __repr__(self):
        return f"<conversation with {self.model.name}: {len(self.history)} messages>"


class Pipeline:
    TASKS = {"feature-extraction": "embed", "sentence-similarity": "embed", "embeddings": "embed",
             "text-classification": "classify", "sentiment-analysis": "classify", "token-classification": "tokens",
             "ner": "tokens", "fill-mask": "fill_mask", "text-generation": "generate"}

    def __init__(self, task, model):
        if task not in self.TASKS:
            raise HubError(f"unknown task '{task}' (tasks: {', '.join(self.TASKS)})")
        self.task, self.model = task, model
        self.fn = getattr(model, self.TASKS[task])

    def __call__(self, *a, **k):
        return self.fn(*a, **k)

    def __repr__(self):
        return f"<pipeline {self.task}: {self.model.name}>"


# ======================================================================================== tokenizers
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


_B2U = _bytes_to_unicode()
_U2B = {v: k for k, v in _B2U.items()}
# GPT-2's split: contractions, letters, numbers, other symbols, whitespace (\p{L} / \p{N} written for Python's re)
_L = r"[^\W\d_]"
_GPT2 = re.compile(rf"""'s|'t|'re|'ve|'m|'ll|'d| ?{_L}+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""", re.UNICODE)


class Encoding:
    def __init__(self, ids, tokens):
        self.ids, self.tokens = ids, tokens
        self.attention_mask = [1] * len(ids)

    def __repr__(self):
        return f"Encoding(ids={self.ids}, tokens={self.tokens})"


class Tokenizer:
    """tokenizer.json (Hugging Face `tokenizers` format): WordPiece (BERT family), byte-level BPE (GPT-2, RoBERTa,
    Llama 3, Qwen …) and SentencePiece-style BPE with byte fallback (Llama 2, Mistral …)."""

    def __init__(self, spec):
        self.spec = spec
        m = spec["model"]
        self.type = m.get("type") or ("WordPiece" if "continuing_subword_prefix" in m and "merges" not in m else "BPE")
        if self.type == "Unigram":
            self.scores = {piece: (i, float(score)) for i, (piece, score) in enumerate(m["vocab"])}
            self.vocab = {piece: i for piece, (i, _) in self.scores.items()}
            self.unk_id = m.get("unk_id")
            self.byte_fallback = bool(m.get("byte_fallback"))
            self.max_len = max((len(p) for p in self.vocab), default=1)
            self.min_score = min((sc for _, sc in self.scores.values()), default=0.0)
            self.unk = m["vocab"][self.unk_id][0] if self.unk_id is not None else None
        else:
            self.vocab = m["vocab"]
        self.inv = {i: t for t, i in self.vocab.items()}
        if self.type != "Unigram":
            self.unk = m.get("unk_token")
        self.added = {a["content"]: a["id"] for a in spec.get("added_tokens", [])}
        self.inv.update({i: t for t, i in self.added.items()})
        self.special = {a["content"] for a in spec.get("added_tokens", []) if a.get("special")}
        norm = spec.get("normalizer") or {}
        pre = spec.get("pre_tokenizer") or {}
        self.norms = norm.get("normalizers", [norm]) if norm else []
        self.pres = pre.get("pretokenizers", [pre]) if pre else []
        self.byte_level = any(p.get("type") == "ByteLevel" for p in self.pres)
        self.metaspace = next((p for p in self.pres if p.get("type") == "Metaspace"), None)
        self.post = spec.get("post_processor")
        if self.type == "BPE":
            merges = m.get("merges", [])
            pairs = [tuple(x.split(" ", 1)) if isinstance(x, str) else tuple(x) for x in merges]
            self.ranks = {p: i for i, p in enumerate(pairs)}
            self.byte_fallback = bool(m.get("byte_fallback"))
            self.cache = {}
        elif self.type == "WordPiece":
            self.prefix = m.get("continuing_subword_prefix", "##")
            self.max_chars = m.get("max_input_chars_per_word", 100)
        elif self.type == "Unigram":
            pass
        else:
            raise HubError(f"tokenizer model '{self.type}' is not supported natively (WordPiece and BPE are); "
                           "use hf_pipeline with --allow-python")
        if self.added:
            alts = sorted(self.added, key=len, reverse=True)
            self.added_re = re.compile("(" + "|".join(re.escape(a) for a in alts) + ")")
        else:
            self.added_re = None

    @classmethod
    def from_file(cls, path):
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

    # ---- normalisation and pre-tokenisation
    def _normalize(self, s):
        for n in self.norms:
            t = n.get("type")
            if t == "BertNormalizer":
                if n.get("clean_text", True):
                    s = "".join(" " if c in "\t\n\r" or unicodedata.category(c) == "Zs" else c for c in s
                                if not (ord(c) == 0 or ord(c) == 0xFFFD or unicodedata.category(c) in ("Cc", "Cf")
                                        and c not in "\t\n\r"))
                if n.get("handle_chinese_chars", True):
                    s = "".join(f" {c} " if _cjk(c) else c for c in s)
                lower = n.get("lowercase", True)
                strip = n.get("strip_accents")
                if strip is None:
                    strip = lower
                if lower:
                    s = s.lower()
                if strip:
                    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
            elif t == "Lowercase":
                s = s.lower()
            elif t in ("NFC", "NFD", "NFKC", "NFKD"):
                s = unicodedata.normalize(t, s)
            elif t == "StripAccents":
                s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
            elif t == "Prepend":
                s = n["prepend"] + s
            elif t == "Replace":
                pat = n["pattern"]
                s = re.sub(pat["Regex"], n["content"], s) if "Regex" in pat else s.replace(pat["String"], n["content"])
            elif t == "Strip":
                s = s.strip()
            elif t == "Precompiled":
                s = unicodedata.normalize("NFKC", s)            # SentencePiece's nmt_nfkc, without its char map
        return s

    def _pretokenize(self, s):
        words = [s]
        for p in self.pres:
            t = p.get("type")
            if t == "BertPreTokenizer" or t == "WhitespaceSplit" or t == "Whitespace":
                out = []
                for w in words:
                    if t == "BertPreTokenizer":
                        out += re.findall(r"\w+|[^\w\s]", w, re.UNICODE)
                    elif t == "Whitespace":
                        out += re.findall(r"\w+|[^\w\s]+", w, re.UNICODE)
                    else:
                        out += w.split()
                words = out
            elif t == "ByteLevel":
                out = []
                for w in words:
                    if p.get("add_prefix_space") and not w.startswith(" "):
                        w = " " + w
                    pieces = _GPT2.findall(w) if p.get("use_regex", True) else [w]
                    out += ["".join(_B2U[b] for b in piece.encode("utf-8")) for piece in pieces]
                words = out
            elif t == "Metaspace":
                rep = p.get("replacement", "▁")
                out = []
                for w in words:
                    w = w.replace(" ", rep)
                    if p.get("prepend_scheme", "always" if p.get("add_prefix_space", True) else "never") != "never" \
                            and not w.startswith(rep):
                        w = rep + w
                    if p.get("split", True):
                        out += [x for x in re.split(f"(?={re.escape(rep)})", w) if x]
                    else:
                        out.append(w)
                words = out
            elif t == "Digits":
                out = []
                for w in words:
                    out += re.findall(r"\d|\D+", w) if p.get("individual_digits") else re.findall(r"\d+|\D+", w)
                words = out
            elif t == "Split":
                pat = p["pattern"].get("Regex") or re.escape(p["pattern"].get("String", ""))
                out = []
                for w in words:
                    out += [x for x in re.split(f"({_py_regex(pat)})", w) if x]
                words = out
            elif t in ("Punctuation",):
                out = []
                for w in words:
                    out += re.findall(r"[^\W_]+|[^\w\s]|_|\s+", w)
                words = out
        return words

    # ---- models
    def _bpe(self, word):
        hit = self.cache.get(word)
        if hit is not None:
            return hit
        parts = self._merge(word) if len(word) > 12 else self._merge_short(word)
        out = []
        for p in parts:
            if p in self.vocab:
                out.append(p)
            elif self.byte_fallback:
                out += [f"<0x{b:02X}>" for b in p.encode("utf-8")]
            elif self.unk:
                out.append(self.unk)
        if len(self.cache) > 200_000:                  # bounded: long documents do not grow it forever
            self.cache.clear()
        self.cache[word] = out
        return out

    def _merge(self, word):
        """BPE merges with a heap: the lowest-ranked pair first, equal ranks left to right (the order of the
        reference implementation), in O(n log n) instead of O(n^2) for long words (text without spaces: Thai,
        Chinese, Japanese, code)."""
        import heapq
        ranks = self.ranks
        sym = list(word)
        n = len(sym)
        nxt = list(range(1, n + 1))
        prv = list(range(-1, n - 1))
        alive = [True] * n
        heap = []
        for i in range(n - 1):
            r = ranks.get((sym[i], sym[i + 1]))
            if r is not None:
                heap.append((r, i, sym[i], sym[i + 1]))
        heapq.heapify(heap)
        while heap:
            r, i, a, b = heapq.heappop(heap)
            j = nxt[i] if i < n else n
            if not alive[i] or j >= n or sym[i] != a or sym[j] != b:
                continue                                # stale: one side was merged already
            sym[i] = a + b
            alive[j] = False
            nxt[i] = nxt[j]
            if nxt[i] < n:
                prv[nxt[i]] = i
            p = prv[i]
            if p >= 0:
                rr = ranks.get((sym[p], sym[i]))
                if rr is not None:
                    heapq.heappush(heap, (rr, p, sym[p], sym[i]))
            q = nxt[i]
            if q < n:
                rr = ranks.get((sym[i], sym[q]))
                if rr is not None:
                    heapq.heappush(heap, (rr, i, sym[i], sym[q]))
        return [sym[i] for i in range(n) if alive[i]]

    def _merge_short(self, word):
        parts = list(word)
        while len(parts) > 1:
            best, bi = None, -1
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            a, b = parts[bi], parts[bi + 1]
            merged, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        return parts

    def _unigram(self, word):
        cache = self.__dict__.setdefault("cache", {})
        hit = cache.get(word)
        if hit is None:
            hit = self._viterbi(word)
            if len(cache) > 200_000:
                cache.clear()
            cache[word] = hit
        return hit

    def _viterbi(self, word):
        """SentencePiece's Viterbi: the segmentation with the highest total log-probability; characters no piece
        covers become <unk> (consecutive ones merged), or bytes with byte fallback."""
        n = len(word)
        best = [(-1e300, -1, None)] * (n + 1)
        best[0] = (0.0, -1, None)
        unk_score = self.min_score - 10.0
        for i in range(n):
            if best[i][0] <= -1e299:
                continue
            base = best[i][0]
            found = False
            for j in range(i + 1, min(n, i + self.max_len) + 1):
                hit = self.scores.get(word[i:j])
                if hit is not None:
                    found = found or j == i + 1
                    sc = base + hit[1]
                    if sc > best[j][0]:
                        best[j] = (sc, i, word[i:j])
            if not found:
                sc = base + unk_score
                if sc > best[i + 1][0]:
                    best[i + 1] = (sc, i, None)                    # None marks an unknown character
        pieces, j = [], n
        while j > 0:
            _, i, piece = best[j]
            pieces.append((piece, word[i:j]))
            j = i
        pieces.reverse()
        out = []
        for piece, text in pieces:
            if piece is not None:
                out.append(piece)
            elif self.byte_fallback:
                out += [f"<0x{b:02X}>" for b in text.encode("utf-8")]
            elif out and out[-1] == self.unk:
                continue
            else:
                out.append(self.unk)
        return out

    def _wordpiece(self, word):
        if len(word) > self.max_chars:
            return [self.unk]
        out, start = [], 0
        while start < len(word):
            end, cur = len(word), None
            while start < end:
                sub = word[start:end]
                if start > 0:
                    sub = self.prefix + sub
                if sub in self.vocab:
                    cur = sub
                    break
                end -= 1
            if cur is None:
                return [self.unk]
            out.append(cur)
            start = end
        return out

    def _tokens(self, text):
        chunks = self.added_re.split(text) if self.added_re else [text]
        toks = []
        for k, chunk in enumerate(chunks):
            if not chunk:
                continue
            if self.added_re and k % 2 == 1:
                toks.append(chunk)
                continue
            for w in self._pretokenize(self._normalize(chunk)):
                toks += self._bpe(w) if self.type == "BPE" else (
                    self._unigram(w) if self.type == "Unigram" else self._wordpiece(w))
        return toks

    def encode(self, text, pair=None, add_special_tokens=True):
        toks = self._tokens(text)
        toks2 = self._tokens(pair) if pair is not None else None
        if add_special_tokens and self.post:
            toks = self._post(toks, toks2)
        elif toks2:
            toks = toks + toks2
        unk = self.vocab.get(self.unk, 0) if self.unk is not None else 0
        ids = [self.added.get(t, self.vocab.get(t, unk)) for t in toks]
        return Encoding(ids, toks)

    def _post(self, a, b):
        p = self.post
        procs = p.get("processors", [p]) if p.get("type") == "Sequence" else [p]
        for q in procs:
            t = q.get("type")
            if t == "TemplateProcessing":
                tpl = q["pair"] if b is not None else q["single"]
                out = []
                for item in tpl:
                    if "SpecialToken" in item:
                        out.append(item["SpecialToken"]["id"])
                    else:
                        out += a if item["Sequence"]["id"] == "A" else (b or [])
                return out
            if t == "BertProcessing":
                cls, sep = q["cls"][0], q["sep"][0]
                return [cls] + a + [sep] + ((b + [sep]) if b is not None else [])
            if t == "RobertaProcessing":
                cls, sep = q["cls"][0], q["sep"][0]
                return [cls] + a + [sep] + (([sep] + b + [sep]) if b is not None else [])
        return a + (b or [])

    def decode(self, ids, skip_special_tokens=True):
        toks = [self.inv.get(int(i), "") for i in ids]
        if skip_special_tokens:
            toks = [t for t in toks if t not in self.special]
        if self.type == "WordPiece":
            text = " ".join(toks).replace(" " + self.prefix, "")
            return re.sub(r" ([.,!?;:')])", r"\1", text)
        if self.byte_level:
            return bytes(_U2B.get(c, 0) for c in "".join(toks)).decode("utf-8", "replace")
        out = bytearray()
        for t in toks:
            m = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", t)
            out += bytes([int(m.group(1), 16)]) if m else t.encode("utf-8")
        text = out.decode("utf-8", "replace")
        dec = self.spec.get("decoder") or {}
        decs = dec.get("decoders", [dec]) if dec else []
        rep = (self.metaspace or {}).get("replacement") or next(
            (d.get("replacement", "▁") for d in decs if d.get("type") == "Metaspace"), None) or next(
            (d["pattern"].get("String") for d in decs if d.get("type") == "Replace" and d.get("content") == " "), None)
        if rep is None and any(n.get("type") == "Prepend" for n in self.norms):
            rep = "▁"
        if rep:
            text = text.replace(rep, " ")
            if text.startswith(" "):
                text = text[1:]
        return text

    def __call__(self, text, **kw):
        return self.encode(text, **kw)

    @property
    def vocab_size(self):
        return max(max(self.vocab.values(), default=-1), max(self.added.values(), default=-1)) + 1

    def __repr__(self):
        return f"<tokenizer {self.type}, {self.vocab_size:,} tokens>"


def _cjk(c):
    o = ord(c)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x20000 <= o <= 0x2A6DF or 0x2A700 <= o <= 0x2B73F
            or 0x2B740 <= o <= 0x2B81F or 0x2B820 <= o <= 0x2CEAF or 0xF900 <= o <= 0xFAFF or 0x2F800 <= o <= 0x2FA1F)


_LETTERS = ("a-zA-Z\u00aa\u00b5\u00ba\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u02af\u0370-\u03ff\u0400-\u052f"
            "\u0531-\u0587\u05d0-\u05ea\u0620-\u064a\u0671-\u06d3\u0900-\u0939\u0e01-\u0e30\u0e40-\u0e46"
            "\u1100-\u11ff\u3041-\u3096\u30a1-\u30fa\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7a3")
_NUMBERS = "0-9\u0660-\u0669\u06f0-\u06f9\u0966-\u096f\u0e50-\u0e59\uff10-\uff19"


def _py_regex(pat):
    """Oniguruma \\p{L} / \\p{N} / \\p{M} classes (as in tokenizer.json) rewritten for Python's re: outside a
    character class they become Unicode-aware classes; inside one, explicit ranges of the common scripts."""
    out, i, depth = [], 0, 0
    while i < len(pat):
        c = pat[i]
        if c == "\\" and i + 1 < len(pat):
            two = pat[i:i + 5]
            if two in ("\\p{L}", "\\p{N}", "\\p{M}", "\\P{L}", "\\P{N}"):
                kind = two[3]
                neg = two[1] == "P"
                if depth:
                    out.append({"L": _LETTERS, "N": _NUMBERS, "M": "\u0300-\u036f"}[kind])
                else:
                    cls = {"L": "[^\\W\\d_]", "N": "\\d", "M": "[\u0300-\u036f]"}[kind]
                    out.append({"L": "[\\W\\d_]", "N": "\\D"}.get(kind, cls) if neg else cls)
                i += 5
                continue
            out.append(pat[i:i + 2])
            i += 2
            continue
        if c == "[":
            depth += 1
        elif c == "]" and depth:
            depth -= 1
        out.append(c)
        i += 1
    return "".join(out)


def module(interp=None):
    hub = Hub(interp)

    def login(token):
        tok = getattr(token, "reveal", None)
        _STATE["token"] = tok() if callable(tok) else str(token)
        return True

    def logout():
        _STATE["token"] = None

    def offline(on=True):
        _STATE["offline"] = bool(on)
        return _STATE["offline"]

    def clear_cache(repo=None):
        root = os.path.join(cache_dir(), "models", repo.replace("/", "--")) if repo else cache_dir()
        shutil.rmtree(root, ignore_errors=True)

    def tokenizer_from_file(path):
        if interp is not None:
            from .security import check_path
            path = check_path(interp, path, "read", "hub.tokenizer_from_file()")
        return Tokenizer.from_file(path)

    def model_from_folder(folder, device="auto", gpu_layers=None, memory="auto"):
        """Load a model that runs natively from a local folder in Hugging Face layout."""
        if interp is not None:
            from .security import check_path
            folder = check_path(interp, folder, "read", "hub.model_from_folder()")
        return Model(folder, device=device, gpu_layers=gpu_layers, memory=memory)

    def serve(model, host="127.0.0.1", port=8000, api_key=None, name=None, max_queue=64, max_tokens=4096,
              timeout=600.0, background=False, log=True):
        """Serve a decoder with the OpenAI API (/v1/chat/completions, /v1/completions, /v1/models, /health,
        /metrics). background=true returns the server (url, stop()) instead of serving until Ctrl+C."""
        from . import ai_server
        if interp is not None:
            from .security import check_net
            check_net(interp, str(host), int(port), "hub.serve()")
        return ai_server.serve(model, host, int(port), api_key, name, int(max_queue), int(max_tokens),
                               float(timeout.v if hasattr(timeout, "v") else timeout), bool(background), bool(log))

    def devices():
        """The NVIDIA GPUs the hub can use: [{"index", "name", "memory_gb", "compute_capability", ...}] (or [])."""
        from .native import cuda
        return cuda.devices()

    return {"devices": devices, "model": hub.model, "pipeline": hub.pipeline, "model_from_folder": model_from_folder, "serve": serve, "info": hub.info, "files": hub.files, "search": hub.search, "download": hub.download,
            "snapshot": hub.snapshot, "load": hub.load, "config": hub.config, "tokenizer": hub.tokenizer,
            "tokenizer_from_file": tokenizer_from_file, "login": login, "logout": logout, "offline": offline,
            "cache_dir": cache_dir, "clear_cache": clear_cache, "HubError": HubError}
