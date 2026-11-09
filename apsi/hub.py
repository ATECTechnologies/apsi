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
import json
import os
import re
import shutil
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

from .errors import RuntimeErr

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
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
        """One file → its local path (cached by commit; resumed; SHA-256 checked for large files)."""
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
            with open(part, "ab" if have else "wb") as out:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if progress and size > (32 << 20) and time.time() - shown > 1:
                        shown = time.time()
                        rate = (done - have) / max(1e-6, shown - t0)
                        print(f"\r  {name}: {done / 2**20:,.0f} / {size / 2**20:,.0f} MB  {rate / 2**20:,.1f} MB/s",
                              end="", flush=True)
            r.close()
            if progress and size > (32 << 20):
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

    @staticmethod
    def _remember_ref(refs, rev, commit):
        try:
            d = {}
            if os.path.exists(refs):
                with open(refs, encoding="utf-8") as fh:
                    d = json.load(fh)
            if d.get(rev) != commit:
                d[rev] = commit
                with open(refs, "w", encoding="utf-8") as fh:
                    json.dump(d, fh)
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
                out = {}
                for s in shards:
                    out.update(models.load_safetensors(self.download(repo, s, revision, token), dtype=dtype,
                                                       nested=False))
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
            return models.load_safetensors(path, dtype=dtype, nested=False)
        raise HubError(f"hub.load reads .safetensors and .gguf files; use hub.download for {filename}")

    def tokenizer(self, repo, revision="main", token=None):
        return Tokenizer.from_file(self.download(repo, "tokenizer.json", revision, token))


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
        self.vocab = m["vocab"]
        self.inv = {i: t for t, i in self.vocab.items()}
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
        out = []
        for p in parts:
            if p in self.vocab:
                out.append(p)
            elif self.byte_fallback:
                out += [f"<0x{b:02X}>" for b in p.encode("utf-8")]
            elif self.unk:
                out.append(self.unk)
        self.cache[word] = out
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
                toks += self._bpe(w) if self.type == "BPE" else self._wordpiece(w)
        return toks

    def encode(self, text, pair=None, add_special_tokens=True):
        toks = self._tokens(text)
        toks2 = self._tokens(pair) if pair is not None else None
        if add_special_tokens and self.post:
            toks = self._post(toks, toks2)
        elif toks2:
            toks = toks + toks2
        ids = [self.added.get(t, self.vocab.get(t, self.vocab.get(self.unk, 0))) for t in toks]
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

    return {"info": hub.info, "files": hub.files, "search": hub.search, "download": hub.download,
            "snapshot": hub.snapshot, "load": hub.load, "config": hub.config, "tokenizer": hub.tokenizer,
            "tokenizer_from_file": tokenizer_from_file, "login": login, "logout": logout, "offline": offline,
            "cache_dir": cache_dir, "clear_cache": clear_cache, "HubError": HubError}
