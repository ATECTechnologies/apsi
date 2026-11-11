"""Training text in any shape, tokenized with any tokenizer, packed into training windows.

    use data
    let tok = hub.tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    let ds = data.text_dataset("corpus/*.jsonl", tokenizer=tok, seq_len=1024) |> shuffle(256) |> batch(8)
    for xb, yb in ds: ...                         # xb, yb: [8, 1024] token ids; yb is xb moved by one

Sources: a path, a glob or a list of them (.txt .md .jsonl .json .csv .tsv .parquet), a list of strings or records,
or any pipeline. Each record becomes text by its shape, found by itself (format="auto"):

* plain text, or a record with `text` / `content` / `body`
* chat: `messages` / `conversations` ([{role, content}] or ShareGPT's [{from, value}])
* instruction (Alpaca): `instruction`, `input`, `output`
* `prompt` + `completion` / `response`, and `question` + `answer`
* anything else: a template ("Q: {question}\\nA: {answer}"), a function record -> text, or field="name"

With a chat-model tokenizer, chat records use the model's own format (chatml, llama3, gemma, mistral). For chat,
instruction and prompt/completion records the prompt part is labelled -100 (not trained on) unless
train_on_prompt=true; cross_entropy leaves those positions out.

Tokenizers: a hub tokenizer (or lm.tokenizer), anything with encode(text), a function text -> ids, "bytes"
(256 byte values, no download) or "chars" (one id per character, the vocabulary built from the data).
"""
import glob as _glob
import json
import os

import numpy as np

from .errors import RuntimeErr

TEXT_KEYS = ("text", "content", "body", "document", "story")
CHAT_KEYS = ("messages", "conversations", "conversation", "dialog")
ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "bot": "assistant",
            "model": "assistant", "system": "system", "tool": "tool"}
IGNORE = -100


# ------------------------------------------------------------------ reading any file
def _files(src):
    out = []
    for s in (src if isinstance(src, (list, tuple)) else [src]):
        s = os.path.expanduser(str(s))
        if os.path.isdir(s):
            for root, _, names in sorted(os.walk(s)):
                out += [os.path.join(root, n) for n in sorted(names) if _kind(n)]
        elif any(c in s for c in "*?["):
            out += sorted(_glob.glob(s, recursive=True))
        else:
            out.append(s)
    if not out:
        raise RuntimeErr(f"text_dataset: no files match {src!r}")
    return out


def _kind(name):
    n = name.lower()
    for ext, k in ((".jsonl", "jsonl"), (".ndjson", "jsonl"), (".json", "json"), (".csv", "csv"), (".tsv", "tsv"),
                   (".parquet", "parquet"), (".txt", "text"), (".md", "text"), (".text", "text")):
        if n.endswith(ext):
            return k
    return None


def read_records(path, text_split="file"):
    """Records of one file: dicts for jsonl/json/csv/parquet, strings for text files ("file": the whole file is one
    document, "line": one per line, "paragraph": split at blank lines)."""
    kind = _kind(path) or "text"
    if not os.path.exists(path):
        raise RuntimeErr(f"text_dataset: '{path}' does not exist")
    if kind == "jsonl":
        with open(path, encoding="utf-8-sig") as fh:
            for n, line in enumerate(fh, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except ValueError as e:
                        raise RuntimeErr(f"{os.path.basename(path)} line {n}: not valid JSON ({e})")
    elif kind == "json":
        with open(path, encoding="utf-8-sig") as fh:
            v = json.load(fh)
        if isinstance(v, dict):                    # {"data": [...]} / {"train": [...]}
            lists = [x for x in v.values() if isinstance(x, list)]
            v = lists[0] if len(lists) == 1 else [v]
        yield from v
    elif kind in ("csv", "tsv"):
        import csv
        with open(path, encoding="utf-8-sig", newline="") as fh:
            yield from csv.DictReader(fh, delimiter="\t" if kind == "tsv" else ",")
    elif kind == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise RuntimeErr("reading .parquet needs pyarrow — install it with: apsi pkg install data")
        for b in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from b.to_pylist()
    else:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        if text_split == "line":
            yield from (l for l in text.splitlines() if l.strip())
        elif text_split == "paragraph":
            yield from (p.strip() for p in text.split("\n\n") if p.strip())
        else:
            yield text


# ------------------------------------------------------------------ a record -> (prompt, answer) text parts
def _get(rec, k):
    if isinstance(rec, dict):
        return rec.get(k)
    return getattr(rec, k, None)


def _keys(rec):
    if isinstance(rec, dict):
        return list(rec)
    f = getattr(rec, "fields", None)
    return list(f() if callable(f) else f or [])


def detect(rec):
    """The format of one record: text, chat, instruction, completion, qa or other."""
    if isinstance(rec, str):
        return "text"
    keys = set(_keys(rec))
    if keys & set(CHAT_KEYS):
        return "chat"
    if "instruction" in keys and ("output" in keys or "response" in keys):
        return "instruction"
    if "prompt" in keys and keys & {"completion", "response", "chosen"}:
        return "completion"
    if "question" in keys and "answer" in keys:
        return "qa"
    if keys & set(TEXT_KEYS):
        return "text"
    return "other"


def _chat_turns(msgs):
    out = []
    for m in msgs or []:
        role = _get(m, "role") or _get(m, "from") or "user"
        text = _get(m, "content")
        if text is None:
            text = _get(m, "value")
        if isinstance(text, list):                # [{type: text, text: ...}] content parts
            text = "".join(str(_get(p, "text") or "") for p in text)
        out.append((ROLE_MAP.get(str(role).lower(), str(role).lower()), "" if text is None else str(text)))
    return out


def _render_chat(turns, style):
    """[(trained?, text)] for a whole conversation in the model's chat format."""
    parts = []
    for role, text in turns:
        train = role == "assistant"
        if style == "chatml":
            parts += [(False, f"<|im_start|>{role}\n"), (train, text + "<|im_end|>"), (False, "\n")]
        elif style == "llama3":
            parts += [(False, f"<|start_header_id|>{role}<|end_header_id|>\n\n"), (train, text + "<|eot_id|>")]
        elif style == "gemma":
            r = "model" if role == "assistant" else "user"
            parts += [(False, f"<start_of_turn>{r}\n"), (train, text + "<end_of_turn>"), (False, "\n")]
        elif style == "mistral":
            parts += [(False, f"[INST] {text} [/INST]")] if not train else [(True, f" {text}</s>")]
        else:
            name = {"user": "User", "assistant": "Assistant", "system": "System"}.get(role, role.title())
            parts += [(False, f"{name}: "), (train, text), (False, "\n")]
    if style == "llama3" and parts:
        parts.insert(0, (False, "<|begin_of_text|>"))
    return parts


def to_parts(rec, fmt="auto", field=None, template=None, chat_style="plain"):
    """One record as [(trained?, text)] in order."""
    if template is not None:
        if callable(template):
            out = template(rec)
            if isinstance(out, (list, tuple)) and len(out) == 2 and all(isinstance(x, str) for x in out):
                return [(False, out[0]), (True, out[1])]          # (prompt, answer)
            return [(True, str(out))]
        vals = rec if isinstance(rec, dict) else {k: _get(rec, k) for k in _keys(rec)}
        try:
            return [(True, str(template).format(**{k: "" if v is None else v for k, v in vals.items()}))]
        except KeyError as e:
            raise RuntimeErr(f"text_dataset: the template uses {e} but a record has only {sorted(vals)}")
    if field is not None:
        v = rec if isinstance(rec, str) else _get(rec, field)
        if v is None:
            raise RuntimeErr(f"text_dataset: a record has no field '{field}' (it has {_keys(rec)})")
        return [(True, str(v))]
    kind = detect(rec) if fmt == "auto" else fmt
    if kind == "text":
        if isinstance(rec, str):
            return [(True, rec)]
        return [(True, str(next(_get(rec, k) for k in TEXT_KEYS if _get(rec, k) is not None)))]
    if kind == "chat":
        msgs = next(_get(rec, k) for k in CHAT_KEYS if _get(rec, k) is not None)
        turns = _chat_turns(msgs)
        if _get(rec, "system") and not any(r == "system" for r, _ in turns):
            turns.insert(0, ("system", str(_get(rec, "system"))))
        return _render_chat(turns, chat_style)
    if kind == "instruction":
        inp = _get(rec, "input")
        out = _get(rec, "output")
        out = _get(rec, "response") if out is None else out
        if chat_style != "plain":
            turns = [("user", str(_get(rec, "instruction")) + (f"\n\n{inp}" if inp else "")), ("assistant", str(out))]
            return _render_chat(turns, chat_style)
        p = f"### Instruction:\n{_get(rec, 'instruction')}\n\n" + (f"### Input:\n{inp}\n\n" if inp else "") + \
            "### Response:\n"
        return [(False, p), (True, str(out))]
    if kind in ("completion", "qa"):
        q = _get(rec, "prompt") if kind == "completion" else _get(rec, "question")
        a = next((_get(rec, k) for k in ("completion", "response", "chosen", "answer") if _get(rec, k) is not None), "")
        if chat_style != "plain":
            return _render_chat([("user", str(q)), ("assistant", str(a))], chat_style)
        if kind == "qa":
            return [(False, f"Question: {q}\nAnswer: "), (True, str(a))]
        return [(False, str(q)), (True, str(a))]
    strs = [str(_get(rec, k)) for k in _keys(rec) if isinstance(_get(rec, k), str)]
    if not strs:
        raise RuntimeErr(f"text_dataset: cannot tell how to read a record with fields {_keys(rec)}; "
                         "give field=\"name\" or template=\"... {name} ...\"")
    return [(True, "\n".join(strs))]


# ------------------------------------------------------------------ tokenizers
class _Bytes:
    vocab_size = 257
    eos_id = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(int(i) for i in ids if 0 <= int(i) < 256).decode("utf-8", errors="replace")


class _Chars:
    def __init__(self):
        self.stoi, self.itos = {}, []
        self.eos_id = 0
        self.itos.append("")

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text):
        out = []
        for ch in text:
            i = self.stoi.get(ch)
            if i is None:
                i = self.stoi[ch] = len(self.itos)
                self.itos.append(ch)
            out.append(i)
        return out

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids if 0 < int(i) < len(self.itos))


def make_tokenizer(tok):
    if tok is None or tok == "bytes":
        return _Bytes()
    if tok in ("chars", "char"):
        return _Chars()
    if isinstance(tok, str):
        raise RuntimeErr(f"text_dataset: tokenizer '{tok}' is not known: give hub.tokenizer(\"repo\"), "
                         "lm.tokenizer, a function, \"bytes\" or \"chars\"")
    return tok


def _encode(tok, text):
    if callable(getattr(tok, "encode", None)):
        try:
            ids = tok.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tok.encode(text)
    elif callable(tok):
        ids = tok(text)
    else:
        raise RuntimeErr("text_dataset: the tokenizer needs an encode(text) method or must be a function")
    ids = getattr(ids, "ids", ids)
    return [int(i) for i in ids]


def _eos(tok, eos):
    if eos is not None:
        return None if eos is False else int(eos)
    for name in ("eos_id", "eos_token_id"):
        v = getattr(tok, name, None)
        if isinstance(v, (int, np.integer)):
            return int(v)
    added = getattr(tok, "added", None) or {}
    for t in ("<|endoftext|>", "<|im_end|>", "<|eot_id|>", "</s>", "<eos>"):
        if t in added:
            return int(added[t])
    return None


def chat_style_of(tok, style):
    if style not in (None, "auto"):
        return style
    added = getattr(tok, "added", None) or {}
    for mark, name in (("<|start_header_id|>", "llama3"), ("<start_of_turn>", "gemma"), ("<|im_start|>", "chatml")):
        if mark in added:
            return name
    return "plain"


# ------------------------------------------------------------------ the dataset
def token_stream(records, tok, fmt="auto", field=None, template=None, chat_style="plain", eos=None,
                 train_on_prompt=False):
    """(ids, labels) of each record; labels are the ids, -100 where not trained on."""
    eos_id = _eos(tok, eos)
    for rec in records:
        ids, lab = [], []
        for train, text in to_parts(rec, fmt, field, template, chat_style):
            if not text:
                continue
            t = _encode(tok, text)
            ids += t
            lab += t if (train or train_on_prompt) else [IGNORE] * len(t)
        if eos_id is not None:
            ids.append(eos_id)
            lab.append(eos_id)
        if ids:
            yield ids, lab


def windows(stream, seq_len, pack=True, stride=None, pad_id=0):
    """Training windows (x, y) of seq_len tokens; y is x moved by one (the next token at each place).
    pack=true joins records end to end (no padding, nothing wasted); pack=false gives one window per record,
    cut or padded (padding labelled -100)."""
    n = int(seq_len)
    if n < 1:
        raise RuntimeErr("text_dataset: seq_len must be at least 1")
    step = int(stride) if stride else n
    if not pack:
        for ids, lab in stream:
            ids, lab = ids[:n + 1], lab[:n + 1]
            x = np.full(n, pad_id, np.int32)
            y = np.full(n, IGNORE, np.int32)
            x[:len(ids) - 1] = ids[:-1]
            y[:len(lab) - 1] = lab[1:]
            yield x, y
        return
    buf_x, buf_y = [], []
    for ids, lab in stream:
        buf_x += ids
        buf_y += lab
        while len(buf_x) >= n + 1:
            yield np.asarray(buf_x[:n], np.int32), np.asarray(buf_y[1:n + 1], np.int32)
            del buf_x[:step], buf_y[:step]


def text_dataset(source, tokenizer="bytes", seq_len=512, format="auto", field=None, template=None, chat="auto",
                 eos=None, train_on_prompt=False, pack=True, stride=None, text_split="file", path_check=None):
    from .data import Pipeline
    tok = make_tokenizer(tokenizer)
    style = chat_style_of(tok, chat)

    def records():
        if hasattr(source, "_factory") or (isinstance(source, (list, tuple)) and source and
                                           not all(isinstance(s, str) and (_kind(s) or os.path.exists(s) or
                                                                           any(c in s for c in "*?["))
                                                   for s in source)):
            yield from source
            return
        for f in _files(source):
            if path_check:
                f = path_check(f)
            yield from read_records(f, text_split)

    def factory():
        return windows(token_stream(records(), tok, format, field, template, style, eos, train_on_prompt),
                       seq_len, pack, stride)
    p = Pipeline(factory, None, f"text_dataset({format}, seq_len={seq_len})")
    p.tokenizer = tok
    return p
