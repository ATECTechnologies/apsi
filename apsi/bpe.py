"""Train a byte-level BPE tokenizer (the kind GPT-2, Llama 3 and Qwen use), fast, in Apsi itself.

    use data
    let tok = data.train_tokenizer("corpus/*.txt", vocab_size=16000)
    tok.save("my-tokenizer.json")                 # tokenizer.json: loads in Apsi (hub.tokenizer_from_file) and in
                                                  # Hugging Face `tokenizers` / `transformers`
    let ds = data.text_dataset("corpus/*.txt", tokenizer=tok, seq_len=512)

How it is fast: the text is split into pre-tokens once (the GPT-2 pattern, in the regular-expression engine) and only
the distinct pre-tokens with their counts are kept; every pair of symbols knows the words it occurs in, and a merge
touches only those words, updating the counts of the pairs around each change (no full recount per merge). The best
pair comes from a heap with lazy deletion. Equal counts break ties by the pair's text, so the same corpus always gives
the same tokenizer. 32 000 merges over a few hundred megabytes of text take minutes, not hours. With Hugging Face
`tokenizers` installed (`pip install tokenizers`) training runs in its Rust trainer, several times faster again, and
gives the same file format; engine="apsi" keeps it in Apsi.
"""
import heapq
import json
from collections import Counter, defaultdict

from .hub import Tokenizer, _B2U, _GPT2


def pretokens(texts, lower=False):
    """Distinct byte-level pre-tokens and their counts (the GPT-2 split, then bytes as printable characters)."""
    counts = Counter()
    for t in texts:
        if not isinstance(t, str):
            t = str(t)
        if lower:
            t = t.lower()
        counts.update(_GPT2.findall(t))
    out = Counter()
    for w, c in counts.items():
        out["".join(_B2U[b] for b in w.encode("utf-8"))] += c
    return out


def train(texts, vocab_size=8000, min_frequency=2, special_tokens=("<|endoftext|>",), lower=False, progress=None,
          engine="auto"):
    """A byte-level BPE Tokenizer trained on texts (an iterable of strings). vocab_size counts the 256 bytes, the
    merges and the special tokens. min_frequency: pairs seen fewer times are not merged. progress(merges_done,
    merges_wanted) is called every 1000 merges. engine: "apsi" (this file), "rust" (Hugging Face `tokenizers`, several
    times faster on large corpora, same format) or "auto" (rust when installed)."""
    if engine in ("auto", "rust"):
        try:
            return _train_rust(texts, vocab_size, min_frequency, special_tokens, lower)
        except ImportError:
            if engine == "rust":
                raise
    words = pretokens(texts, lower)
    alphabet = [_B2U[b] for b in range(256)]
    sym = {s: i for i, s in enumerate(alphabet)}
    names = list(alphabet)
    specials = [s for s in (special_tokens or ()) if s]
    want = max(0, int(vocab_size) - len(alphabet) - len(specials))
    seqs, freq = [], []
    for w, c in words.items():
        seqs.append([sym[ch] for ch in w])
        freq.append(c)
    pair_count = defaultdict(int)
    where = defaultdict(set)
    for wi, s in enumerate(seqs):
        f = freq[wi]
        for a, b in zip(s, s[1:]):
            pair_count[(a, b)] += f
            where[(a, b)].add(wi)

    def key(p):
        return (-pair_count[p], names[p[0]], names[p[1]])
    heap = [(key(p), p) for p in pair_count]
    heapq.heapify(heap)
    merges = []
    while len(merges) < want and heap:
        k, p = heapq.heappop(heap)
        cnt = pair_count.get(p, 0)
        if cnt <= 0 or k != key(p):                       # stale entry: the count changed since it was pushed
            if cnt > 0:
                heapq.heappush(heap, (key(p), p))
            continue
        if cnt < min_frequency:
            break
        a, b = p
        new = len(names)
        names.append(names[a] + names[b])
        merges.append((names[a], names[b]))
        touched = set()
        for wi in list(where[p]):
            s, f = seqs[wi], freq[wi]
            i, out = 0, []
            changed = False
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    out.append(new)
                    i += 2
                    changed = True
                else:
                    out.append(s[i])
                    i += 1
            if not changed:
                continue
            for x, y in zip(s, s[1:]):                       # the old pairs of this word go ...
                pair_count[(x, y)] -= f
                touched.add((x, y))
            for x, y in zip(out, out[1:]):                   # ... and its new pairs come
                pair_count[(x, y)] += f
                where[(x, y)].add(wi)
                touched.add((x, y))
            seqs[wi] = out
        pair_count.pop(p, None)
        where.pop(p, None)
        for q in touched:
            c = pair_count.get(q, 0)
            if c > 0:
                heapq.heappush(heap, (key(q), q))
            elif q in pair_count:
                del pair_count[q]
        if progress is not None and len(merges) % 1000 == 0:
            progress(len(merges), want)
    vocab = {n: i for i, n in enumerate(names)}
    added = [{"id": len(names) + j, "content": s, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": True} for j, s in enumerate(specials)]
    spec = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": {"type": "Lowercase"} if lower else None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": [f"{x} {y}" for x, y in merges]},
    }
    return TrainedTokenizer(spec)


def _train_rust(texts, vocab_size, min_frequency, special_tokens, lower):
    from tokenizers import Tokenizer as HF, models, normalizers, pre_tokenizers, trainers, decoders
    hf = HF(models.BPE())
    if lower:
        hf.normalizer = normalizers.Lowercase()
    hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    hf.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=int(vocab_size), min_frequency=int(min_frequency),
                             special_tokens=list(special_tokens or ()), show_progress=False,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    hf.train_from_iterator((t if isinstance(t, str) else str(t) for t in texts), tr)
    return TrainedTokenizer(json.loads(hf.to_str()))


class TrainedTokenizer(Tokenizer):
    """A Tokenizer that can also be saved (tokenizer.json) and knows its size and end-of-text id."""

    @property
    def vocab_size(self):
        return len(self.inv)

    @property
    def eos_id(self):
        for t in ("<|endoftext|>", "</s>", "<eos>"):
            if t in self.added:
                return self.added[t]
        return None

    def encode(self, text, pair=None, add_special_tokens=False):
        return Tokenizer.encode(self, text, pair, add_special_tokens).ids

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.spec, fh, ensure_ascii=False)
        return path


def train_from(source, vocab_size=8000, min_frequency=2, special_tokens=("<|endoftext|>",), lower=False,
               format="auto", field=None, template=None, text_split="file", path_check=None, progress=None,
               engine="auto"):
    """train() over what text_dataset reads: files, globs, folders, jsonl / csv / parquet records or strings."""
    from .textdata import _files, _path_like, read_records, to_parts

    def texts():
        src = source
        if isinstance(src, str) and not _path_like(src):
            yield src
            return
        if isinstance(src, (list, tuple)) and src and not all(_path_like(s) for s in src):
            for rec in src:
                yield from _record_texts(rec, format, field, template)
            return
        for f in _files(src):
            if path_check:
                f = path_check(f)
            for rec in read_records(f, text_split):
                yield from _record_texts(rec, format, field, template)

    def _record_texts(rec, fmt, fld, tpl):
        if isinstance(rec, str):
            yield rec
            return
        for _, t in to_parts(rec, fmt, fld, tpl):
            yield t
    return train(texts(), vocab_size, min_frequency, special_tokens, lower, progress, engine)
