# Hugging Face models without Python (`use hub`)

`use hub` downloads models from the Hugging Face Hub, tokenizes text and **runs the models natively** — embeddings,
classification, named entities, fill-mask and text generation — with Apsi's own code. You do not need Python's
`transformers`, `torch`, `huggingface_hub` or `tokenizers` packages, and you do not need `--allow-python`.

```apsi
use hub

let sentiment = hub.pipeline("sentiment-analysis", "distilbert/distilbert-base-uncased-finetuned-sst-2-english")
print(sentiment("Apsi is a joy to use"))                 # {"label": "POSITIVE", "score": 0.99…}

let st = hub.model("sentence-transformers/all-MiniLM-L6-v2")
print(st.similarity("A cat sits on the mat", "A kitten is on the rug"))
let vectors = st.embed(["first text", "second text"])     # [2, 384], normalised like sentence-transformers

let lm = hub.model("HuggingFaceTB/SmolLM2-135M")
print(lm.generate("The three laws of motion are", max_new_tokens=40))
lm.generate("Once upon a time", max_new_tokens=60, temperature=0.8, top_p=0.95, stream=fn(t) => print(t, end=""))
```

```apsi
use hub

print(hub.search("sentence similarity", task="sentence-similarity", limit=5))
print(hub.files("sentence-transformers/all-MiniLM-L6-v2"))

let cfg = hub.config("sentence-transformers/all-MiniLM-L6-v2")              # config.json as a dict
let w = hub.load("sentence-transformers/all-MiniLM-L6-v2")                  # {name: tensor} from safetensors
print(w["embeddings.word_embeddings.weight"].shape)

let tok = hub.tokenizer("sentence-transformers/all-MiniLM-L6-v2")           # tokenizer.json, natively
let e = tok.encode("Apsi reads like Python.")
print(e.tokens, e.ids, e.attention_mask)
print(tok.decode(e.ids))
```

Run it with `apsi run app.apsi --allow-net=huggingface.co`. Downloads of large files are redirected by the Hub to
its storage hosts; the permission for `huggingface.co` covers them.

## Functions

| | |
|---|---|
| `hub.search(query, task=, limit=20, sort="downloads")` | models matching a query: id, task, downloads, likes |
| `hub.info(repo, revision=, kind="model")` | the Hub's metadata for a model, dataset (`kind="dataset"`) or Space |
| `hub.files(repo, revision=)` | the files in the repository |
| `hub.download(repo, file, revision="main")` | one file → its local path |
| `hub.snapshot(repo, patterns=["*.json", "*.safetensors", "*.txt", "*.model"])` | the matching files → a folder |
| `hub.config(repo)` | `config.json` as a dict |
| `hub.load(repo)` / `hub.load(repo, "file.gguf")` | weights as `{name: tensor}`: `model.safetensors`, sharded safetensors (`model.safetensors.index.json`), or a GGUF file |
| `hub.tokenizer(repo)` / `hub.tokenizer_from_file(path)` | a tokenizer from `tokenizer.json` |
| `hub.model(repo)` / `hub.model_from_folder(path)` | a model that runs natively (below) |
| `hub.pipeline(task, repo)` | a function for one task: `"sentiment-analysis"` / `"text-classification"`, `"feature-extraction"` / `"sentence-similarity"`, `"ner"` / `"token-classification"`, `"fill-mask"`, `"text-generation"` |
| `hub.login(token)` / `hub.logout()` | a Hub access token for private and gated models |
| `hub.offline(true)` | use only the local cache |
| `hub.cache_dir()` / `hub.clear_cache(repo)` | where files are kept / remove them |

## Tokens, private and gated models

```apsi
use env
hub.login(env.secret("HF_TOKEN"))              # or set HF_TOKEN in the environment, or pass token= to a call
let w = hub.load("my-org/private-model")
```

* The token is sent only to the Hub's own address, never to the storage host a download is redirected to.
* Gated models (Llama, Gemma …) need you to accept their terms on the model page once; after that the token works.
* `HF_ENDPOINT` points the client at a mirror or a self-hosted Hub.

## Cache, resuming and checking

* Files are kept in `~/.apsi/hub` (or `APSI_HUB_CACHE`), one folder per commit, so a model that changes on the Hub
  never mixes old and new files.
* An interrupted download continues where it stopped the next time you ask for the file.
* Large files are checked against the SHA-256 the Hub publishes; a file that does not match is deleted, never used.
* `hub.offline(true)` (or `APSI_HUB_OFFLINE=1`) answers from the cache only — for machines without internet.

## Tokenizers

`tokenizer.json` files of the Hugging Face `tokenizers` format are read natively:

| Family | Examples | Supported |
|---|---|---|
| WordPiece with the BERT normaliser | BERT, DistilBERT, MiniLM, E5, BGE | yes |
| Byte-level BPE | GPT-2, RoBERTa, Llama 3, Qwen, Mistral Nemo | yes |
| SentencePiece-style BPE with byte fallback | Llama 2, Mistral 7B, Gemma | yes |
| Unigram (SentencePiece) | T5, ALBERT, XLM-R, mBART | yes (the `Precompiled` normaliser is applied as NFKC) |

`encode(text, pair=none, add_special_tokens=true)` returns `ids`, `tokens` and `attention_mask`; special tokens are
added by the tokenizer's own post-processor template. `decode(ids, skip_special_tokens=true)` turns ids back into
text.

Pre-tokenizer patterns written with `\p{L}` / `\p{N}` (Llama 3, Qwen) are translated to Python's regular
expressions; letters of the common scripts (Latin, Greek, Cyrillic, Arabic, Hebrew, Devanagari, Thai, Hangul, kana,
CJK) are covered, so very rare scripts can split differently from the Rust `tokenizers` library.

## Running models natively

`hub.model(repo)` downloads `config.json`, the safetensors weights (also sharded; F32, F16 and BF16), the tokenizer
and, for sentence-transformers models, their pooling settings, then runs the network with NumPy.

| Family (`model_type`) | Examples | Methods |
|---|---|---|
| `bert`, `distilbert`, `roberta`, `xlm-roberta`, `camembert` | BERT, DistilBERT, RoBERTa, XLM-R, MiniLM, E5, BGE, GTE and other sentence-transformers built on them | `embed`, `similarity`, `classify`, `tokens`, `fill_mask`, `hidden` |
| `gpt2` | GPT-2, DistilGPT-2 | `generate` |
| `llama`, `mistral`, `qwen2`, `qwen3`, `gemma` | Llama 2 / 3 (with Llama 3 RoPE scaling), TinyLlama, SmolLM, Mistral, Qwen 2 / 2.5 / 3, Gemma | `generate` |

* **`embed(texts, normalize=)`** — sentence vectors with the model's own pooling (mean, CLS or max) and, when
  the model says so, L2 normalisation, as sentence-transformers does. **`similarity(a, b)`** — cosine similarity.
* **`classify(texts, top_k=1)`** — labels from the checkpoint's own head and `id2label` (softmax, or sigmoid for
  multi-label models).
* **`tokens(text)`** — one label per token (named-entity models). **`fill_mask(text, top_k=5)`** — the best words
  for `[MASK]` / `<mask>`.
* **`generate(prompt, max_new_tokens=64, temperature=0, top_k=0, top_p=1, repetition_penalty=1, seed=, stream=)`**
  — text generation with a key/value cache. `temperature=0` is greedy. It stops at the model's end-of-sequence token
  (from `generation_config.json` or `config.json`). `stream` receives each new piece of text as it appears.
* **`hidden(texts)`** — the last hidden states, for your own heads.

How it is checked: tiny checkpoints written in the exact Hugging Face file layout are run in the test suite. The
encoder is compared with an independent per-head reference implementation, padding is checked not to change results,
and cached generation is compared with recomputing the whole sequence at every step. On a machine with
`transformers` and `torch`, `APSI_HF_COMPARE=1 python -m unittest tests.test_hub_models` also compares against the
reference library.

### Methods and what they return

| Call | Input | Returns |
|---|---|---|
| `m.embed("text")` / `m.embed(["a", "b"])` | one text or a list | `Tensor[H]` / `Tensor[B, H]` (float32) |
| `m.similarity(a, b)` | two texts | `Float` in −1 … 1 (cosine of the normalised embeddings) |
| `m.classify("text", top_k=1)` | one text or a list | `{"label": "POSITIVE", "score": 0.998}`; with `top_k > 1` a list of those, best first; for a list of texts, one entry per text |
| `m.tokens("text")` | one text | `[{"token": "Paris", "label": "B-LOC", "score": 0.99}, …]` — special tokens are left out; sub-word pieces keep the tokenizer's spelling (`##ton`, `Ġworld`) |
| `m.fill_mask("Paris is the [MASK] of France.", top_k=5)` | one text with the model's mask token | `[{"token": "capital", "id": 3007, "score": 0.97}, …]` |
| `m.generate(prompt, …)` | a text, or a list of token ids | the generated continuation as text (without the prompt) |
| `m.hidden(texts)` | one text or a list | `Tensor[T, H]` / `Tensor[B, T, H]`, padding included (positions past each text are padding) |
| `m.tokenizer`, `m.config`, `m.labels` | | the tokenizer, `config.json` as a dict, `{id: label}` |

Texts longer than the model's `max_position_embeddings` are cut at that length. Lists are padded to the longest text
and processed as one batch, which is several times faster than one call per text.

`generate` options:

| Option | Default | Meaning |
|---|---|---|
| `max_new_tokens` | 64 | stop after this many new tokens |
| `temperature` | 0 | 0 = always the most likely token (deterministic); 0.7–1.0 for varied text |
| `top_k` | 0 (off) | sample only from the k most likely tokens |
| `top_p` | 1.0 (off) | sample from the smallest set of tokens whose probabilities add up to p |
| `repetition_penalty` | 1.0 (off) | values like 1.1–1.3 discourage repeating tokens already in the text |
| `seed` | none | a number makes sampling repeatable |
| `stream` | none | a function that receives each new piece of text as soon as it is decoded |
| `add_special_tokens` | true | whether the tokenizer adds its start token (`<s>`, `<|begin_of_text|>`) |

### Recipes

**Semantic search over your documents**

```apsi
use hub
let st = hub.model("sentence-transformers/all-MiniLM-L6-v2")
let docs = ["How to reset a password", "Shipping times to Europe", "Refunds for damaged items"]
let index = st.embed(docs)                              # [3, 384], normalised
let q = st.embed("my parcel arrived broken")
let scores = index @ q                                   # cosine similarity, because both are normalised
let best = argmax(scores)
print(docs[best], scores[best])                          # Refunds for damaged items
```

For many documents, embed them in lists of 32–128 and keep the matrix (`save_npy`, or `use memory` for more than
fits in RAM).

**Classifying a batch**

```apsi
let clf = hub.pipeline("sentiment-analysis", "distilbert/distilbert-base-uncased-finetuned-sst-2-english")
for r in clf(["great product", "arrived late and broken"]):
    print(r["label"], round(r["score"], 3))
```

**Named entities, joined into words**

```apsi
let ner = hub.model("dslim/bert-base-NER")
var words = []
for t in ner.tokens("Ada Lovelace was born in London"):
    if t["token"].startswith("##") and len(words) > 0:
        words[-1]["token"] += t["token"][2:]
    elif t["label"] != "O":
        words.append(t)
print(words)          # Ada (B-PER), Lovelace (I-PER), London (B-LOC)
```

**Streaming text as it is generated**

```apsi
let lm = hub.model("Qwen/Qwen2.5-0.5B-Instruct")
let prompt = "<|im_start|>user\nName three uses of lasers.<|im_end|>\n<|im_start|>assistant\n"
lm.generate(prompt, max_new_tokens=120, temperature=0.7, top_p=0.9, seed=1, stream=fn(t) => print(t, end=""))
```

Chat models expect their own prompt format (above: Qwen's). The format is in the model card and in
`tokenizer_config.json` under `chat_template`.

**A model you already have on disk** (downloaded elsewhere, or your own fine-tune saved by `transformers`)

```apsi
let m = hub.model_from_folder("models/my-classifier")   # config.json, model.safetensors, tokenizer.json
```

### Errors you may see

| Message | Meaning and fix |
|---|---|
| a permission error naming `huggingface.co` | run with `--allow-net=huggingface.co`, or `hub.offline(true)` after a first download |
| `401` / `403` for a model | private or gated: `hub.login(token)` and accept the model's terms on its page |
| `model type 'phi3' does not run natively yet` | see "Not native yet" below; use `hf_pipeline` with `--allow-python` |
| `embed() needs an encoder model` | `embed`, `classify`, `tokens`, `fill_mask` are for encoders; `generate` for decoders |
| `the text has no mask token` | write the model's mask token: `[MASK]` (BERT) or `<mask>` (RoBERTa, XLM-R) |
| `… has no safetensors or GGUF weights (only PyTorch .bin files …)` | look for a safetensors version of the model, or see the note on `.bin` files below |

### Speed and size

Everything runs on the CPU in float32 with NumPy's multi-threaded BLAS. Encoders such as MiniLM and BERT-base
handle sentences in milliseconds; decoders up to about 1–2 billion parameters generate at interactive speed on a
modern CPU (a 1.1B model needs about 4.4 GB of RAM in float32). For bigger models, use GGUF with llama.cpp, or
PyTorch through `hf_pipeline`.

### Not native yet

* Gemma 2 and 3 (extra norms and soft-capping), Phi, Falcon, MPT and mixture-of-experts models, T5 and BART
  (encoder–decoder), vision and audio models.
* Chat templates (Jinja in `tokenizer_config.json`): write the prompt in the model's format yourself.
* Quantised GGUF formats other than Q8_0 and Q4_0 for `hub.load`.

For those, `hf_pipeline(task, model)` in `use models` runs Python's `transformers` (needs `--allow-python`).
PyTorch `.bin` checkpoints are pickles and can run code when loaded; `hub.load` refuses them. Download them with
`hub.download` and open them with `torch_load` under `--allow-python` only if you trust the publisher.
