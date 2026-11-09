# Hugging Face models without Python (`use hub`)

`use hub` downloads models, datasets and tokenizers from the Hugging Face Hub with Apsi's own HTTP client, loads
weights into Apsi tensors and tokenizes text natively. You do not need Python's `transformers`, `huggingface_hub`
or `tokenizers` packages, and you do not need `--allow-python`.

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
| Unigram | T5, ALBERT, XLM-R | not yet — use `hf_pipeline` with `--allow-python` |

`encode(text, pair=none, add_special_tokens=true)` returns `ids`, `tokens` and `attention_mask`; special tokens are
added by the tokenizer's own post-processor template. `decode(ids, skip_special_tokens=true)` turns ids back into
text.

Pre-tokenizer patterns written with `\p{L}` / `\p{N}` (Llama 3, Qwen) are translated to Python's regular
expressions; letters of the common scripts (Latin, Greek, Cyrillic, Arabic, Hebrew, Devanagari, Thai, Hangul, kana,
CJK) are covered, so very rare scripts can split differently from the Rust `tokenizers` library.

## Running the model

`use hub` gives you the weights and the tokenizer. To run the network:

* write the architecture with `use layers` (the weight names are in `hub.load(repo).keys()`), or
* load a GGUF model with `use models` (`load_gguf`), or an ONNX export with `onnx_model`, or
* use `hf_pipeline("sentiment-analysis")` from `use models`, which needs Python's `transformers` and
  `--allow-python`.

PyTorch `.bin` checkpoints are pickles and can run code when loaded; `hub.load` refuses them. Download them with
`hub.download` and open them with `torch_load` under `--allow-python` only if you trust the publisher.
