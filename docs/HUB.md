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
| `hub.model(repo, device="auto", gpu_layers=, memory="auto", update=false)` / `hub.model_from_folder(path, device=, gpu_layers=, memory=)` | a model that runs natively (below); decoders use an NVIDIA GPU when there is one. A model downloaded before starts from the cache with no network request (the Hub is asked for a newer version at most once a day, or now with `update=true`) |
| `hub.devices()` | the NVIDIA GPUs the hub can use: `[{"index", "name", "memory_gb", "compute_capability", "multiprocessors"}]`, or `[]` |
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
* `apsi cache` shows every cache and its size, `apsi cache models` the downloaded models, `apsi cache clean` removes
  what can never be used again (kernels of other versions, downloads stopped long ago, unused int8 copies) and
  `apsi cache clean models --model owner/name` one model.

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
* **`chat(messages, system=, max_new_tokens=512, temperature=0.7, top_p=0.9, repetition_penalty=1.1, stream=)`** —
  the model's reply in a conversation. `messages` is a text (one user message) or a list of
  `{"role": "user" | "assistant" | "system", "content": text}`. The prompt is written in the model's own chat format,
  read from its chat template (ChatML for Qwen and SmolLM, Llama 3, Gemma, Mistral `[INST]`); generation stops at
  the end of the turn; when the conversation no longer fits the model's window the oldest turns are left out.
  `chat_format()` names the format, `chat_prompt(messages, system=)` shows the exact text the model sees.
* **`conversation(system=, **options)`** — a conversation that remembers its turns: `c.say(text, stream=fn)` returns
  the reply, `c.reset(system=)` starts over, `c.history` is the list of messages, `c.transcript()` is readable text.
  A failed or interrupted turn leaves no half entry behind.
* `m.last_tokens` — the number of tokens the last `generate()` / `chat()` produced; `m.device` — where it runs.

How it is checked: tiny checkpoints written in the exact Hugging Face file layout are run in the test suite. The
encoder is compared with an independent per-head reference implementation, padding is checked not to change results,
and cached generation is compared with recomputing the whole sequence at every step. On a machine with
`transformers` and `torch`, `APSI_HF_COMPARE=1 python -m unittest tests.test_hub_models` also compares against the
reference library.

A chat assistant in a few lines:

```
use hub
let lm = hub.model("Qwen/Qwen2.5-1.5B-Instruct")
let c = lm.conversation(system="You are a concise assistant.")
fn show(piece):
    print(piece, end="")
c.say("What is a prime number?", stream=show)
c.say("Give three examples.", stream=show)       # remembers the question before
```

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

Encoders run on the CPU in float32 with NumPy's multi-threaded BLAS and handle sentences in milliseconds.

Decoders (`generate`, `chat`) run on an **NVIDIA graphics card by themselves** when one is present, with nothing
to install but the display driver: no CUDA Toolkit, CuPy or PyTorch. The whole network runs on the card: the weights,
the key/value cache, norms, rotary embeddings, attention and the MLP, in kernels that Apsi generates (as PTX, like
`@gpu`) and the driver compiles for the card in use. Per new token only its embedding goes to the card and the
next-token scores come back. Any GPU from the GTX 1660 / RTX 20xx on (compute capability 7.5) works.

* **Precision.** Weights are kept as bfloat16, the format Qwen, Llama, Gemma and Mistral are published in, so for
  them the copy on the card is exact; sums and activations are float32. A 0.5B model needs about 1 GB, 1.5B about
  3 GB, 3B about 6 GB, 7B about 15 GB.
* **Start-up.** A bfloat16 checkpoint goes from the file to the card as it is (no float32 copy is ever made); the
  driver starts and the kernels load while the file is read; the kernels are compiled once and kept in Apsi's cache
  folder; a model downloaded before needs no network request.
* **Measured** on an RTX 3060 with Qwen2.5-0.5B: 193 tokens/s greedy, 190 sampled, 211 through `chat`; a prompt
  of 512 tokens in 118 ms; a token after 2,000 tokens of context in 5.5 ms; ready 0.86 s after `hub.model(...)`.
* **Prompts** are processed in passes of up to 512 positions with a tiled matrix product; single tokens with a product
  that normalises its input on the fly, so a token is about 170 kernel launches with no work on the processor
  between them.
* **Larger than the card.** With `device="auto"` the first layers that fit in the card's free memory run on the
  GPU and the rest on the processor; `lm.device` says how many (`cuda:0 (...) + cpu (20 of 28 layers on the GPU)`).
  `gpu_layers=N` sets the number. `memory="disk"` maps the weights from the file (an SSD) instead of reading them
  into RAM, for models larger than the memory; `"auto"` does that by itself when the model is larger than half of
  the free memory.
* **Choosing the next token on the card.** With a temperature, the repetition penalty, the temperature and the
  softmax denominator are computed on the GPU, and only the tokens that top-k / top-p can pick come back (a few
  dozen numbers instead of the 150,000 scores of a Qwen vocabulary); a very flat distribution falls back to the
  processor. The choice rule is the same as on the processor.
* **Conversations.** A prompt that starts with the tokens of the previous generation (a conversation re-sends its
  earlier turns) reuses that generation's key/value cache: only the new part is computed. On the GPU and on the
  processor alike.
* **Failures.** Running out of GPU memory frees the scratch space and goes on in smaller passes; any other driver
  error moves the model to the processor, which carries on from the tokens already produced (a streamed answer sees
  every token once). It is reported once per kind of failure; `APSI_QUIET=1` silences it. One model can serve
  several threads; they take turns.

```
let lm = hub.model("Qwen/Qwen2.5-3B-Instruct")                 # device="auto": the GPU when there is one
print(lm.device)                                               # cuda:0 (NVIDIA GeForce RTX ...) or cpu
let cpu = hub.model("Qwen/Qwen2.5-0.5B-Instruct", device="cpu")
let big = hub.model("Qwen/Qwen2.5-7B-Instruct", gpu_layers=20, memory="disk")
lm.to("cpu")                                                   # move it later; returns the device
```

`device=` is `"auto"` (default), `"cpu"`, `"cuda"` or `"cuda:N"`. With `"auto"`, a machine without a usable card
stays on the CPU and `lm.device_note` says why; `"cuda"` raises an error instead, also when the model does not fit
the card. `APSI_NO_GPU=1` keeps everything on the CPU.

On the processor, a bfloat16 checkpoint runs on all cores with matrix products compiled by Apsi's native compiler.
By default the matrices are kept as **int8 with one scale per 32 values** (the Q8_0 layout of llama.cpp): a token reads
half the bytes of bfloat16, so it is about twice as fast where memory is the limit, and the model needs half the RAM
(a 0.5B model about 0.55 GB). The error is below bfloat16's own rounding of the activations. The int8 copy is made
the first time (a few seconds) and kept in `~/.apsi/cache/cpu-int8`, memory-mapped on later starts.
`APSI_CPU_WEIGHTS=bf16` keeps the exact bfloat16 weights. The products start on every core through a native thread
team (about a microsecond, instead of tens of microseconds per thread from Python). Prompts of 24 tokens or more use
NumPy's BLAS. Other checkpoints run in float32 with NumPy. `APSI_HUB_NATIVE=0` turns the native products off.

On NVIDIA cards from the RTX 30 series (Ampere) on, prompts use the **tensor cores** (`mma.sync`, bfloat16 inputs,
float32 sums); the activations are split into two bfloat16 parts, so the result matches the float32 product to about
1e-5. The kernel is checked against the float32 one when the GPU starts and is used only if they agree;
`APSI_TENSOR_CORES=0` turns it off. For bigger models, use GGUF with llama.cpp, or PyTorch through `hf_pipeline`.

### Not native yet

* Gemma 2 and 3 (extra norms and soft-capping), Phi, Falcon, MPT and mixture-of-experts models, T5 and BART
  (encoder–decoder), vision and audio models.
* Chat templates other than ChatML, Llama 3, Gemma and Mistral (the Jinja in `tokenizer_config.json` is not run):
  `chat()` falls back to a plain "User: / Assistant:" transcript; write the prompt yourself with `generate()`.
* Quantised GGUF formats other than Q8_0 and Q4_0 for `hub.load`.

For those, `hf_pipeline(task, model)` in `use models` runs Python's `transformers` (needs `--allow-python`).
PyTorch `.bin` checkpoints are pickles and can run code when loaded; `hub.load` refuses them. Download them with
`hub.download` and open them with `torch_load` under `--allow-python` only if you trust the publisher.
