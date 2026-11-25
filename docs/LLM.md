# Your own language model: `use llm`

`use llm` builds, trains, tunes and talks to a language model in a few lines. It is written in Apsi and is the same
file for `apsi run` and for native programs (`apsi build --native`), where it trains several times faster.

```apsi
use llm
let m = llm.create(data={"faq.jsonl": 3, "notes/": 1}, layers=4, d=256, heads=8, kv_heads=2, experts=4)
m.train(steps=2000)
print(m.chat("What is Apsi?"))
m.save("my_model")
let again = llm.load("my_model")
```

## Data

- `data=` a file, a folder, a glob (`"notes/**/*.txt"`), a list of strings or of `[question, answer]` pairs, or
  `{source: weight}`: a weight is how often a source's records are drawn (data importance; 0 keeps a source out of
  training but its characters in the tokenizer).
- `.txt` / `.md` files are text; `.jsonl` / `.json` rows can be `{"text": …}`, `{"question": …, "answer": …}`,
  `{"prompt": …, "response": …}`, `{"instruction": …, "output": …}` or `{"messages": [{"role": …, "content": …}]}`.
- Question / answer and chat data learn only the answers (the loss is masked on the questions), and training
  windows start at the start of a record, so the model learns to answer rather than to continue the question.
- `m.add_data(source, weight)` adds more later.

## The model

`llm.create(...)` takes `tokenizer` (`"bpe"` with `vocab`, `"chars"`, `"bytes"`), `layers`, `d`, `heads`,
`kv_heads` (grouped-query attention), `ff`, `context`, `experts` / `top_k` / `moe_every` (mixture of experts),
`dropout`, `tie` (tied output embedding), `seed`, `system` (a default system prompt) and any `llm.config` field
(`rope_base`, `aux_weight`, `z_weight`, `dtype`, …).

Inside: token embedding, RMSNorm, rotary positions, grouped-query attention with a key / value cache, SwiGLU
feed-forward, a mixture of experts with top-k routing, sparse dispatch, the load-balancing loss and the router
z-loss. `print(m)` shows the parameters, and how many of them one token uses.

## Training

`m.train(steps, batch, lr, warmup, accumulate, clip, weight_decay, eval_every, val_fraction, log_every, seed,
label_smoothing, checkpoint)`: AdamW with decoupled weight decay on matrices, warm-up then cosine learning rate,
global gradient clipping, gradient accumulation, validation loss and perplexity (with 20 records or more),
checkpoints. `log_every=0` prints only the last step, `verbose=false` nothing. It returns `loss`, `steps`,
`seconds` (and `val_loss`, `perplexity`).

`m.finetune(data, steps, rank, alpha, lr, targets, replay)`: low-rank adaptation (LoRA). The weights stay frozen;
`A @ B` of rank `rank` is learned for the chosen matrices (`wq wk wv wo w1 w3 w2`), a small share of the
parameters. `replay` (0.3) mixes in what the model learned before so new data does not overwrite it. `merge=true`
folds the adapters into the weights, so the tuned model runs as fast as before.

`m.loss(text)`, `m.loss(question, answer)` and `m.perplexity(text)` measure it.

## Talking to it

- `m.chat(message, history, system)`: the most likely answer (greedy); give `temperature`, `top_k`, `top_p` and
  `repetition_penalty` for varied replies. History that does not fit the context loses its oldest turns first;
  the system prompt and the new message always stay, and no turn is cut in half.
- `m.conversation(system, keep_first)`: a chat that remembers. Every turn's keys and values stay cached, so a new
  message costs only its own tokens, not the whole history again (`c.tokens_processed()`). When the context fills
  up the oldest turns leave (the system prompt and the first `keep_first` exchanges are pinned) and the cache is
  rebuilt once.
- `m.generate(prompt, max_new, temperature, …)` continues a text.
- `m.extend_context(factor)` reads longer texts (position interpolation of the rotary angles); a short `train()` at
  the new length makes it fluent there.

## Your own parts

Everything is a plain Apsi function you can copy and change: `llm.forward`, `llm.attention`, `llm.moe`,
`llm.swiglu`, `llm.rms_norm`, `llm.rope_tables`, `llm.token_loss`, `llm.init_params`. A model of your own trains
and talks with the same machinery:

```apsi
fn my_init(cfg, seed) = record(w=tensor(zeros(cfg.vocab, cfg.vocab), "float32"))
fn my_forward(cfg, p, ids, caches, offset, train, seed) = p.w[ids]      # logits, or [logits, aux, z]
let b = llm.create(data="faq.jsonl", tokenizer="chars", init=my_init, forward=my_forward)
b.train(steps=200)
```

`llm.load(path, forward=my_forward)` loads one back.
