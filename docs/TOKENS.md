# Fewer tokens, nothing lost (`use tokens`)

Language models charge by the token and read slower the more they get. `use tokens` removes redundancy from what
you send — **with transforms that are exactly reversible**, so no information is summarised, dropped or guessed:

| transform | reverse | what it removes | measured |
|---|---|---|---|
| `pack(value)` | `unpack(text)` | repeated JSON keys, quotes and spacing: a list of records becomes a table with the keys once and one compact line per record | 200 orders: pretty JSON 9,802 → 3,625 tokens (**63 % fewer**); minified JSON 8,401 → 3,625 (57 %) |
| `dedupe(text)` | `expand(text)` | lines that repeat: written once in a header, referenced as `§N` | a 500-line log: 5,689 → 2,339 (**59 %**) |
| `diff(old, new)` | `patch(old, delta)` | resending a whole document when a few lines changed | a 5,951-token document with two edits: 19 tokens (**99.7 %**) |
| `best(value)` | `restore(text, method)` | tries the transforms that apply and keeps the smallest | |

```apsi
use tokens
let rows = load_json("orders.json")
let text = tokens.pack(rows)
print(tokens.savings(json_str(rows, indent=2), text))   # 9,802 → 3,625 tokens: 63.0 % saved (2.70× fewer)
assert tokens.unpack(text) == rows                        # exact, always

let prompt = "Orders as a table (first line = column names, one JSON-encoded row per line):\n" + text
```

`tokens.count(text)` is exact when `tiktoken` is installed (`apsi pkg install tiktoken`) and a close estimate
(±10–15 %) otherwise; the measurements above are estimates.

## What to expect

Savings depend on how much redundancy the data has — there is no transform that shrinks everything:

* **Large:** lists of records with the same fields (API responses, database rows, logs as JSON), logs and transcripts
  with repeated lines, documents sent again after small edits.
* **None:** prose, source code without repetition, records whose fields differ from one to the next. `best()` then
  returns the input unchanged, so it never costs extra.

Models read the packed forms well when the prompt says what they are (one sentence, as above). `diff` is meant for
a model that already has the previous version — in the conversation or in a provider's prompt cache.
