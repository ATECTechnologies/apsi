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

## Why use it

* **Lower cost, same answer.** You pay for every input token. A packed list of 200 records costs about a third of the
  pretty-printed JSON, and the model receives exactly the same data.
* **More fits in the context window.** A 128k-token window holds about 2.7 times as many packed records.
* **Faster replies.** Time to first token grows with the prompt length; a shorter prompt starts answering sooner.
* **Safe by construction.** Each transform has an exact inverse, and the test suite checks
  `unpack(pack(x)) == x`, `expand(dedupe(t)) == t` and `patch(old, diff(old, new)) == new` on random data. There is
  no setting that trades accuracy for size.

## Reference

| Function | Returns | What it does |
|---|---|---|
| `count(text, model="gpt-4o")` | `Int` | Tokens in `text` (a non-string value is counted as minified JSON). Exact when `tiktoken` is installed, otherwise an estimate. `model` picks the tiktoken encoding; unknown names use `o200k_base`. |
| `exact()` | `Bool` | `true` when `count()` is exact. |
| `savings(before, after, model=)` | `Savings` | Counts both texts. Fields: `before`, `after`, `saved`, `percent`, `ratio`. Prints as `9,802 → 3,625 tokens: 63.0 % saved (2.70× fewer)`. |
| `pack(value)` | `Str` | A list of two or more records with the same keys in the same order becomes a table: `@table ["id","name"]` followed by one line per record (`1,"ann"`). Any other JSON value becomes minified JSON. |
| `unpack(text)` | value | The exact inverse of `pack`. |
| `pack_jsonl(text)` | `Str` | JSON Lines (one object per line) → `pack` of the records. Unpacks to the list of parsed objects. |
| `dedupe(text, min_chars=24, min_count=2)` | `Str` | Lines of at least `min_chars` characters that occur at least `min_count` times are written once in an `@refs N` header and replaced by `§1`, `§2` … where they occur. A `§` already in the text is escaped as `§§`. Text with no repeats becomes `@plain` + the text. |
| `expand(text)` | `Str` | The exact inverse of `dedupe`. |
| `diff(old, new)` | `Str` | Line operations that turn `old` into `new`: `=N` keep N lines, `-N` drop N lines, `+text` insert a line. |
| `patch(old, delta)` | `Str` | Applies a `diff` result. |
| `best(value, model=)` | `(Str, Str)` | Tries the transforms that apply (`plain`, `dedupe` for text; `pack`, `pack+dedupe` for other values) and returns the smallest text and the method name. |
| `restore(text, method)` | value | The inverse of `best`, given its method name. |

### The packed table format

```text
@table ["id","customer","total","paid"]
1,"ann",12.5,true
2,"bob",8,false
3,"cid",null,true
```

The first line holds the column names as a JSON array. Every following line is the values of one record, JSON-encoded
and separated by commas, without the surrounding brackets. Strings keep their JSON quotes and escapes, so a comma or a
line break inside a value cannot be confused with a separator. Nested values (lists, objects) stay as JSON inside
the row.

Records are packed as a table only when every record has the same keys **in the same order**; otherwise the whole
value is minified JSON, which is still smaller than pretty-printed JSON and still exact. To get the table form for
data whose key order varies, put the keys in one order first:

```apsi
let keys = ["id", "customer", "total"]
let rows2 = [{k: r.get(k) for k in keys} for r in rows]
```

## Recipes

**Records into a prompt**

```apsi
use tokens
let rows = db.query("select id, customer, total from orders where day = ?", today)
let prompt = "Orders (first line: column names; then one JSON row per line):\n" + tokens.pack(rows) +
             "\n\nWhich customers spent more than 100 in total?"
```

**A long log**

```apsi
let log = read_text("server.log")
let short = tokens.dedupe(log, min_chars=30)
print(tokens.savings(log, short))
let prompt = "Server log. Lines starting with § refer to the numbered lines after @refs.\n" + short
```

**A document that changes while you talk about it**

```apsi
var doc = read_text("spec.md")
# … the model has already seen doc earlier in the conversation …
let edited = read_text("spec.md")
let prompt = "The spec changed. Line operations against the previous version " +
             "(=N keep, -N delete, +text insert):\n" + tokens.diff(doc, edited)
doc = edited
```

**Let Apsi choose**

```apsi
let text, method = tokens.best(payload)
send(text)                                   # and keep method to restore: tokens.restore(text, method)
```

## When it still does not fit: `focus` (built on GLA)

The transforms above never lose anything, so they cannot shrink text that has no repetition — a large source file,
a long manual. When such a context is longer than your budget, `tokens.focus` keeps **the parts the question needs**
and replaces the rest by short numbered markers that say exactly what was left out. Nothing is lost for good:
`unfold` puts any part back, byte for byte.

```apsi
use tokens
let src = read_text("lexer.py")                                   # 3,886 tokens
let f = tokens.focus(src, "how are string escapes handled?", budget=1500)
print(f.saved)                    # 3,886 → 1,493 tokens: 61.6 % saved (2.60× fewer)
send(f.text)                      # the parts that matter, in their original order, with markers for the rest
let full = tokens.unfold(f)       # == src, exactly; tokens.unfold(f, [2]) writes out only part 2
```

What the model receives looks like this (a Python file; text gets the same markers without `#`):

```text
# [[map: 5 parts whole, 0 outlined; shortened parts 1, 2, 3 can be asked for by number]]
"""Tokenizer for Apsi: indentation-sensitive, Unicode identifiers, unit literals."""
import re
…
# [[omitted 1: lines 23-72: class Token, def _ident_char, def tokenize, def _tokenize, def _tokenize (part)]]
…the start of _tokenize, whole…
# [[omitted 2: lines 79-179: def _tokenize (part)]]
        if c == '"' or (c in "fb" and i + 1 < n and src[i + 1] == '"'):
            …the string and escape handling, whole…
# [[omitted 3: lines 230-296]]
```

A function that does not fit whole can appear as its outline instead:

```text
# [[outline 4: lines 610-622]]
def gla_loss(outs, y, d=None, experts=None, weights=None):
    """GLA 3.0's training loss on the outputs of gla(): writer + 0.3 validity + 0.3 issues + 0.2 alignment + router"""
    # calls: _gla_rest, arange, asarray, astype, int, len, log_softmax, raw, reshape, shape, sum, sum_
    ...
```

### How it chooses

It reads the context the way GLA reads code and text:

1. **Units.** Python is cut along its syntax tree: top-level statements, functions, classes, methods; a function or
   block longer than 60 lines is cut further into its header and the statements of its body, so a long function can
   keep only the part that matters. Other code and text are cut into blocks between blank lines and at headings, and
   long paragraphs into sentences. The units cover the context exactly once.
2. **A graph.** For Python, GLA's code property graph (`gla_code_graph`): calls, data flow from definitions to uses,
   dependences — so a function brings the functions it calls and the names it reads. For text, neighbouring units and
   units that share rare words.
3. **Scores.** Each unit scores by the question's words it contains — rarer words count more, `snake_case` and
   `camelCase` names are split into their pieces, a unit whose name the question mentions scores high — and the
   scores then spread over the graph (a personalised PageRank), so what the relevant part depends on scores too.
4. **The budget.** Units are taken by score until the budget (in the model's own tokens, `tokens.count`) is full.
   A function that does not fit whole may stay as its **outline**: its signature, the first line of its docstring
   and the names it calls. The markers' own tokens are counted, so the result is always within the budget.

A context that already fits is returned unchanged.

### Conversations: `fit_chat`

```apsi
let msgs = tokens.fit_chat(history, budget=8000, keep_last=4)
```

The system messages and the last `keep_last` messages stay word for word; older messages are focused on the latest
user message, each in proportion to its size; only if that is still too long are the oldest dropped. It works with any
model: Apsi's own (`use hub`), a model behind an API, or a model you trained — it is plain text in and out.

### Reference

| Function | Returns | What it does |
|---|---|---|
| `focus(context, query="", budget=2000, model="gpt-4o", keep=[], kind="auto", map=true, outlines=true)` | `Focus` | The parts of `context` that matter for `query` within `budget` tokens. `keep`: line numbers or names (`"def parse"`) that must stay whole. `kind`: `"auto"`, `"python"` or `"text"`. `map`: a first line listing the shortened parts. `outlines`: allow outlines. |
| `Focus` | | `.text` (what to send), `.tokens`, `.saved` (a `Savings`), `.kept`, `.outlined`, `.omitted` (unit numbers), `.units`, `.blocks` (number → what the text shows, the original lines) |
| `unfold(f, which=none)` | `Str` | The focused text with the shortened parts written out again: all of them (= the original, exactly) or the numbers in `which`. |
| `fit_chat(messages, budget=8000, model="gpt-4o", keep_last=4)` | list | A conversation within `budget` tokens (above). |
| `outline(source)` | `Str` | Python source as its outline: every function's signature, first docstring line and the names it calls. |

### When to use which

| Situation | Use |
|---|---|
| records, logs, repeated lines, a document resent after edits | `pack`, `dedupe`, `diff` (exact, always safe) |
| a long file or manual and one question about it | `focus` |
| a long conversation | `fit_chat` |
| the model should see the structure of a whole project | `outline` of each file, then `focus` on the file it asks for |

`focus` is optional: if you never call it, nothing changes. When the model needs a part that was left out, it can ask
for it by number (the map line tells it so) and you send `unfold(f, [n])`. Measured on Apsi's own sources: a
15,137-token module focused on one question to 1,333 tokens (91 % fewer) with the relevant functions whole;
`docs/GPU.md` (10,633 tokens) asked how to lower the card's temperature: 550 tokens, the right troubleshooting entry
whole.

## Limits

* Token counts without `tiktoken` are estimates; the savings percentage is reliable to a few points because both
  sides are estimated the same way.
* Packing changes nothing for data with no repetition; `best()` then returns the smallest form, never a larger one.
* A model must be told the format in one sentence (examples above). Without that sentence it may misread `§N` or
  `=N` lines.
* `focus` chooses by words and structure, not by meaning: a question that uses none of the words of the part it
  needs (a synonym, another language) may not find it. Name the function or the topic, or give `keep=`.
