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

## Limits

* Token counts without `tiktoken` are estimates; the savings percentage is reliable to a few points because both
  sides are estimated the same way.
* Packing changes nothing for data with no repetition; `best()` then returns the smallest form, never a larger one.
* A model must be told the format in one sentence (examples above). Without that sentence it may misread `§N` or
  `=N` lines.
