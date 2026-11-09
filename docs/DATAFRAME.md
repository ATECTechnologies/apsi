# Tables (`use dataframe`)

A columnar table engine built into Apsi: typed columns, whole-column kernels, no per-cell objects. It reads CSV,
JSON and JSON Lines, filters, groups, joins, sorts and writes, with an API close to Polars.

```apsi
use dataframe as df

let sales = df.read_csv("sales.csv")                 # needs --allow-read for the file
let eu = sales.filter((df.col("amount") > 90) & (df.col("region") == "EU"))
let by = sales.group_by("customer").agg(total=df.col("amount").sum(), n=df.count(), avg=df.col("amount").mean())
let top = by.sort("total", descending=true).head(3)
let full = top.join(df.read_csv("customers.csv"), on="customer", how="left")
print(full)
full.write_csv("top.csv")                            # needs --allow-write
```

```text
shape: (3, 5)
┌──────────┬───────┬─────┬────────┬──────┐
│ customer │ total │   n │    avg │ tier │
│ str      │ float │ int │  float │  str │
├──────────┼───────┼─────┼────────┼──────┤
│ cid      │   300 │   1 │    300 │ gold │
│ ann      │ 219.5 │   2 │ 109.75 │ gold │
│ dan      │   150 │   1 │    150 │ null │
└──────────┴───────┴─────┴────────┴──────┘
```

## Building tables

| | |
|---|---|
| `df.read_csv(path, sep=",", columns=[…])` | typed columns (int, float, bool, date, str); a C parser is used when a C compiler is present |
| `df.read_jsonl(path)`, `df.read_json(path)` | JSON Lines / a JSON array of objects |
| `df.from_records([{…}, …])`, `df.frame({"a": [1, 2], "b": ["x", "y"]})` | from Apsi values |
| `df.from_csv_text(text)` | CSV already in memory |
| `df.concat(a, b, …)` | rows of several tables |

## Expressions

`df.col("name")` refers to a column; `df.lit(v)` is a constant; `df.count()` counts rows in a group. Expressions
combine with `+ - * / % ** == != < <= > >= & | ~` and are evaluated column-wise, without Python loops.

* numbers: `abs sqrt log exp round(n) cast("float")`
* aggregates: `sum mean min max std var median count first last n_unique`
* windows: `cumsum shift(k) rolling_mean(w)`
* nulls and sets: `is_null fill_null(v) is_in([…])`
* strings: `str_lower str_upper str_strip str_len str_contains str_startswith str_endswith str_replace(a, b)`
* names: `alias("new")`, or keyword arguments: `with_columns(double=df.col("x") * 2)`
* anything else: `map(fn)` (runs per value — slower)

## Table operations

`select`, `with_columns`, `drop`, `rename`, `filter`, `head`, `tail`, `sample(n, seed=)`, `sort(by, descending=)`,
`unique(subset=)`, `group_by(…).agg(…)`, `join(other, on= | left_on= right_on=, how="inner" | "left" | "outer",
suffix="_right")`, `value_counts(column)`, `describe()`, `concat`.

Reading back: `height`, `width`, `shape`, `columns`, `dtypes`, `row(i)`, `rows()`, `to_dicts()`, `to_dict()`,
`to_numpy()`, `to_pandas()`, `memory_bytes()`; writing: `write_csv`, `write_jsonl`, `write_json`, `to_json`.

## How it is fast

* **Columns are contiguous typed arrays** (int64, float64, bool, datetime64). A million numbers take 8 MB.
* **Strings are dictionary-encoded**: an int32 code per row into a list of unique values. Comparing, grouping and
  joining on strings compares integers.
* **Group-by** factorises the keys once and reduces with `bincount` / `reduceat`. **Joins** match the factorised keys
  by direct addressing. **Filters** are boolean masks.

Measured against pandas 2 on 2 million rows (2-core VM, best of 3, results checked equal):

| operation | pandas | Apsi | |
|---|---|---|---|
| filter with two conditions | 146 ms | 6.1 ms | **24× faster** |
| group by 50,000 string keys: sum, mean, count | 180 ms | 42 ms | **4.3× faster** |
| group by two keys: std, max | 187 ms | 44 ms | **4.2× faster** |
| left join 2M × 50k on a string key | 227 ms | 74 ms | **3.1× faster** |
| sort by a float column | 196 ms | 111 ms | **1.8× faster** |
| arithmetic on columns | 8.9 ms | 6.5 ms | 1.4× faster |
| read a 1M-row CSV | 0.36 s | 0.30 s | 1.2× faster |
| memory for the table | 262 MB | 65 MB | **4× less** |

Reproduce with `benchmarks/dataframe/run.py` and `benchmarks/dataframe/csv.py`.
