# Databases (`use db`)

PostgreSQL, Redis and SQLite clients built into Apsi's runtime — nothing to install, no driver to compile.

```apsi
use db
let pg = db.postgres("postgres://shop:secret@localhost:5432/shop")   # needs --allow-net=localhost:5432
pg.execute("create table if not exists item (id serial primary key, name text, price numeric)")
pg.query("insert into item (name, price) values ($1, $2)", "pen", 1.25)
for row in pg.query("select id, name, price from item where price < $1", 10):
    print(row.id, row.name, row.price)               # a record; row["name"] works too
with pg.transaction():                                # commit, or roll back on an error
    pg.query("update item set price = price * 1.1")
```

## PostgreSQL

* **Protocol:** the v3 wire protocol; SCRAM-SHA-256, md5 and password authentication; TLS with `sslmode=require`
  (or `verify-full`) in the URL.
* **Parameters are never pasted into SQL.** `$1, $2 …` travel separately (the extended query protocol), so values
  cannot inject SQL. The server's parameter types are used, so a list goes into an `int[]` or `text[]` column.
* **Prepared statements** are reused per connection (256 per connection, least recently used dropped).
* **Results** become Apsi values: int, float, `Decimal` for numeric, bool, text, bytes, json/jsonb (parsed),
  dates and timestamps, uuid, one-dimensional arrays.

| | |
|---|---|
| `pg.query(sql, *params)` | a `Result`: iterate rows; `.one()`, `.scalar()`, `.dicts()`, `.column(name)`, `.to_dataframe()` |
| `pg.execute(sql)` | statements without results (several separated by `;`) |
| `pg.insert_many(table, rows, columns=)` | bulk insert in batches |
| `with pg.transaction():` | `begin` … `commit`, or `rollback` when the block raises |
| `pg.listen(channel)`, `pg.notify(channel, text)`, `pg.poll(timeout)` | LISTEN / NOTIFY |
| `db.pool(url, size=8)` | a thread-safe pool: `with pool.connection() as c:` |



## Redis

```apsi
let r = db.redis("redis://:secret@localhost:6379/0")
r.set("hits", 0)
r.incr("hits")
r.hset("user:1", "name", "ann")
print(r.hgetall("user:1"))             # {"name": "ann"}
let p = r.pipeline()                   # many commands, one round trip
for i in range(10000):
    p.rpush("queue", i)
p.execute()
let sub = r.subscribe("news")          # pub/sub
```

Every command works by name (`r.zadd(…)`, `r.xadd(…)`) or as `r.command("XADD", …)`. `HGETALL` and similar replies
come back as dicts.

## SQLite

```apsi
let lite = db.sqlite("shop.db")         # needs --allow-read / --allow-write for the file; ":memory:" needs nothing
lite.execute("create table t (a, b)")
lite.insert_many("t", [[1, "x"], [2, "y"]])
print(lite.query("select * from t where a > ?", 1).dicts())
```

## Measured

Local PostgreSQL 16 and Redis 7 on a 2-core VM (`benchmarks/db/run.apsi`, best of three runs):

| operation | time |
|---|---|
| PostgreSQL: insert 20,000 rows with `insert_many` | 191 ms |
| PostgreSQL: 2,000 point queries by primary key (prepared, one round trip each) | 168 ms (84 µs each) |
| PostgreSQL: read 20,000 rows into records | 46 ms |
| Redis: 10,000 pipelined `RPUSH` | 29 ms |
| Redis: 2,000 `SET` round trips | 97 ms (49 µs each) |

## Permissions

Network databases need `--allow-net=HOST:PORT`. Passwords in URLs are not printed in errors or in `print(pg)`.
Use `use env` to keep them out of the source: `db.postgres(env.secret("DATABASE_URL"))`.

## MySQL and MongoDB

Not built in yet. From a converted or `--allow-python` program, `pyimport("pymysql")` and `pyimport("pymongo")` work
with the real Python drivers.
