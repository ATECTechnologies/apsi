# Databases (`use db`)

PostgreSQL, MySQL / MariaDB, MongoDB, Redis and SQLite clients built into Apsi's runtime — nothing to install, no
driver to compile.

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

## MySQL and MariaDB

```apsi
let my = db.mysql("mysql://shop:secret@localhost:3306/shop")          # needs --allow-net=localhost:3306
my.execute("create table if not exists item (id int auto_increment primary key, name text, price decimal(9,2))")
let r = my.query("insert into item (name, price) values (?, ?)", "pen", 1.25)
print(r.rowcount, r.insert_id)
for row in my.query("select id, name, price from item where price < ?", 10):
    print(row.id, row.name, row.price)                 # price is a Decimal
with my.transaction():
    my.insert_many("item", [["ink", 3.5], ["pad", 2.0]], columns=["name", "price"])
```

* **Authentication:** `caching_sha2_password` (the MySQL 8 default: fast path; full path over TLS or with the
  server's RSA key) and `mysql_native_password` (MariaDB, MySQL 5.7), including authentication switches.
* **TLS:** `?ssl=true` or `?ssl-mode=REQUIRED` (`VERIFY_CA`, `VERIFY_IDENTITY` check the certificate).
* **Values** are escaped by the client on a utf8mb4 connection, so a value can never end a string literal.
  Placeholders: `?`, `%s` or `$1`. Lists become `(1, 2, 3)` for `IN ?`. Results: int, float, `Decimal`, str, bytes
  (binary columns), dates, times, JSON.
* `db.mariadb(…)` is the same client. `db.pool("mysql://…", size=8)` shares connections between threads.
* Several statements in one `execute` run in order; the result of the last one is returned.

## MongoDB

```apsi
let mongo = db.mongo("mongodb://app:secret@localhost:27017/shop?authSource=admin")   # --allow-net=localhost:27017
let items = mongo.collection("items")                  # or mongo["shop"]["items"]
let id = items.insert_one({"name": "pen", "price": 1.25, "tags": ["office"]})
items.insert_many([{"name": f"n{i}", "price": i} for i in range(1000)])
for doc in items.find({"price": {"$lt": 10}}, sort={"price": 1}, limit=20):
    print(doc["name"], doc["price"])
items.update_one({"name": "pen"}, {"$set": {"price": 1.5}})
print(items.count_documents({}), items.distinct("tags"))
print(items.aggregate([{"$group": {"_id": none, "total": {"$sum": "$price"}}}]))
items.create_index({"name": 1}, unique=true)
```

* **Protocol:** OP_MSG (MongoDB 3.6 and newer); cursors are followed with `getMore` automatically.
* **Authentication:** SCRAM-SHA-256 and SCRAM-SHA-1 (`authSource`, `authMechanism`); the server's signature is
  verified too. TLS with `?tls=true`. `mongodb+srv://` addresses need the host written out.
* **Values:** dicts keep their key order; int (32/64-bit), float, `Decimal` (decimal128), str, bytes, bool, none,
  datetime (UTC), `ObjectId`, regular expressions, UUIDs.
* Collections: `insert_one`, `insert_many`, `find`, `find_one`, `update_one`, `update_many`, `replace_one`,
  `delete_one`, `delete_many`, `count_documents`, `aggregate`, `distinct`, `create_index`, `drop`; any other
  command with `mongo.command({...})`.

## How the clients are tested

PostgreSQL 16 and Redis 7 are tested against real servers. The MySQL and MongoDB clients are tested against
in-process servers that speak the real wire protocols (handshakes, both MySQL authentication methods and an
authentication switch, SCRAM-SHA-256 checked on the server side, cursors with several batches, errors), and SCRAM
against the RFC 7677 test vector. `APSI_TEST_MYSQL=mysql://…` and `APSI_TEST_MONGO=mongodb://…` run the same tests
against your own servers.
