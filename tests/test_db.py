"""use db against real servers (skipped when none is reachable).

APSI_TEST_PG=postgres://user:pw@host:port/db   APSI_TEST_REDIS=redis://:pw@host:port/0"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apsi import db  # noqa: E402

PG = os.environ.get("APSI_TEST_PG", "postgres://apsi:secretpw@127.0.0.1:55432/postgres")
RD = os.environ.get("APSI_TEST_REDIS", "redis://:redispw@127.0.0.1:56379/0")


def _try(f):
    try:
        f().close()
        return True
    except Exception:          # noqa: BLE001
        return False


HAVE_PG = _try(lambda: db.Postgres(PG))
HAVE_REDIS = _try(lambda: db.Redis(RD))


@unittest.skipUnless(HAVE_PG, "no PostgreSQL server")
class PostgresTest(unittest.TestCase):
    def setUp(self):
        self.c = db.Postgres(PG)
        self.c.execute("drop table if exists t_apsi; create table t_apsi (id serial primary key, name text, "
                       "v numeric, tags text[], meta jsonb, ok bool, raw bytea)")

    def tearDown(self):
        self.c.execute("drop table if exists t_apsi")
        self.c.close()

    def test_roundtrip_types_and_injection(self):
        evil = "x'); drop table t_apsi; --"
        self.c.query("insert into t_apsi (name, v, tags, meta, ok, raw) values ($1, $2, $3, $4, $5, $6)",
                     evil, 1.5, ["a", 'q"x', None], {"k": [1, 2]}, True, b"\x00\xff")
        r = self.c.query("select name, v, tags, meta, ok, raw from t_apsi").one()
        self.assertEqual(r.name, evil)
        self.assertEqual(str(r.v), "1.5")
        self.assertEqual(r.tags, ["a", 'q"x', None])
        self.assertEqual(r.meta, {"k": [1, 2]})
        self.assertIs(r.ok, True)
        self.assertEqual(r.raw, b"\x00\xff")

    def test_errors_transactions_bulk(self):
        with self.assertRaises(db.DatabaseError) as cm:
            self.c.query("select * from missing_table")
        self.assertEqual(cm.exception.code, "42P01")
        self.assertEqual(self.c.query("select 1").scalar(), 1)       # the connection still works
        with self.assertRaises(ValueError):
            with self.c.transaction():
                self.c.query("insert into t_apsi (name) values ($1)", "gone")
                raise ValueError
        self.assertEqual(self.c.query("select count(*) from t_apsi").scalar(), 0)
        n = self.c.insert_many("t_apsi", [[f"n{i}", i] for i in range(2500)], ["name", "v"])
        self.assertEqual(n, 2500)
        self.assertEqual(self.c.query("select count(*) from t_apsi where v >= $1", 2000).scalar(), 500)

    def test_pool_threads(self):
        import threading
        pool = db.Pool(lambda: db.Postgres(PG), 4)
        out = []
        ts = [threading.Thread(target=lambda i=i: out.append(pool.query("select $1::int * 2", i).scalar()))
              for i in range(20)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(sorted(out), [i * 2 for i in range(20)])
        self.assertLessEqual(pool.made, 4)


@unittest.skipUnless(HAVE_REDIS, "no Redis server")
class RedisTest(unittest.TestCase):
    def test_commands_pipeline(self):
        r = db.Redis(RD)
        r.delete("apsi:t", "apsi:h")
        self.assertEqual(r.incr("apsi:t"), 1)
        r.hset("apsi:h", "a", 1, "b", "x")
        self.assertEqual(r.hgetall("apsi:h"), {"a": "1", "b": "x"})
        p = r.pipeline()
        for i in range(100):
            p.incr("apsi:t")
        self.assertEqual(p.execute()[-1], 101)
        with self.assertRaises(db.RedisError):
            r.command("NOPE")
        self.assertEqual(r.get("apsi:t"), "101")
        r.delete("apsi:t", "apsi:h")


class SQLiteTest(unittest.TestCase):
    def test_memory(self):
        s = db.SQLite()
        s.execute("create table t (a int, b text)")
        s.insert_many("t", [[i, str(i)] for i in range(10)])
        self.assertEqual(s.query("select sum(a) as total from t where a > $1", 4).one().total, 35)


if __name__ == "__main__":
    unittest.main()
