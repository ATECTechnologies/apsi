"""use config: layered settings files, environment and command-line overrides, types checked."""
import os
import tempfile
import unittest
from unittest import mock

from apsi import config


class Config(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def w(self, name, text):
        p = os.path.join(self.d, name)
        with open(p, "w") as fh:
            fh.write(text)
        return p

    def test_layers_env_args_and_sources(self):
        a = self.w("a.toml", 'name = "x"\n[server]\nport = 1\nhost = "h"\n[t]\nout = "${name}-${server.port}"\n')
        b = self.w("b.json", '{"server": {"port": 2}}')
        c = self.w("c.ini", "[server]\nhost = k\n")
        e = self.w(".env", "SERVER__DEBUG=true\n")
        with mock.patch.dict(os.environ, {"APP_SERVER__PORT": "3", "APP_T__LIST": "[1, 2]"}):
            cfg = config.load([a, b, c, e, a + "x?"], env="APP", args=True, argv=["--set", "t.n=5", "z.w=0.5"],
                              defaults={"t": {"epochs": 10}})
        self.assertEqual(cfg.server.port, 3)
        self.assertEqual(cfg.server.host, "k")
        self.assertIs(cfg.server.debug, True)
        self.assertEqual(cfg.t.list, [1, 2])
        self.assertEqual((cfg.t.n, cfg.z.w, cfg.t.epochs), (5, 0.5, 10))
        self.assertEqual(cfg.t.out, "x-3")
        src = config._WHERE[id(cfg)][1]
        self.assertEqual(src["server.port"], "environment")
        self.assertEqual(src["server.host"], "c.ini")
        self.assertEqual(src["t.epochs"], "default")

    def test_errors_name_the_file_and_key(self):
        a = self.w("a.toml", "[server]\nport = \"eighty\"\n")
        with self.assertRaisesRegex(config.ConfigError, r"a.toml: 'server.port' should be a whole number"):
            config.load([a], defaults={"server": {"port": 80}})
        with self.assertRaisesRegex(config.ConfigError, "settings missing: db.url"):
            config.load([a], required=["db.url"])
        with self.assertRaisesRegex(config.ConfigError, "does not exist"):
            config.load([os.path.join(self.d, "none.toml")])

    def test_save_roundtrip(self):
        cfg = config.load([self.w("a.json", '{"a": 1, "b": {"c": [1, 2], "d": "x y", "e": true}, "f": 0.5}')])
        for ext in ("toml", "json"):
            p = config.save(os.path.join(self.d, "out." + ext), cfg)
            back = config.load([p])
            self.assertEqual(config._plain(back), config._plain(cfg))


if __name__ == "__main__":
    unittest.main()
