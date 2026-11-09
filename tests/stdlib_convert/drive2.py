import _pydecimal as D, _pydatetime as dt, argparse, configparser, fnmatch, base64, string, tomllib, hparser, uparse
D.getcontext().prec = 50
print(D.Decimal(1) / D.Decimal(7), D.Decimal(2).sqrt(), D.Decimal("1.1") + D.Decimal("2.2"), D.Decimal(10) ** -30)
print(D.Decimal("3.14159").quantize(D.Decimal("0.01")), D.Decimal(1).exp(), D.Decimal(100).ln(), D.Decimal(-0.0))
d = dt.datetime(2026, 11, 10, 3, 4, 5, 678) + dt.timedelta(days=40, hours=30)
print(d, d.isoformat(), d.strftime("%Y/%m/%d %H:%M"), d.weekday(), dt.date(2024, 2, 29).toordinal(), dt.timedelta(seconds=-1))
print(dt.datetime.fromisoformat("2026-01-02T03:04:05+07:00"), dt.timezone(dt.timedelta(hours=7)))
p = argparse.ArgumentParser(prog="tool", description="demo")
p.add_argument("files", nargs="+"); p.add_argument("-n", type=int, default=3); p.add_argument("--flag", action="store_true")
print(p.parse_args(["a", "b", "-n", "7", "--flag"]))
print(p.format_help())
c = configparser.ConfigParser(); c.read_string("[s]\na = 1\nb = ${a}x\n[t]\nk: v")
print(c.sections(), c["s"]["a"], c.getint("s", "a"), dict(c["t"]))
print(fnmatch.fnmatch("data.csv", "*.c?v"), fnmatch.filter(["a.py", "b.txt", "c.py"], "*.py"), fnmatch.translate("[!x]*.py"))
print(base64.b64encode(b"hello world"), base64.b32decode(base64.b32encode(b"xyz")), base64.a85encode(b"abc"), base64.urlsafe_b64encode(b"\xff\xfe"))
print(string.Template("$who likes ${what}").substitute(who="tim", what="pie"), string.capwords("hello big  world"), string.Formatter().format("{0}-{x}", 1, x=2))
print(tomllib.loads('title = "T"\n[owner]\nname = "x"\ndob = 1979-05-27T07:32:00-08:00\n[[arr]]\nv = [1, 2.5, true]\n'))
class H(hparser.HTMLParser):
    def handle_starttag(self, tag, attrs): print("start", tag, attrs)
    def handle_data(self, data):
        if data.strip(): print("data", data.strip())
H().feed('<div class="a"><p id=x>Hi &amp; bye</p><br/></div>')
u = uparse.urlparse("https://user:pw@example.com:8080/p/a?q=1&r=two#frag")
print(u, u.port, uparse.parse_qs(u.query), uparse.quote("a b/ç"), uparse.urljoin("http://a/b/c", "../d"), uparse.urlencode({"k": "v w", "n": 1}))
