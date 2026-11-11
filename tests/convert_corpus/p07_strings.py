s = "  Hello, World  "
print(s.strip().split(", "), "-".join(["a", "b"]), s.find("W"), s.count("l"), s.replace("l", "L", 1))
print("abc".startswith("a"), "x".zfill(3), "ab".center(6, "*"), "A1".isalnum(), "3".isdigit())
print(ord("A"), chr(66), "ß".upper(), "a,b,,c".split(","), "line1\nline2".splitlines())
import re
print(re.findall(r"\d+", "a1b22c333"), re.sub(r"(\w)(\d)", r"\2\1", "a1 b2"))
m = re.match(r"(?P<w>\w+)@(\w+)", "me@host")
print(m.group("w"), m.groups())
b = "héllo".encode("utf-8"); print(b, b.decode(), len(b))
print(f"{1234567.891:,.2f} {255:#x} {255:08b} {0.25:%} {'x':^5}|")
