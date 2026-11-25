import re
text = "Contact: ann@example.com, Bob <bob.smith@mail.org>; tel 555-1234 or (555) 987-6543."
print(re.findall(r"[\w.]+@[\w.]+\.\w+", text))
print(re.findall(r"(\d{3})-(\d{4})", text), re.findall(r"\((\d+)\)", text))
m = re.search(r"(?P<user>\w+)@(?P<host>[\w.]+)", text)
print(m.group(), m.group(1), m.group("host"), m.groups(), m.groupdict(), m.span(), m.start(2), m)
print(re.sub(r"\d", "#", "a1b22c333"), re.sub(r"(\w+)@(\w+)", r"\2 at \1", "x@y and p@q"), re.subn("a", "b", "banana"))
print(re.split(r"[,;]\s*", "a, b;c,  d"), re.split(r"(\d)", "a1b2c"), re.split(r"x*", "axbc"), re.split(",", "a,b,c", maxsplit=1))
print(re.match(r"\d+", "123abc").group(), re.match(r"\d+", "abc"), bool(re.fullmatch(r"[a-z]+", "hello")), re.fullmatch(r"a|ab", "ab").group())
print(re.findall(r"a*|b", "b"), re.findall(r"\b\w", "hello big world"), re.findall(r"^\w+", "one\ntwo\nthree", re.M), re.findall(r"o$", "foo\nboo", re.M))
print(re.findall(r"(?i)hello", "Hello HELLO hello"), re.findall("A.B", "a\nb A\nB", re.S | re.I), re.findall(r"(\w)\1", "aabbcd ee"))
print(re.findall(r"\d+(?=px)", "10px 20em 30px"), re.findall(r"(?<!\$)\b\d+", "$5 and 7 and $9 10"), re.findall(r"(?<=@)\w+", "a@b c@d"))
print(re.sub(r"(?P<w>\w+)", lambda m: m.group("w").upper(), "go now"), re.escape("a.b*c?"), re.compile(r"\s+").split("a  b\tc"))
p = re.compile(r"""
   (\d+)   # number
   \s*     # space
   ([a-z]+)  # unit
""", re.X)
print(p.findall("10 kg 20m 3  cm"), p.pattern.count("\n"), re.findall(r"[^aeiou\s]+", "quick brown"), re.findall(r"x{2,3}?", "xxxxx"))
print(re.findall(r"\w+", "héllo wörld"), re.sub(r"\s+$", "", "trail   "), re.search(r"(a)|(b)", "b").groups(), re.findall(r"[\d.]+", "v1.2.3 and 4"))
print([m.span() for m in re.finditer(r"\w+", "ab cd  e")], re.sub(r"", "-", "abc"), re.findall(r"(?:ab)+", "ababab ab"))
print(re.findall(r"\w+", "สวัสดี ครับ ภาษาไทย 123"), re.sub(r"\s+", " ", "ก  ข\tค"), re.findall(r"[ก-ฮ]", "กขab"))
