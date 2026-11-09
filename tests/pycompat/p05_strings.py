import json
text = """line one
line two
line three"""
lines = text.splitlines()
print(len(lines), lines[-1].title(), text.count("line"))
print("  pad  ".strip(), "a,b,,c".split(","), "abc".startswith("ab"), "abc".endswith("z"))
print("x".join(["1", "2"]), "abc".find("c"), "Hello".center(11, "*"), "7".zfill(3))
data = {"name": "apsi", "ver": [2, 0, 0], "ok": True}
s = json.dumps(data, sort_keys=True)
print(s)
back = json.loads(s)
print(back["ver"][0], back.get("missing", "default"), list(back.keys()))
print("{} + {} = {}".format(1, 2, 3), "%d%%" % 50, f"{3.14159:.2f}|{42:>5}|{'x':<3}|")
word = "mississippi"
freq = {}
for ch in word:
    freq[ch] = freq.get(ch, 0) + 1
print(sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])))
print(word.upper().isupper(), "123".isdigit(), "ab".isalpha(), ord("a") - ord("A"))
