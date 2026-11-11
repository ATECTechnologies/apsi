import json, copy, sys
d = {"a": [1, 2, {"b": None}], "c": True, "d": 1.5}
s = json.dumps(d, sort_keys=True)
print(s, json.loads(s) == d, json.dumps({"x": "é"}), json.dumps([1], indent=2))
e = copy.deepcopy(d); e["a"][2]["b"] = 1; print(d["a"][2], e["a"][2])
print(type(None).__name__, type([]).__name__, callable(print), sys.version_info[0] >= 3)
import string
print(string.ascii_lowercase[:5], str(None), str([1, "a"]), repr("a'b"), repr('a"b'))
xs = [3, 1, 2]; xs.sort(); xs.extend([9]); xs.insert(0, 0); xs.remove(9); print(xs, xs.index(2), xs[-1])
t = (1,); print(t, type(t).__name__, (1, 2) + (3,), [1, 2] * 2)
print(dict(a=1) | {"b": 2}, {1, 2} & {2, 3}, {1, 2} ^ {2, 3}, frozenset([1]))
