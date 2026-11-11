class Ctx:
    def __enter__(self):
        print("enter"); return self
    def __exit__(self, t, v, tb):
        print("exit", t.__name__ if t else None); return True
with Ctx() as c:
    raise ValueError("x")
print("after")
from contextlib import contextmanager
@contextmanager
def tag(n):
    print("<" + n + ">"); yield n; print("</" + n + ">")
with tag("b") as t:
    print("in", t)
import tempfile, os
p = os.path.join(tempfile.gettempdir(), "apsi_conv_t.txt")
with open(p, "w") as f:
    f.write("a\nb\n")
with open(p) as f:
    print([l.rstrip() for l in f])
os.remove(p)
