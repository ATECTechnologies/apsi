import dataclasses as dc, enum, copy, operator, contextlib, _py_abc, pickle, reprlib, json, random, numbers
@dc.dataclass(order=True, frozen=True)
class P:
    x: int
    y: list = dc.field(default_factory=list, compare=False)
p = P(1, [2]); print(p, dc.asdict(p), dc.replace(p, x=5), P(1) < P(2), dc.fields(P)[1].name, hash(P(3)) == hash(P(3)))
try: p.x = 3
except dc.FrozenInstanceError as e: print("frozen", type(e).__name__)
class C(enum.IntFlag):
    R = 1; W = 2; X = 4
class S(enum.StrEnum):
    A = enum.auto(); B = enum.auto()
print(C.R | C.W, list(C), C(6), S.A, S("b"), S.A.upper(), [m.name for m in S])
print(copy.deepcopy({"a": [1, {2}]}), operator.itemgetter(1, 0)("ab"), operator.add(2, 3))
with contextlib.ExitStack() as st:
    st.callback(print, "cleanup")
    print("body")
class Ab(metaclass=_py_abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, o): return hasattr(o, "quack") or NotImplemented
class Duck:
    def quack(self): pass
print(issubclass(Duck, Ab), isinstance(1, Ab))
data = pickle.dumps({"k": [1, 2.5, None, ("t",)], "b": b"x"}, protocol=4)
print(len(data), pickle.loads(data), pickle.loads(pickle._dumps([1, 2], protocol=0)))
print(reprlib.repr(list(range(100))), reprlib.repr("x" * 100))
print(json.dumps({"a": [1, 2.5, {"b": None}], "u": "é\n"}, indent=1, ensure_ascii=True), json.loads('[1e3, "\\u00e9", {"k": false}]'))
print(json.encoder.py_encode_basestring_ascii("é\"x"), json.decoder.py_scanstring('ab\\n"', 0))
random.seed(42); print(random.random(), random.randint(1, 100), random.choice("abc"), random.sample(range(10), 3), random.gauss(0, 1))
print(isinstance(3, numbers.Integral), isinstance(2.5, numbers.Rational))
