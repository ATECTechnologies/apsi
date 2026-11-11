class A:
    def who(self): return "A"
class B(A):
    def who(self): return "B" + super().who()
class C(A):
    def who(self): return "C" + super().who()
class D(B, C):
    def who(self): return "D" + super().who()
print(D().who(), [k.__name__ for k in D.__mro__])
class Base:
    def __init_subclass__(cls, **kw):
        super().__init_subclass__()
        cls.tag = cls.__name__.lower()
class X(Base): pass
print(X.tag)
class Meta(type):
    def __call__(cls, *a):
        o = super().__call__(*a); o.made = True; return o
class M(metaclass=Meta):
    def __init__(self, v): self.v = v
m = M(3); print(m.v, m.made)
class Slots:
    __slots__ = ("a",)
    def __init__(self): self.a = 1
s = Slots(); print(s.a, hasattr(s, "b"), getattr(s, "b", "no"))
