class Animal:
    count = 0
    def __init__(self, name):
        self.name = name
        Animal.count += 1
    def speak(self):
        return f"{self.name} makes a sound"
    def __repr__(self):
        return f"Animal({self.name!r})"
class Dog(Animal):
    def speak(self):
        return super().speak() + " (woof)"
    @property
    def upper(self):
        return self.name.upper()
    @staticmethod
    def kind():
        return "dog"
    @classmethod
    def make(cls, n):
        return cls(n)
d = Dog.make("rex")
print(d.speak(), d.upper, Dog.kind(), Animal.count, repr(d), [d])
print(isinstance(d, Animal), type(d).__name__)
class V:
    def __init__(self, x, y): self.x, self.y = x, y
    def __add__(self, o): return V(self.x + o.x, self.y + o.y)
    def __eq__(self, o): return self.x == o.x and self.y == o.y
    def __lt__(self, o): return (self.x, self.y) < (o.x, o.y)
    def __len__(self): return 2
    def __getitem__(self, i): return (self.x, self.y)[i]
    def __str__(self): return f"<{self.x},{self.y}>"
    def __hash__(self): return hash((self.x, self.y))
v = V(1, 2) + V(3, 4)
print(v, v == V(4, 6), len(v), v[1], sorted([V(2, 1), V(1, 5)])[0], {V(1,1), V(1,1)}.__len__())
