from dataclasses import dataclass

class Animal:
    def __init__(self, name, sound="..."):
        self.name = name
        self.sound = sound

    def speak(self):
        return f"{self.name} says {self.sound}"

class Dog(Animal):
    def __init__(self, name):
        super().__init__(name, "woof")
        self.tricks = []

    def add_trick(self, t):
        self.tricks.append(t)
        return self

    def __str__(self):
        return f"Dog({self.name}, {len(self.tricks)} tricks)"

@dataclass
class Point:
    x: float
    y: float = 0.0

    def norm(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

class Counter:
    count = 0
    def __init__(self):
        Counter.count += 1

d = Dog("Rex").add_trick("sit").add_trick("roll")
print(d.speak())
print(str(d))
print(Animal("Cat", "meow").speak())
p = Point(3.0, 4.0)
print(p.norm(), p.x, Point(1.0).y)
for _ in range(3):
    Counter()
print(Counter.count)
print(isinstance(d, Dog), isinstance(d, Animal))
