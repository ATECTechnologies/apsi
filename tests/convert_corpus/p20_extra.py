class D:
    @classmethod
    def make(cls, n):
        return cls()
    @property
    def up(self):
        return "UP"
print(D.make(1).up)
