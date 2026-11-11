class A:
    count = 0
    def __init__(self):
        A.count += 1
    @staticmethod
    def kind():
        return "k"
A(); A()
print(A.count, A.kind())
