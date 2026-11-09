import numpy as np

a = np.arange(12).reshape(3, 4)
b = np.ones((4, 2))
print(a.shape, (a @ b).tolist())
x = np.linspace(0, 1, 5)
print(np.round(np.sin(x), 4).tolist())
m = np.array([[2.0, 1.0], [1.0, 3.0]])
print(np.round(np.linalg.inv(m), 4).tolist(), round(float(np.linalg.det(m)), 4))
print(float(a.sum()), float(a.mean()), a.max(), a[:, 1].tolist(), a[1].tolist())
v = np.zeros(5)
v[2] = 7.0
v += 1
print(v.tolist(), float(np.dot(v, v)))
print(np.where(np.array([1, -2, 3]) > 0, 1, 0).tolist())
