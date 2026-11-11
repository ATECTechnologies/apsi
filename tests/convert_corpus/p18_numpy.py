import numpy as np
a = np.arange(6).reshape(2, 3)
print(a.sum(axis=0), a.T.shape, (a * 2).tolist(), a.mean(), np.dot(a, a.T).tolist())
b = np.linspace(0, 1, 5)
print(np.round(b, 2).tolist(), b[b > 0.4].tolist(), np.argmax([1, 5, 2]), np.zeros(2).dtype)
print(np.linalg.norm([3, 4]), np.cumsum([1, 2, 3]).tolist(), np.where(a > 2, 1, 0).tolist())
