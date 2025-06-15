import numpy as np

# np.where
arr = np.array([1, 4, 5, 2, 3, 6])
res = np.where(arr > 3)
print(res)

res = np.where(arr > 3, 1, 0)
print(res)
