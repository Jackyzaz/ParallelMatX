import parallelmatx
import numpy as np

if __name__ == "__main__":
    size = 100
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    result = parallelmatx.cross_product(A, B)
    print(np.matmul(A, B))
    print("result:\n", result)
