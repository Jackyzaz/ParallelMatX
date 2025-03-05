import parallelmatx
import numpy as np

if __name__ == "__main__":
    size = 100
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    result = parallelmatx.parallel_matrix_multiplication(A, B)
    print("result:\n", result)
