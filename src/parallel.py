import numpy as np
from concurrent.futures import ProcessPoolExecutor


def compute_row(A, B_T, row_index):
    return np.dot(A[row_index], B_T)


def parallel_matrix_multiplication(A, B):
    """
    Parallel Matrix Multiplication Using Process Pool Executor

    """
    A = np.array(A)
    B = np.array(B)

    # Transpose B for efficiency
    B_T = B.T

    # Initial Result Matrix
    result = np.zeros((A.shape[0], B.shape[1]))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_row, A, B_T, i) for i in range(A.shape[0])]
        for i, future in enumerate(futures):
            result[i] = future.result()

    return result
