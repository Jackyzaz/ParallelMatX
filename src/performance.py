import time
import numpy as np
import matplotlib.pyplot as plt
from normal import normal_matrix_multiplication
from parallel import parallel_matrix_multiplication


def measure_time(func, A, B):
    start = time.time()
    func(A, B)
    end = time.time()
    return end - start


def test_performance_between_normal_and_parallel():
    START_SIZE = 100
    MAX_SIZE = 400
    DISTANCE = 100

    matrix_sizes = [x for x in range(START_SIZE, MAX_SIZE, DISTANCE)]

    # Store results
    normal_times = []
    parallel_times = []

    for size in matrix_sizes:
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)

        # Measure times
        normal_times.append(
            measure_time(normal_matrix_multiplication, A.tolist(), B.tolist())
        )
        parallel_times.append(
            measure_time(parallel_matrix_multiplication, A.tolist(), B.tolist())
        )

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(matrix_sizes, normal_times, marker="o", linestyle="-", label="Normal")
    plt.plot(
        matrix_sizes,
        parallel_times,
        marker="s",
        linestyle="-",
        label="Parallel (Concurrency)",
    )

    plt.xlabel("Matrix Size (NxN)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Performance Comparison of Matrix Multiplication Methods")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    test_performance_between_normal_and_parallel()
