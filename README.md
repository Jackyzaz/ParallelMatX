# ParallelMatX

This project is developed as part of the **240-123 Data Structure Algorithm and Programming Module** in my concurrency assignment

## Overview

ParallelMatX is an open-source Python library designed for parallel matrix multiplication.

It utilizes parallel processing techniques to optimize performance, competible with large-scale matrix computation

## Getting Started

### Installation

To install ParallelMatX, use pip:

```
pip install parallematx
```

### Usage Example

Here's a basic example demonstrating how to use ParallelMatX for parallel matrix multiplication:

```python
import parallelmatx
import numpy as np

if __name__ == "__main__": # Need main to run parallel
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    result = parallelmatx.parallel_matrix_multiplication(A, B)
    print("Result:\n", result)

    # result:
    # [[ 30.  36.  42.]
    # [ 66.  81.  96.]
    # [102. 126. 150.]]
```

## How It Works
### Understanding how matrix multiplication
Matrix multiplication is a binary operation that produces a matrix from two matrices. For matrix multiplication, the number of columns in the first matrix must be equal to the number of rows in the second matrix. The resulting matrix, known as the matrix product (ref: https://en.wikipedia.org/wiki/Matrix_multiplication)

![image](https://github.com/user-attachments/assets/41defda7-09d3-4742-8a83-98914758ff18)

### Problem of calculation
**Iterative algorithm** is most basic multiplication algorithmn. Its easy to understand how each row operation. 
The problem is its slowness, In computer science we can analyze the time complexity for time and space. This is analysis of this approch (assume we have matrix n x n)

Time Complexity: O(n<sup>3</sup>)

Space Complexity: O(n<sup>2</sup>)

![image](https://github.com/user-attachments/assets/0361c4e9-b355-463c-a3d9-a1b742876bd6)

Its slow because we have to do multiplication for each element. You can see from this picture how we traverse along matrix

![Row_and_column_major_order svg](https://github.com/user-attachments/assets/34141298-1f81-4ffd-bcec-43489bc2a779)

### Optimization
We can see that each role have independent result, So we can do parallel calculation for each row. Then we combine together 

![image](https://github.com/user-attachments/assets/755bf4c0-e8a5-4e7b-be77-105694d8080d)

After we analysis new complexity we got this

**Time Complexity**
- Worst Case (No Parallelism, max_workers = 1): O(n<sup>3</sup>), equivalent to traditional matrix multiplication.
- Average Case (When max_workers is moderate ) : O(n<sup>3</sup>/logn) ≈ O(n<sup>2.5</sup>)
- Best Case (Full Parallelism, max_workers = n): O(n<sup>2</sup>), where row computations are fully distributed.

**Space Complexity**: O(n<sup>2</sup>), as the final result matrix requires n<sup>2</sup> storage.

### Implementation
Using ProcessPoolExecutor from concurrnt.future library from python
```py
def compute_row(A: np.ndarray, B: np.ndarray, row_index: int) -> np.ndarray:
    return np.array([np.dot(A[row_index], B[:, col]) for col in range(B.shape[1])])


def parallel_matrix_multiplication(
    A: list | np.ndarray, B: list | np.ndarray, max_workers: int | None = None
) -> np.ndarray:
    # Format input array
    A = np.array(A)
    B = np.array(B)

    # Check matrix compatibility
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrix dimensions are incompatible for multiplication.")

    # Initial Result Matrix
    result = np.zeros((A.shape[0], B.shape[1]))

    # Run Process Pool
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Init parallel process store
        parallel_row_results = []

        # Start parallel compute
        for i in range(A.shape[0]):
            parallel_row_results.append(executor.submit(compute_row, A, B, i))

        # Retrieving all row results
        for i, row in enumerate(parallel_row_results):
            result[i] = row.result()

    return result
```

## Benchmarking
### Testing traditional approch vs parallel approch

