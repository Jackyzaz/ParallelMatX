import unittest
import numpy as np
from parallelmatx import parallel_matrix_multiplication


class TestMatrixMultiplication(unittest.TestCase):
    def test_square_matrices(self):
        """
        Test matrix multiplication with square matrices
        """
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        numpy_result = np.dot(A, B)
        parallel_result = parallel_matrix_multiplication(A, B)

        np.testing.assert_array_equal(
            parallel_result,
            numpy_result,
            "Parallel matrix multiplication does not match NumPy multiplication for square matrices",
        )

    def test_rectangular_matrices(self):
        """
        Test matrix multiplication with rectangular matrices
        """
        A = np.array([[1, 2], [3, 4], [5, 6]])
        B = np.array([[1, 2, 3], [4, 5, 6]])

        numpy_result = np.dot(A, B)
        parallel_result = parallel_matrix_multiplication(A, B)

        np.testing.assert_array_equal(
            parallel_result,
            numpy_result,
            "Parallel matrix multiplication does not match NumPy multiplication for rectangular matrices",
        )

    def test_incompatible_matrices(self):
        """
        Test that incompatible matrix dimensions raise a ValueError
        """
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        with self.assertRaises(
            ValueError, msg="Did not raise ValueError for incompatible matrices"
        ):
            parallel_matrix_multiplication(A, B)


if __name__ == "__main__":
    unittest.main()
