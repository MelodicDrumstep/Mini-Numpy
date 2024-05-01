"""
Feel free to add more test classes and/or tests that are not provided by the skeleton code!
Make sure you follow these naming conventions: https://docs.pytest.org/en/reorganize-docs/goodpractices.html#test-discovery
for your new tests/classes/python files or else they might be skipped.
"""
from utils import *
import numc as nc

"""
For each operation, you should write tests to test correctness on matrices of different sizes.
Hint: use rand_dp_nc_matrix to generate dumbpy and numc matrices with the same data and use
      cmp_dp_nc_matrix to compare the results
"""

class TestBasicCorrectness:
    def test_init(self):
        # This creates a 3 * 3 matrix with entries all zeros
        mat1 = nc.Matrix(3, 3)

        # This creates a 2 * 3 matrix with entries all ones
        mat2 = nc.Matrix(3, 3, 1)

        # This creates a 2 * 3 matrix with first row 1, 2, 3, second row 4, 5, 6
        mat3 = nc.Matrix([[1, 2, 3], [4, 5, 6]])

        # This creates a 1 * 2 matrix with entries 4, 5
        mat4 = nc.Matrix(1, 2, [4, 5])

        print("pass basic init")

class TestAddCorrectness:
    def test_small_add(self):
        pass


    def test_medium_add(self):
        # TODO: YOUR CODE HERE
        pass

    def test_large_add(self):
        # TODO: YOUR CODE HERE
        pass

class TestSubCorrectness:
    def test_small_sub(self):
        # TODO: YOUR CODE HERE
        pass

    def test_medium_sub(self):
        # TODO: YOUR CODE HERE
        pass

    def test_large_sub(self):
        # TODO: YOUR CODE HERE
        pass

class TestAbsCorrectness:
    def test_small_abs(self):
        # TODO: YOUR CODE HERE
        pass

    def test_medium_abs(self):
        # TODO: YOUR CODE HERE
        pass

    def test_large_abs(self):
        # TODO: YOUR CODE HERE
        pass

class TestNegCorrectness:
    def test_small_neg(self):
        # TODO: YOUR CODE HERE
        pass

    def test_medium_neg(self):
        # TODO: YOUR CODE HERE
        pass

    def test_large_neg(self):
        # TODO: YOUR CODE HERE
        pass

class TestMulCorrectness:
    def test_small_mul(self):
        # TODO: YOUR CODE HERE
        pass

    def test_medium_mul(self):
        # TODO: YOUR CODE HERE
        pass

    def test_large_mul(self):
        # TODO: YOUR CODE HERE
        pass

class TestPowCorrectness:
    def test_small_pow(self):
        # TODO: YOUR CODE HERE
        pass

    def test_medium_pow(self):
        # TODO: YOUR CODE HERE
        pass

    def test_large_pow(self):
        # TODO: YOUR CODE HERE
        pass

class TestGetCorrectness:
    def test_get(self):
        # TODO: YOUR CODE HERE
        pass

class TestSetCorrectness:
    def test_set(self):
        # TODO: YOUR CODE HERE
        pass

if __name__ == '__main__':
    test = TestBasicCorrectness()
    test.test_init()
    print("pass basic init")