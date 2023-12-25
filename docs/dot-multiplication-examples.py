import numpy as np

# VECTOR MULTIPLICATION EXAMPLE
vector1 = np.array([2, 4, 6])
vector2 = np.array([3, 5, 7])
dot_product = np.dot(vector1, vector2)
print(dot_product)  # Output: 2*3 + 4*5 + 6*7 = 68

# MATRIX MULTIPLICATION EXAMPLE
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
result = np.dot(matrix1, matrix2)
print(result)  # Output: [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]] = 19,22

# VECTOR AND MATRIX MULTIPLICATION EXAMPLE
matrix = np.array([[1, 2], [3, 4], [5, 6]])
vector = np.array([7, 8])
result = np.dot(matrix, vector)
print(result)  # Output: [1*7 + 2*8, 3*7 + 4*8, 5*7 + 6*8] = 43,50

# HIGHER DIMENSIONAL ARRAY MULTIPLICATION EXAMPLE
tensor1 = np.random.rand(3, 4, 5)
tensor2 = np.random.rand(5, 2)
result = np.dot(tensor1, tensor2)
print(result.shape)  # Output: (3, 4, 2)

# SIMPLE MULTIPLICATION EXAMPLE
a = np.array([[2]])
b = np.array([[2]])
print(np.dot(a, b)) # Output: 2*2 = 4
