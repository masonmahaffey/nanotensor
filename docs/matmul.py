# !/usr/bin/env python3
import numpy as np
import time

# Simple but also horrible matmul
def matmul(matrix1, matrix2):
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Number of columns in the first matrix must be equal to the number of rows in the second matrix.")

    result = []
    for i in range(len(matrix1)):
        row = []
        for j in range(len(matrix2[0])):
            dot_product = 0
            for k in range(len(matrix2)):
                dot_product += matrix1[i][k] * matrix2[k][j]
            row.append(dot_product)
        result.append(row)

    return result

# much better
def matmul_optimized(matrix1, matrix2):
    # Check for valid matrix sizes for multiplication
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Matrix sizes are incompatible for multiplication.")

    # Transpose the second matrix for better memory access patterns
    matrix2_t = [list(x) for x in zip(*matrix2)]

    # Perform matrix multiplication using list comprehensions
    return [[sum(a * b for a, b in zip(row, col)) for col in matrix2_t] for row in matrix1]


# worse even though in theory it's better?
# might be constrained by the python interpreter
def matmul_further_optimized(matrix1, matrix2, block_size=64):
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Matrix sizes are incompatible for multiplication.")

    n, m, p = len(matrix1), len(matrix2[0]), len(matrix2)
    result = [[0] * m for _ in range(n)]

    for i0 in range(0, n, block_size):
        for j0 in range(0, m, block_size):
            for k0 in range(0, p, block_size):
                for i in range(i0, min(i0 + block_size, n)):
                    for j in range(j0, min(j0 + block_size, m)):
                        sum = result[i][j]
                        for k in range(k0, min(k0 + block_size, p)):
                            sum += matrix1[i][k] * matrix2[k][j]
                        result[i][j] = sum

    return result



N = 100
if __name__ == "__main__":
    

    A = np.random.randn(N,N).astype(np.float32)
    B = np.random.randn(N,N).astype(np.float32)

    totaltime = 0
    iterations = 100

    dots = []
    for i in range(iterations):
        start = time.monotonic()
        # # numpy
        # C = A @ B
        C = matmul(A, B)


        end = (time.monotonic() - start) * 1000
        totaltime += end

        if i % 5 == 0:
            output = ""
            dots.append(".")
            for dot in dots:
                output = output + dot
            print(output)

    avg1 = totaltime / iterations


    totaltime = 0
    dots = []
    for i in range(iterations):
        start = time.monotonic()
        # matmul
        C = matmul_optimized(A, B)
        
        end = (time.monotonic() - start) * 1000
        totaltime += end

        if i % 5 == 0:
            output = ""
            dots.append(".")
            for dot in dots:
                output = output + dot
            print(output)
    avg2 = totaltime / iterations


    totaltime = 0
    dots = []
    for i in range(iterations):
        start = time.monotonic()
        # matmul
        C = matmul_further_optimized(A, B)
        
        end = (time.monotonic() - start) * 1000
        totaltime += end

        if i % 5 == 0:
            output = ""
            dots.append(".")
            for dot in dots:
                output = output + dot
            print(output)
    avg3 = totaltime / iterations

    print("Average matmul: ", avg1)
    print("Average matmul optimized: ", avg2)
    print("Average matmul further optimized: ", avg3)