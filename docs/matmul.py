# !/usr/bin/env python3
import numpy as np
import time
import os
os.environ['OMP_NUM_THREADS'] = '1'

def matmul(matrix1, matrix2):
    # comp check
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Number of columns in the first matrix must be equal to the number of rows in the second matrix.")
    
    # Determine the data type of the result based on the data types of input matrices
    result_dtype = np.result_type(matrix1, matrix2)

    # Initialize the result as a NumPy array filled with zeros and the determined data type
    result = np.zeros((len(matrix1), len(matrix2[0])), dtype=result_dtype)

    # matrix 2 x traversal
    for m2x in range(len(matrix2[0])):
        # matrix1 y traversal
        for m1y in range(len(matrix1)):
            # matrix 1 x traversal
            dp = 0
            for m1x in range(len(matrix1[m1y])):
                dp += matrix1[m1y,m1x] * matrix2[m1x,m2x]
                # how do I set this dot product to the new result?
            result[m1y,m2x] = dp

    return result


# Simple but also horrible matmul
def matmul2(matrix1, matrix2):
    # compare the x axis of matrix 1 with the y axis of matrix 2
    # or in otherwords, ensure the width of rows is equivalent to the height of columns
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Number of columns in the first matrix must be equal to the number of rows in the second matrix.")

    # Determine the data type of the result based on the data types of input matrices
    result_dtype = np.result_type(matrix1, matrix2)

    # Initialize the result as a NumPy array filled with zeros and the determined data type
    result = np.zeros((len(matrix1), len(matrix2[0])), dtype=result_dtype)

    # matrix 1 y axis/column traversal
    for i in range(len(matrix1)):
        # matrix 2 x axis/row traversal
        for j in range(len(matrix2[0])):
            dot_product = 0
            # matrix 2 y axis/column traversal?
            for k in range(len(matrix2)):
                dot_product += matrix1[i,k] * matrix2[k,j]
            result[i, j] = dot_product

    return result

N = 1000
if __name__ == "__main__":

    A = np.random.randn(N,N).astype(np.float32)
    B = np.random.randn(N,N).astype(np.float32)

    flop = 2*N*N*N

    totaltime = 0
    gflops = 0
    for i in range(100):
        start = time.monotonic()
        C = A @ B
        end = time.monotonic()
        s = end-start
        gflops += flop/s * 1e-9

        print(f"{flop/s * 1e-9:.2f} GFLOP/S, {s*1e3:.2f} ms")
    
    print()
    print(f"{gflops/100:.2f} Average GFLOP/S, {s*1e3:.2f} ms")
    print()