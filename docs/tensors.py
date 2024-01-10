# !/usr/bin/env python3
import sys
import os

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)
sys.path.append(parent_directory)

import numpy as np
import time
from nanotensor.tensor import Tensor

N = 4096
if __name__ == "__main__":
    # N^2
    A = np.random.randn(N,N).astype(np.float32)
    # N^2
    B = np.random.randn(N,N).astype(np.float32)

    # floating point operation
    flop = N * N * 2 * N
    print(f"{flop / 1e9:.2f} GFLOP")

    for i in range(100):
        start_time = time.monotonic()
        C = A @ B
        end_time = time.monotonic()

        s = end_time - start_time
        elapsed_time_ms = (end_time - start_time) * 1000
        print(elapsed_time_ms)
        print(f"{flop/s * 1e-12:.2f} TFLOP/S")


    
    