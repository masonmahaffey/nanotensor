import sys
import os

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)
sys.path.append(parent_directory)

import numpy as np
import time
from nanotensor.tensor import Tensor

x = Tensor(2)
x()

y = x * 2 + 4