<div align="center">

[![logo](https://raw.githubusercontent.com/masonmahaffey/nanotensor/main/docs/logoprimary.png)](https://masonmahaffey.com)
</div>

---

This is a fun project I've started to learn about automatic differentiation, matmul, tensor multiplication, GPUs, and autograd libraries.

## Features
- **Simplicity in Design**: Keeping the API simple and the architecture transparent, nanotensor is ideal for those beginning their journey in neural network implementation.
- **Educational Tool**: With an emphasis on clarity and comprehensibility, nanotensor serves as an excellent educational tool for understanding the fundamentals of neural networks.

## Getting Started
To get started with nanotensor:

1. Clone the repository: git clone https://github.com/masonmahaffey/nanotensor.git
2. Install dependencies:
3. Run a sample model:


## CUDA Backend Integration
nanotensor is soon to be integrated with a CUDA backend, enabling it to leverage GPU acceleration for neural network computations. This integration allows for significant performance improvements, particularly in larger and more complex models.

### Setting Up CUDA
Ensure that your system has the appropriate NVIDIA drivers and CUDA toolkit installed. For detailed instructions, refer to the [CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads).

### Using CUDA with nanotensor
To enable CUDA:

Import the CUDA module:
```python
from nanotensor.backend import cuda
cuda.enable()
```

nanotensor will automatically utilize the GPU for computations where possible.

## Contributions
Contributions to nanotensor are welcome!

## License
nanotensor is released under the MIT License.