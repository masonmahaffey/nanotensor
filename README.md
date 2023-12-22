<div align="center">

[![logo](https://raw.githubusercontent.com/masonmahaffey/nanotensor/main/docs/logomain.png)](https://masonmahaffey.com)
</div>

## NanoTensor: An Extremely Small Neural Network Framework 

NanoTensor is a lightweight, easy-to-understand, yet performant neural network framework. Inspired by the work of Ray Solomonoff, a pioneer in algorithmic information theory, NanoTensor seeks to embody the principle that "Intelligence is compression." This framework is designed to make neural network concepts accessible and intuitive, perfect for educational purposes and demonstrative applications.

## Features
- **Simplicity in Design**: Keeping the API simple and the architecture transparent, NanoTensor is ideal for those beginning their journey in neural network implementation.
- **Performance Oriented**: Despite its simplicity, NanoTensor does not compromise on performance, making it suitable for a range of demonstration purposes.
- **Educational Tool**: With an emphasis on clarity and comprehensibility, NanoTensor serves as an excellent educational tool for understanding the fundamentals of neural networks.

## Getting Started
To get started with NanoTensor:

1. Clone the repository: git clone https://github.com/masonmahaffey/nanotensor.git
2. Install dependencies:
3. Run a sample model:


## CUDA Backend Integration
NanoTensor is soon to be integrated with a CUDA backend, enabling it to leverage GPU acceleration for neural network computations. This integration allows for significant performance improvements, particularly in larger and more complex models.

### Setting Up CUDA
Ensure that your system has the appropriate NVIDIA drivers and CUDA toolkit installed. For detailed instructions, refer to the [CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads).

### Using CUDA with NanoTensor
To enable CUDA:

Import the CUDA module:
```python
from nanotensor.backend import cuda
cuda.enable()
```

NanoTensor will automatically utilize the GPU for computations where possible.

## Contributions
Contributions to NanoTensor are welcome!

## License
NanoTensor is released under the MIT License.