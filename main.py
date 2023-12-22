import numpy as np

# Define a simple linear layer
class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros(output_size)

    def forward(self, input):
        return np.dot(input, self.weights) + self.bias

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Mean Squared Error loss
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Neural Network class
class NeuralNetwork:
    def __init__(self):
        self.layers = [Layer(3, 5), Layer(5, 2)]

    def forward(self, x):
        for layer in self.layers:
            x = relu(layer.forward(x))
        return x

    def compute_loss(self, y_true, y_pred):
        return mse_loss(y_true, y_pred)

# Example usage
network = NeuralNetwork()
input_data = np.array([[0.1, 0.2, 0.3]])
target = np.array([[1, 0]])

# Forward pass
predictions = network.forward(input_data)
loss = network.compute_loss(target, predictions)
print("Loss:", loss)
