import numpy as np

# Define a simple linear layer
class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros(output_size)
        self.input = None
        self.output = None
        self.gradient_weights = None
        self.gradient_bias = None

    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        return self.output

    def backward(self, gradient_output):
        # Compute gradients for weights and bias
        self.gradient_weights = np.dot(self.input.T, gradient_output)
        self.gradient_bias = np.sum(gradient_output, axis=0, keepdims=True)
        # Compute gradient with respect to input (for passing to previous layer)
        gradient_input = np.dot(gradient_output, self.weights.T)
        return gradient_input

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Mean Squared Error loss
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# SGD optimizer
class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, layer):
        layer.weights -= self.learning_rate * layer.gradient_weights
        # Corrected bias update
        layer.bias -= self.learning_rate * layer.gradient_bias.reshape(layer.bias.shape)

# Neural Network class
class NanoTensor:
    def __init__(self, layers):
        self.layers = layers
        self.optimizer = SGD(learning_rate=0.01)

    def predict(self, x):
        for layer in self.layers:
            x = relu(layer.forward(x))
        return x

    def compute_loss(self, y_true, y_pred):
        return mse_loss(y_true, y_pred)

    def backpropagate(self, gradient_output):
        for layer in reversed(self.layers):
            gradient_output = layer.backward(gradient_output)

    def update_weights(self):
        for layer in self.layers:
            self.optimizer.update(layer)

# Example Usage
net = NanoTensor([Layer(2, 5), Layer(5, 5), Layer(5, 5), Layer(5, 1)])

# XOR Example
input_data = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
target = np.array([[1], [1], [0], [1]])

# Training loop
epochs = 5000
learning_rate = 0.001
for epoch in range(epochs):
    # Forward pass
    predictions = net.predict(input_data)
    loss = net.compute_loss(target, predictions)

    # Backpropagation
    gradient_output = 2 * (predictions - target)  # Gradient of MSE loss
    net.backpropagate(gradient_output)

    # Weight update using SGD optimizer
    net.update_weights()

    # if epoch % 100 == 0:
        # print(f"Epoch {epoch}, Loss: {loss}")
    print(f"Epoch {epoch}, Loss: {loss}")

print("Final Predictions:", net.predict(input_data))