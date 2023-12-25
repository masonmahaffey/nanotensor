import numpy as np

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

# Define a simple linear layer
class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros(output_size)
        self.input = None
        self.output = None
        self.weight_gradients = None
        self.bias_gradient = None

    def forward(self, input):
        self.input = input
        # You need to transpose matrices to ensure that the dimensions match up the dot product
        self.output = np.dot(input, self.weights) + self.bias
        return self.output

    def backward(self, gradient_output):
        # Compute gradients for weights and bias
        print(self.input.shape)
        self.weight_gradients = np.dot(self.input.T, gradient_output)
        self.bias_gradient = np.sum(gradient_output, axis=0, keepdims=True)
        # Compute gradient with respect to input (for passing to previous layer)
        gradient_input = np.dot(gradient_output, self.weights.T)
        return gradient_input

# Neural Network class
class NanoTensor:
    def __init__(self, layers, lr=0.01):
        self.layers = layers
        self.optimizer = SGD(learning_rate=lr)

    def predict(self, x):
        for layer in self.layers:
            # I think there is a bug here given the output could never be -1 if the output layer
            # is a relu activation function.
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
learning_rate = 0.001
layers=[Layer(2, 5), Layer(5, 5), Layer(5, 5), Layer(5, 1)]
net = NanoTensor(layers, learning_rate)

# XOR Example
input_data = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
target = np.array([[1], [1], [0], [1]])

# Training loop
epochs = 100000

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