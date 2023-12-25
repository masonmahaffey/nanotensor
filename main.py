import numpy as np

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Linear activation function
def linear(x):
    return x

# Mean Squared Error loss
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# SGD optimizer
class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, layer):
        layer.weights -= self.learning_rate * layer.weight_gradients
        # Corrected bias update
        layer.bias -= self.learning_rate * layer.bias_gradients.reshape(layer.bias.shape)

# Define a simple linear layer
class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros(output_size)
        self.input = None
        self.output = None
        self.weight_gradients = None
        self.bias_gradients = None

    def forward(self, input):
        self.input = input
        # This creates a dense layer connecting every neuron in the previous layer to the current layer
        # via a weight matrix of size input_size x output_size
        self.output = np.dot(input, self.weights) + self.bias
        return self.output

    def backward(self, gradient_output):
        # Compute gradients for weights and bias
        # You need to transpose matrices to ensure that the dimensions match up the dot product
        self.weight_gradients = np.dot(self.input.T, gradient_output)
        self.bias_gradients = np.sum(gradient_output, axis=0, keepdims=True)
        # Compute gradient with respect to input (for passing to previous layer)
        gradient_input = np.dot(gradient_output, self.weights.T)
        return gradient_input

# Neural Network class
class NanoTensor:
    def __init__(self, layers, lr=0.01):
        self.layers = layers
        self.optimizer = SGD(learning_rate=lr)

    def predict(self, x):
        # Here we need to use a different activation function than ReLU for the output
        # layer or else our network will not be able to predict negative values due to
        # the way ReLU works, it is more performant but the tradeoff is that it cannot
        # output negative values, only positive
        for index, layer in enumerate(self.layers):
            if index == len(self.layers) - 1:
                x = linear(layer.forward(x))
            else:
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
layers=[Layer(2, 5), Layer(5, 10), Layer(10, 5), Layer(5, 1)]
net = NanoTensor(layers, learning_rate)

# XOR Example
input_data = np.array([[0, -1], [-1, 0], [0, 0], [-1, -1]])
target = np.array([[-1], [-1], [0], [-1]])

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