import numpy as np

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Linear activation function
def linear(x):
    return x

# Mean Squared Error loss
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Gradient of MSE
def mse_gradient(y_true, y_pred):
    return 2 * (y_pred - y_true)

# SGD optimizer
class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, layer):
        # Decrements because you are trying to descend the gradient to the minima
        layer.weights -= self.learning_rate * layer.weight_gradients
        layer.bias -= self.learning_rate * layer.bias_gradients.reshape(layer.bias.shape)

# Define a simple linear layer
class Layer:
    def __init__(self, input_size, output_size):
        # With the example it's a 2x5 matrix of weights with each input feature having one neuron with a connection
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros(output_size)
        self.input = None
        self.output = None
        self.weight_gradients = None
        self.bias_gradients = None

    def forward(self, input):
        self.input = input
        # Creates a dense layer and allows us to compute all samples at once
        self.output = np.dot(input, self.weights) + self.bias
        return self.output

    def backward(self, gradient_output):
        # Compute the gradient of each weight with respect to the loss using the
        # downstream layer's output gradient
        self.weight_gradients = np.dot(self.input.T, gradient_output)
        self.weight_gradients = np.clip(self.weight_gradients, -1.0, 1.0) # Apply gradient clipping to prevent exploding gradients

        # Compute the gradient of the bias with respect to the downstream layer's output
        self.bias_gradients = np.sum(gradient_output, axis=0, keepdims=True)
        self.bias_gradients = np.clip(self.bias_gradients, -1.0, 1.0)

        # Compute the gradient of the output of the upstream layer which is connected to this layer with respect to the loss
        gradient_input = np.dot(gradient_output, self.weights.T)

        # Return the gradient of the output of the upstream layer so that the gradients
        # for weights in the upstream layer can be computed
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
                # x = mactivation(layer.forward(x))
        return x

    def backpropagate(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def update_weights(self):
        for layer in self.layers:
            self.optimizer.update(layer)

# Example Usage
learning_rate = 0.001
layers=[Layer(2, 5), Layer(5, 10), Layer(10,10), Layer(10, 5), Layer(5, 1)]
net = NanoTensor(layers, learning_rate)

# XOR Example
input_data = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
target = np.array([[1], [1], [0], [1]])

# Training loop
epochs = 500000

for epoch in range(epochs):
    # Forward pass
    predictions = net.predict(input_data)
    # Mean squared error loss to measure model prediction performance
    loss = mse_loss(target, predictions)

    # Backpropagation
    multidim_slope = mse_gradient(target, predictions) # The gradient of the loss, applies only to the output layer
    net.backpropagate(multidim_slope)

    # Weight update using SGD optimizer
    net.update_weights()

    # if epoch % 100 == 0:
        # print(f"Epoch {epoch}, Loss: {loss}")
    print(f"Epoch {epoch}, Loss: {loss}")

print("Final Predictions:", net.predict(input_data))