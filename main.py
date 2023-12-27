import numpy as np

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

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
        layer.weights -= self.learning_rate * layer.weight_gradients
        layer.bias -= self.learning_rate * layer.bias_gradients.reshape(layer.bias.shape)

        if layer.use_batchnorm:
            layer.gamma -= self.learning_rate * layer.gamma_gradients
            layer.beta -= self.learning_rate * layer.beta_gradients

# Define a simple linear layer
class Layer:
    def __init__(self, input_size, output_size, use_batchnorm=False):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros(output_size)
        self.use_batchnorm = use_batchnorm

        # Batch normalization parameters
        if use_batchnorm:
            self.gamma = np.ones(output_size)
            self.beta = np.zeros(output_size)
            self.running_mean = np.zeros(output_size)
            self.running_var = np.zeros(output_size)

        self.input = None
        self.output = None
        self.normalized_output = None
        self.batch_mean = None
        self.batch_var = None
        self.weight_gradients = None
        self.bias_gradients = None
        self.gamma_gradients = None
        self.beta_gradients = None

    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias

        if self.use_batchnorm:
            self.batch_mean = np.mean(self.output, axis=0)
            self.batch_var = np.var(self.output, axis=0)
            self.normalized_output = (self.output - self.batch_mean) / np.sqrt(self.batch_var + 1e-7)
            self.output = self.gamma * self.normalized_output + self.beta

            # Update running statistics for inference
            self.running_mean = 0.9 * self.running_mean + 0.1 * self.batch_mean
            self.running_var = 0.9 * self.running_var + 0.1 * self.batch_var

        return self.output

    def backward(self, gradient_output):
        gradient_clip = 0.001

        if self.use_batchnorm:
            self.gamma_gradients = np.sum(gradient_output * self.normalized_output, axis=0)
            self.beta_gradients = np.sum(gradient_output, axis=0)
            gradient_output = gradient_output * self.gamma / np.sqrt(self.batch_var + 1e-7)

        self.weight_gradients = np.dot(self.input.T, gradient_output)
        self.weight_gradients = np.clip(self.weight_gradients, -gradient_clip, gradient_clip)

        self.bias_gradients = np.sum(gradient_output, axis=0, keepdims=True)
        self.bias_gradients = np.clip(self.bias_gradients, -gradient_clip, gradient_clip)

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
                # x = relu(layer.forward(x))
                x = gelu(layer.forward(x))
                # x = mactivation(layer.forward(x))
        return x

    def backpropagate(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def update_weights(self):
        for layer in self.layers:
            self.optimizer.update(layer)

# Example Usage
learning_rate = 0.0005
layers = [Layer(2, 5, use_batchnorm=True), Layer(5, 10, use_batchnorm=True), Layer(10, 10, use_batchnorm=True), Layer(10, 5, use_batchnorm=True), Layer(5, 1)]
net = NanoTensor(layers, learning_rate)

# XOR Example
input_data = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
target = np.array([[1], [1], [0], [1]])

# Training loop
epochs = 1000000

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