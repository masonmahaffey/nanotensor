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

def softmax(x):
    # Subtracting the max value for numerical stability
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# Mean Squared Error loss
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Gradient of MSE
def mse_gradient(y_true, y_pred):
    return 2 * (y_pred - y_true)

def cross_entropy_loss(y_true, y_pred):
    # Small value to avoid division by zero in log
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    N = y_pred.shape[0]
    ce_loss = -np.sum(y_true * np.log(y_pred)) / N
    return ce_loss

def cross_entropy_gradient(y_true, y_pred):
    N = y_pred.shape[0]
    grad = - (y_true / y_pred) / N
    return grad

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

class Adam:
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer):
        # Initialize m and v for each layer if not already done
        if layer not in self.m:
            self.m[layer] = [np.zeros_like(layer.weights), np.zeros_like(layer.bias)]
            self.v[layer] = [np.zeros_like(layer.weights), np.zeros_like(layer.bias)]

        self.t += 1

        # Update biased first moment estimate and biased second raw moment estimate
        self.m[layer][0] = self.beta_1 * self.m[layer][0] + (1 - self.beta_1) * layer.weight_gradients
        self.m[layer][1] = self.beta_1 * self.m[layer][1] + (1 - self.beta_1) * layer.bias_gradients

        self.v[layer][0] = self.beta_2 * self.v[layer][0] + (1 - self.beta_2) * (layer.weight_gradients ** 2)
        self.v[layer][1] = self.beta_2 * self.v[layer][1] + (1 - self.beta_2) * (layer.bias_gradients ** 2)

        # Compute bias-corrected first moment estimate and second raw moment estimate for weights
        m_hat_weight = self.m[layer][0] / (1 - self.beta_1 ** self.t)
        v_hat_weight = self.v[layer][0] / (1 - self.beta_2 ** self.t)

        # Update weights
        layer.weights -= self.learning_rate * m_hat_weight / (np.sqrt(v_hat_weight) + self.epsilon)

        # Compute bias-corrected first moment estimate and second raw moment estimate for bias
        m_hat_bias = self.m[layer][1].reshape(layer.bias.shape) / (1 - self.beta_1 ** self.t)
        v_hat_bias = self.v[layer][1].reshape(layer.bias.shape) / (1 - self.beta_2 ** self.t)

        # Update biases
        layer.bias -= self.learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)

# Define a simple linear layer
class Layer:
    def __init__(self, input_size, output_size, use_batchnorm=False, activation=relu):
        self.activation = activation

        # He initialization
        if activation == relu or activation == leaky_relu:
            scale = np.sqrt(2.0 / input_size)
        else:  
            # For other activations, use Xavier initialization
            scale = np.sqrt(1.0 / input_size)

        self.weights = np.random.randn(input_size, output_size) * scale
        self.bias = np.zeros(output_size)

        self.input = None
        self.input_computed = None
        self.normalized_output = None
        self.batch_mean = None
        self.batch_var = None
        self.weight_gradients = None
        self.bias_gradients = None

        # Batch normalization parameters
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.gamma = np.ones(output_size)
            self.beta = np.zeros(output_size)
            self.running_mean = np.zeros(output_size)
            self.running_var = np.zeros(output_size)
        self.gamma_gradients = None
        self.beta_gradients = None
        
        
    
    def backward(self, gradient_output, gradient_clip, gradient_minimum):
        # Optional: Apply batch normalization gradients if enabled
        if self.use_batchnorm:
            self.gamma_gradients = np.sum(gradient_output * self.normalized_output, axis=0)
            self.beta_gradients = np.sum(gradient_output, axis=0)
            # Compute gradient w.r.t normalized output
            grad_normalized = gradient_output * self.gamma

            # Compute gradient w.r.t variance
            grad_var = np.sum(grad_normalized * (self.input_computed - self.batch_mean), axis=0) * -0.5 * (self.batch_var + 1e-7)**(-1.5)

            # Compute gradient w.r.t mean
            grad_mean = np.sum(grad_normalized, axis=0) * -1 / np.sqrt(self.batch_var + 1e-7)
            grad_mean += grad_var * np.mean(-2 * (self.input_computed - self.batch_mean), axis=0)

            # Compute gradient w.r.t input_computed
            gradient_output = grad_normalized / np.sqrt(self.batch_var + 1e-7)
            gradient_output += grad_var * 2 * (self.input_computed - self.batch_mean) / gradient_output.shape[0]
            gradient_output += grad_mean / gradient_output.shape[0]

        self.weight_gradients = np.dot(self.input.T, gradient_output)
        # Clip gradients
        self.weight_gradients = np.clip(self.weight_gradients, -gradient_clip, gradient_clip)
        # Apply minimum gradient threshold
        self.weight_gradients = np.where(np.abs(self.weight_gradients) < gradient_minimum,
                                        np.sign(self.weight_gradients) * gradient_minimum,
                                        self.weight_gradients)

        self.bias_gradients = np.sum(gradient_output, axis=0, keepdims=True)
        # Clip gradients
        self.bias_gradients = np.clip(self.bias_gradients, -gradient_clip, gradient_clip)
        # Apply minimum gradient threshold
        self.bias_gradients = np.where(np.abs(self.bias_gradients) < gradient_minimum,
                                    np.sign(self.bias_gradients) * gradient_minimum,
                                    self.bias_gradients)

        gradient_input = np.dot(gradient_output, self.weights.T)
        return gradient_input

    def forward(self, input):
        # For the first layer this is all of the samples (x) and their features (y) whereas
        # for subsequent layers this is the output of the previous layer's activation functions
        self.input = input
        # This is simply multiplying either the output of the previous layer's activation function
        # or for the first layer the input of the network (data, samples and their features) times
        # the weights for this layer and then adding the bias
        self.input_computed = np.dot(input, self.weights) + self.bias

        if self.use_batchnorm:
            if self.training:  # Add a flag to check if the network is in training mode
                self.batch_mean = np.mean(self.input_computed, axis=0)
                self.batch_var = np.var(self.input_computed, axis=0)

                # Update running statistics for inference
                self.running_mean = 0.9 * self.running_mean + 0.1 * self.batch_mean
                self.running_var = 0.9 * self.running_var + 0.1 * self.batch_var
            else:
                self.batch_mean = self.running_mean
                self.batch_var = self.running_var

            self.normalized_output = (self.input_computed - self.batch_mean) / np.sqrt(self.batch_var + 1e-7)
            self.input_computed = self.gamma * self.normalized_output + self.beta

        return self.activation(self.input_computed)

# Neural Network class
class NanoTensorNetwork:
    def __init__(self, layers, learning_rate=0.01, optimizer=SGD, gradient_clipping=0.000000001, gradient_minimum=0.001):
        self.layers = layers
        self.optimizer = optimizer(learning_rate=learning_rate)
        self.gradient_clipping = gradient_clipping
        self.gradient_minimum = gradient_minimum
        self.training = True

        # Add a method to toggle training mode
    def train(self):
        self.training = True
        for layer in self.layers:
            if layer.use_batchnorm:
                layer.training = True

    # Add a method to toggle evaluation mode
    def eval(self):
        self.training = False
        for layer in self.layers:
            if layer.use_batchnorm:
                layer.training = False

    def predict(self, x):
        # Initially x is all of the samples with their features...
        for layer in self.layers:
            x = layer.forward(x)
        return x

    # Backpropagation
    def compute_gradients(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, self.gradient_clipping, self.gradient_minimum)

    # Update all weights, biases, and parameters using the computed gradients via backpropagation
    def apply_gradients(self):
        for layer in self.layers:
            self.optimizer.update(layer)