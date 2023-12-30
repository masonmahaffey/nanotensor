import sys
import os

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)
sys.path.append(parent_directory)

import numpy as np
import time
from nanotensor.nano import Layer, NanoTensorNetwork, Adam, relu, softmax, cross_entropy_loss, cross_entropy_gradient
from keras.datasets import mnist

def nanotensor_mnist_network():
    start_time = time.time()

    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  # Flatten and normalize
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0  # Flatten and normalize

    # Convert labels to one-hot encoding
    y_train_onehot = np.eye(10)[y_train]
    y_test_onehot = np.eye(10)[y_test]

    # Define layers for MNIST
    layers = [
        Layer(784, 784 * 2, use_batchnorm=True, activation=relu),
        Layer(784 * 2, 784, use_batchnorm=True, activation=relu),
        Layer(784, 512, use_batchnorm=True, activation=relu),
        Layer(512, 128, use_batchnorm=True, activation=relu),
        Layer(128, 10, activation=softmax)  # Last layer typically does not use batch normalization
    ]

    # Initialize network
    net = NanoTensorNetwork(layers, optimizer=Adam, learning_rate=0.001, gradient_clipping=0.1, gradient_minimum=0.0001)

    # Training loop
    epochs = 20
    losses = []
    for epoch in range(epochs):
        net.train()  # Set the network to training mode
        predictions = net.predict(X_train)
        loss = cross_entropy_loss(y_train_onehot, predictions)
        losses.append(loss)

        # early stopping
        if losses[-1] > loss & losses[-2] > loss:
            break

        prediction_gradients = cross_entropy_gradient(y_train_onehot, predictions)
        net.compute_gradients(prediction_gradients)
        net.apply_gradients()

        print(f"Epoch {epoch}, Loss: {loss}")

    # Evaluate on test data
    net.eval()  # Set the network to evaluation mode
    test_predictions = net.predict(X_test)
    test_loss = cross_entropy_loss(y_test_onehot, test_predictions)
    print("Test Loss:", test_loss)

    # Calculate test accuracy
    predicted_classes = np.argmax(test_predictions, axis=1)
    correct_predictions = np.sum(predicted_classes == y_test)
    accuracy = correct_predictions / len(y_test)
    print("Test Accuracy:", accuracy)

    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000
    print(f"Training completed in {elapsed_time_ms:.2f} ms")

nanotensor_mnist_network()
