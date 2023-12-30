import sys
import os

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)
sys.path.append(parent_directory)

import torch
import torch.nn as nn
import torch.optim as optim
import time
from nanotensor.nano import Layer, NanoTensorNetwork, relu, linear, mse_gradient, mse_loss
import numpy as np

def nanotensor_xor_network():
    start_time = time.time()

    # Define layers
    layers = [
        Layer(2, 5, use_batchnorm=False, activation=relu), 
        Layer(5, 10, use_batchnorm=False, activation=relu), 
        Layer(10, 10, use_batchnorm=False, activation=relu),
        Layer(10, 5, use_batchnorm=False, activation=relu), 
        Layer(5, 1, use_batchnorm=False, activation=linear)
    ]

    # Initialize network
    net = NanoTensorNetwork(layers, learning_rate=0.01, gradient_clipping=0.001)

    # XOR data
    input_data = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    target = np.array([[1], [1], [0], [1]])

    # Training loop
    epochs = 10000000
    for epoch in range(epochs):
        predictions = net.predict(input_data)
        loss = mse_loss(target, predictions)
        prediction_gradients = mse_gradient(target, predictions)
        net.compute_gradients(prediction_gradients)
        net.apply_gradients()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

        if loss <= 0.001:
            break

    print("Final Predictions:", net.predict(input_data))
    end_time = time.time()
    print(f"Training completed in {end_time - start_time} seconds")

if __name__:
    nanotensor_xor_network();