import sys
import os

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)
sys.path.append(parent_directory)

import torch
import torch.nn as nn
import torch.optim as optim
import time
from nanotensor.nano import Layer, NanoTensorNetwork, relu, linear, mse_loss, mse_gradient
import numpy as np
import concurrent.futures

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
    net = NanoTensorNetwork(layers, learning_rate=0.01, gradient_clipping=0.1)

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

        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch}, Loss: {loss}")

        if loss <= 0.01:
            break

    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    # print("Final Predictions:", net.predict(input_data))
    print(f"nanotensor training completed in {elapsed_time_ms:.2f} ms")

def pytorch_xor_network():
    start_time = time.time()

    class PyTorchXORNet(nn.Module):
        def __init__(self):
            super(PyTorchXORNet, self).__init__()
            self.fc1 = nn.Linear(2, 5)
            self.fc2 = nn.Linear(5, 10)
            self.fc3 = nn.Linear(10, 10)
            self.fc4 = nn.Linear(10, 5)
            self.fc5 = nn.Linear(5, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = self.fc5(x)  # Linear activation at the output
            return x

    net = PyTorchXORNet()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    input_data = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]], dtype=torch.float)
    target = torch.tensor([[1], [1], [0], [1]], dtype=torch.float)

    epochs = 10000000
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = net(input_data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch}, Loss: {loss.item()}")

        if loss.item() <= 0.01:
            break

    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    # print("Final Predictions:", net(input_data))
    print(f"PyTorch Training completed in {elapsed_time_ms:.2f} ms")

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit tasks to the executor
        executor.submit(pytorch_xor_network)
        executor.submit(nanotensor_xor_network)
