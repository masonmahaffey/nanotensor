import torch

# Define a tensor x with gradient tracking
x = torch.tensor([4.0], requires_grad=True)

# Define the linear function: f(x) = 3x + 2
f = 3 * x + 2

# Compute gradients
f.backward()

# Get the gradient
x_grad = x.grad

# Outputs tensor[3.])
print("Gradient of f with respect to x:", x_grad)

# Since the function is linear, this gradient is constant. 
# The derivative of 3x with respect to x is 3, indicating that for every unit increase in x, 
# f(x) increases by 3 units.