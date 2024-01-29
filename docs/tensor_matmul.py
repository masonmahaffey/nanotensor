import time
import torch
import numpy as np

def compare_float32_tensors(tensor1, tensor2):
    """
    Compare all values of two float32 tensors up to a precision of 5 decimal places.
    
    Args:
    - tensor1 (torch.Tensor): The first float32 tensor to compare.
    - tensor2 (torch.Tensor): The second float32 tensor to compare.

    Returns:
    - bool: True if all values in the tensors are equal up to 5 decimal places, False otherwise.
    """
    # Check if tensors shapes match
    if tensor1.shape != tensor2.shape:
        return False
        
    # Define precision level
    precision = 5
    
    # Calculate the difference and check if it's within the allowed tolerance for all elements
    difference = torch.abs(tensor1 - tensor2)
    tolerance = 0.5 * 10 ** (-precision)
    return torch.all(difference < tolerance)


def matmul(matrix1, matrix2):
    # Compatibility check
    if matrix1.shape[1] != matrix2.shape[0]:
        raise ValueError("Number of columns in the first matrix must be equal to the number of rows in the second matrix.")

    # Initialize the result tensor filled with zeros
    # The data type is automatically inferred by PyTorch
    result = torch.zeros((matrix1.size(0), matrix2.size(1)), dtype=matrix1.dtype)

    # Matrix multiplication traversal
    for i in range(matrix2.size(1)):  # Columns of matrix2
        for j in range(matrix1.size(0)):  # Rows of matrix1
            # Compute dot product for element (j, i)
            dp = 0
            for k in range(matrix1.size(1)):  # Columns of matrix1 / Rows of matrix2
                dp += matrix1[j, k] * matrix2[k, i]
            result[j, i] = dp

    return result

# Define a function to perform tensor multiplication using a recursive approach.
def tensor_matmul(t1, t2):
    # Check if the first input (tensor1) is an instance of torch.Tensor. If not, convert it to a torch.Tensor.
    if not isinstance(t1, torch.Tensor):
        t1 = torch.as_tensor(t1)
    
    # Check if the second input (tensor2) is an instance of torch.Tensor. If not, convert it to a torch.Tensor.
    if not isinstance(t2, torch.Tensor):
        t2 = torch.as_tensor(t2)

    # Get the shape of the first tensor as a list of its dimensions.
    t1shape = list(t1.size())
    # Get the shape of the second tensor as a list of its dimensions.
    t2shape = list(t2.size())

    # Assert that the inner dimensions of both tensors match; raise an error if not.
    assert t1shape[-1] == t2shape[-2], "Inner dimensions must match for multiplication."

    # Calculate the shape of the resulting tensor by combining the outer dimensions of tensor1 and tensor2.
    result_shape = t1shape[:-1] + t2shape[-1:]

    print("result_shape: ", result_shape)

    # print("t1shape", t1shape)


    # TODO: Write an assertion to enforce batch dimensions to match
    bshape = t1shape[:-2]
    # print("bshape>", bshape)

    flattened_t1 = t1.view(-1, *t1.shape[-2:])
    flattened_t2 = t2.view(-1, *t2.shape[-2:])

    flattened_result = torch.zeros((flattened_t1.shape[0], *result_shape[-2:]), dtype=t1.dtype)

    for i in range(flattened_t1.shape[0]):
        result = matmul(flattened_t1[i], flattened_t2[i])
        flattened_result[i, :, :] = result
        
    result = flattened_result.view(*result_shape)
    
    return result



if __name__ == "__main__":

    # Example: 3D tensors
    
    a = torch.randn(3, 4, 20, 10)
    b = torch.randn(3, 4, 10, 20)
    start = time.monotonic()
    result = torch.matmul(a, b)  # Batched matrix multiplication
    end = time.monotonic()
    print("result.shape", result.shape)

    start2 = time.monotonic()
    result2 = tensor_matmul(a, b)
    end2 = time.monotonic()

    # milliseconds
    totaltime = (end - start) * 1000
    totaltime2 = (end2 - start2) * 1000

    torch.set_printoptions(precision=10)
    
    print("------------------------------------------------------------------------")
    
    # print(result)
    print()
    # print(result2)
    print()
    print("results are equal: ", compare_float32_tensors(result, result2))
    print()
    print(totaltime)
    print(totaltime2)

 
