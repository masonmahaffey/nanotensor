# About Computational Graphs:
# 1) Computational graphs provide a clear and organized 
# representation of all the operations and variables
# in a computation. The graph structure makes it 
# easier to visualize and manage these operations.
# 2) In a computational graph, each node represents an 
# operation (like addition, multiplication, or a more
# complex function), and edges represent the flow of 
# data. During the forward pass, values (or tensors) 
# are propagated through the graph. During the 
# backward pass, gradients are computed by applying 
# the chain rule of calculus at each node, moving 
# from the output back to the inputs.
# 3) With a graph structure, it's easier to identify
# independent operations that can be computed in 
# parallel, which is essential for efficient 
# utilization of modern hardware like GPUs and TPUs.
# 4) Computational graphs allow for advanced 
# optimizations. For instance, some operations 
# can be merged or simplified, and unnecessary 
# computations can be identified and removed. 
# This optimization can lead to significant 
# performance improvements.

class Tensor:
    def __init__(self, data, should_autograd=True) -> None:
        self.__data = data
        self.data = data
        self.grad = None
        self.should_autograd = should_autograd
        self._backward = lambda: None

    def __add__(self, other):
        result = Tensor(self.data + other)
        print("add, other: ", result.data)
        return result

    def __mul__(self, other):
        result = Tensor(self.data * other)
        print('mul, other', result.data)
        return result

    def __call__(self):
        print(self.data)

    # This backward method is an implementation of 
    # the backward pass of backpropagation in a 
    # computational graph. It first determines the 
    # order in which gradients should be computed
    # through topological sorting, and then it 
    # iterates through these nodes in reverse order 
    # to apply the chain rule and compute gradients. 
    # Each node's _backward method would handle the 
    # specifics of how to compute and propagate 
    # these gradients.
    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

