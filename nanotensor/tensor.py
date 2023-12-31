class Tensor:
    def __init__(self, data, should_autograd=True) -> None:
        self.__data = data
        self.data = data
        self.grad = None
        self.should_autograd = should_autograd

    def backward():
        pass

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
    
    
    

