from mytorch.F.Initializer import xavier_init
from mytorch.nn.Module import Module

import cupy as cp

class Linear(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = xavier_init(input_size, output_size)
        self.bias = cp.zeros((output_size,))
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_size={self.weight.shape[0]}, output_size={self.weight.shape[1]})"
    
    def forward(self, x):
        # x is cp array
        self.input = x

        x = cp.matmul(x, self.weight) + self.bias
        return x

    def backward_calc(self, error, lr):
        error = self.error_grad(error)

        delta_weight = cp.einsum("ij,ik->ijk", self.input, error)
        delta_error = cp.matmul(self.weight, error.T)
        
        self.weight -= lr * cp.mean(delta_weight, axis = 0)
        self.bias -= lr * cp.mean(error, axis=0)
        
        return delta_error.T  