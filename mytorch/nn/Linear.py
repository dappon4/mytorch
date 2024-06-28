from mytorch.F.Initializer import xavier_init
from mytorch.nn.Module import Module
import cupy as cp

class Linear(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = xavier_init(input_size, output_size)
        self.bias = None
        self.input = None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_size={self.weight.shape[0]}, output_size={self.weight.shape[1]})"
    
    def forward(self, x):
        self.input = x
        x = cp.matmul(x, self.weight)
        
        if self.bias is None:
            self.bias = cp.zeros(x.shape[1:], dtype=cp.float32)
        
        x += self.bias
        return x

    def backward_calc(self, error, lr):
        input_T_axis = list(range(len(self.input.shape)))
        input_T = self.input.transpose(*input_T_axis[:-2], input_T_axis[-1], input_T_axis[-2])
        
        delta_weight = cp.matmul(input_T, error)
        delta_error = cp.matmul(error, self.weight.T)
        
        self.weight -= lr * cp.mean(delta_weight, axis = 0)
        self.bias -= lr * cp.mean(error, axis=0)
        
        return delta_error