from typing import Any
from F.Initializer import he_init_conv2d, xavier_init
from Tensor import Tensor
import cupy as cp

class Module:
    def __init__(self):
        self.prev = set()
        self.training = True
        
        self.error_grad = lambda x: x
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
        
    def __call__(self, *args):
        self.error_grad = lambda x: x
        
        tensor = args[0]
        self.prev = tensor.prev
        
        return Tensor(self.forward(tensor.tensor, *args[1:]), {self})
    
    def forward(self, x):
        raise NotImplementedError

    def backward_calc(self, error, lr):
        return error
    
    def backward(self, error, lr):
        delta_error = self.backward_calc(error, lr)
        
        if self.prev:
            for prev in self.prev:
                prev.backward(delta_error, lr)
        else:
            return delta_error
    
    def train(self):
        self.training = True
        for prev in self.prev:
            prev.train()
    
    def eval(self):
        self.training = False
        for prev in self.prev:
            prev.eval()
    
    def predict(self, X):
        res = self(Tensor(X)).tensor
        return res

class CompoundModule(Module):
    def __init__(self):
        super().__init__()
    
    def __call__(self, *args):
        self.error_grad = lambda x: x
        tensor = self.forward(*args)
        
        self.prev, tensor.prev = tensor.prev, {self}
        
        return tensor
    
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

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if type(kernel_size) == int:
            self.kernel_h = self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
        
        if type(stride) == int:
            self.stride_h = self.stride_w = stride
        else:
            self.stride_h,self.stride_w = stride

        self.padding = padding
        
        self.filter = he_init_conv2d((out_channels, in_channels, self.kernel_h, self.kernel_w))
        self.bias = None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size=({self.kernel_h},{self.kernel_w}), stride={self.stride}, padding={self.padding})"

    def forward(self,x):

        batch_size = x.shape[0]
        output_h = (x.shape[2] - self.kernel_h + 2 * self.padding)//self.stride_h + 1
        output_w = (x.shape[3] - self.kernel_w + 2 * self.padding)//self.stride_w + 1
        
        if self.bias is None:
            self.bias = cp.zeros((1, self.out_channels, output_h, output_w))
        
        self.window_i = cp.repeat(cp.arange(self.kernel_h), self.kernel_w).reshape(1,-1) + (cp.repeat(cp.arange(output_h), output_w)*self.stride_h).reshape(-1,1)
        self.window_j = cp.tile(cp.arange(self.kernel_w),self.kernel_h).reshape(1,-1) + (cp.tile(cp.arange(output_w),output_h)*self.stride_w).reshape(-1,1)

        self.padded_input = cp.pad(x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)))
        
        flattened = self.padded_input[:,:,self.window_i,self.window_j]
        flattened_filter = self.filter.reshape((self.out_channels, self.in_channels, 1, self.kernel_h*self.kernel_w))
        
        # maybe optimze this
        return cp.tensordot(flattened, flattened_filter, axes=([1,3], [1,3])).transpose(0,2,1,3).reshape(batch_size, self.out_channels, output_h, output_w) + self.bias
    
    def backward_calc(self, error, lr):
        error = self.error_grad(error)
        
        # error is cp array of shape (batch_size, out_channels, new_height, new_width)
        batch_size = self.padded_input.shape[0]
        
        error_h, error_w = error.shape[2:]
        error_size = error_h * error_w
        
        filter_i = cp.tile(self.window_i, self.in_channels)
        filter_j = cp.tile(self.window_j, self.in_channels)
        filter_c = cp.repeat(cp.repeat(cp.arange(self.in_channels), self.kernel_h*self.kernel_w).reshape(1,-1), error_size, axis=0)
        
        sub_matrices = self.padded_input[:,filter_c,filter_i,filter_j]
        sub_matrices = cp.expand_dims(sub_matrices, axis = 1)
        
        flattened_error = error.reshape(batch_size, self.out_channels, -1)
        flattened_error = cp.expand_dims(flattened_error, axis = -1)
        
        delta_filter = cp.sum(sub_matrices * flattened_error,axis = 2).reshape(batch_size, self.out_channels, self.in_channels, self.kernel_h, self.kernel_w)
        
        # calculate delta_error
        flattened_filter = self.filter.reshape(self.out_channels, self.in_channels, 1, -1)
        flattened_filter = cp.expand_dims(flattened_filter, axis = 0)

        flattened_error = cp.expand_dims(flattened_error, axis = 2)

        delta_error = cp.sum(flattened_error * flattened_filter, axis = 1)

        delta_error_base = cp.zeros_like(self.padded_input, dtype=cp.float64)

        cp.add.at(delta_error_base, (slice(None), slice(None), self.window_i, self.window_j), delta_error)
        
        if self.padding > 0:
            delta_error_base = delta_error_base[:,:,self.padding:-self.padding,self.padding:-self.padding]
        
        self.filter -= lr * cp.mean(delta_filter, axis=0)
        self.bias -= lr * cp.mean(error, axis = 0)
        
        return delta_error_base
        
class MaxPool2d(Module):
    def __init__(self,kernel_size, stride=1, padding=0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        
        if type(kernel_size) == int:
            self.kernel_h = self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
        
        if type(stride) == int:
            self.stride_h = self.stride_w = stride
        else:
            self.stride_h,self.stride_w = stride
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kernel_size=({self.kernel_h},{self.kernel_w}), stride={self.stride}, padding={self.padding})"
        
    def forward(self, x):
        batch_size, channel_size = x.shape[:2]
        output_h = (x.shape[2] - self.kernel_h + 2 * self.padding)//self.stride_h + 1
        output_w = (x.shape[3] - self.kernel_w + 2 * self.padding)//self.stride_w + 1
        
        self.window_i = cp.repeat(cp.arange(self.kernel_h), self.kernel_w).reshape(1,-1) + (cp.repeat(cp.arange(output_h), output_w)*self.stride_h).reshape(-1,1)
        self.window_j = cp.tile(cp.arange(self.kernel_w),self.kernel_h).reshape(1,-1) + (cp.tile(cp.arange(output_w),output_h)*self.stride_w).reshape(-1,1)

        self.padded_input = cp.pad(x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)))
        
        flattened = self.padded_input[:,:,self.window_i,self.window_j]
        self.flattened = flattened
        self.max_indices = cp.argmax(flattened, axis = -1)
        
        return cp.max(flattened, axis = -1).reshape(batch_size, channel_size, output_h, output_w)

    def backward_calc(self, error, lr):
        error = self.error_grad(error)
        batch_size, channel_size = error.shape[:2]
        max_sparse = cp.eye(self.kernel_h * self.kernel_w)[self.max_indices]
        flattened_error = error.reshape(batch_size, channel_size, -1, 1)
        max_vals = max_sparse * flattened_error
        
        delta_error = cp.zeros(self.padded_input.shape)
        cp.add.at(delta_error, (slice(None), slice(None), self.window_i, self.window_j), max_vals)
        
        if self.padding > 0:
            delta_error = delta_error[:,:,self.padding:-self.padding,self.padding:-self.padding]
        
        return delta_error
        
class LayerNorm(Module):
    def __init__(self, axis=-1):
        super().__init__()
        # flatten the input after the axis and normalize
        self.axis = axis
        self.gamma = None
        self.beta = None
        self.jacobian = None
        self.original_shape = None
    
    def forward(self,x):
        self.original_shape = x.shape
        reshaped_x = x.reshape(*self.original_shape[:self.axis],-1)
        
        if self.gamma is None:
            self.gamma = cp.ones((*reshaped_x.shape[:-1],1))
            self.beta = cp.zeros((*reshaped_x.shape[:-1],1))

        self.mean = cp.mean(reshaped_x, axis=-1, keepdims=True)
        self.std = cp.std(reshaped_x, axis=-1, keepdims=True)
        
        self.calculate_jacobian(reshaped_x)
        
        self.y = (x - self.mean) / (self.std + 1e-5)
        
        return self.gamma * self.y + self.beta
    
    def backward_calc(self, error, lr):
        delta_error = cp.matmul(error, self.jacobian).reshape(self.original_shape)
        
        delta_gamma = cp.sum(error * self.y, axis=-1)
        delta_beta = cp.sum(error, axis=-1)
        
        # TODO: verify if this is True
        self.gamma -= lr * cp.mean(delta_gamma, axis=0)
        self.beta -= lr * cp.mean(delta_beta, axis=0)
        
        return delta_error
    
    def calculate_jacobian(self, reshaped_x):
        # explenation
        # https://neuralthreads.medium.com/layer-normalization-and-how-to-compute-its-jacobian-for-backpropagation-55a549d5936f
        
        N = reshaped_x.shape[-1]
        I = cp.eye(N)
        
        first_part = (cp.tile(I, (*reshaped_x.shape[:-1],1,1)) - 1) / (N*self.std)
        
        x_1 = cp.expand_dims(reshaped_x - self.mean, -1) # (..., N, 1)
        x_2 = cp.expand_dims(reshaped_x - self.mean, -2) # (..., 1, N)
        
        second_part = cp.matmul(x_1, x_2) / (N*self.std**3)
        
        self.jacobian = self.gamma * (first_part - second_part)
    
if __name__ == "__main__":
    pass