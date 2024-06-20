from typing import Any
from Functions import *
from Tensor import Tensor
import cupy as cp
from cupy.lib.stride_tricks import sliding_window_view, as_strided
import time

class Layer:
    def __init__(self):
        self.prev = set()
        self.training = True
        
        # TODO: initialize error_grad properly
        self.error_grad = lambda x: x
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
        
    def __call__(self, tensor):
        self.error_grad = lambda x: x
        self.prev = tensor.prev
        
        tensor = Tensor(self.forward(tensor.tensor), {self})
        
        return tensor
    
    def forward(self, x):
        raise NotImplementedError

    def backward_calc(self, error, lr):
        raise NotImplementedError
    
    def backward(self, error, lr):
        #print(self)
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

class CompoundLayer(Layer):
    # TODO: connect the previous layer to the first layer
    def __init__(self):
        super().__init__()
        self.final_layer = set()
    
    def __call__(self, tensor):
        self.error_grad = lambda x: x # is this needed?
        self.prev = tensor.prev
            
        tensor = self.forward(tensor)
        self.final_layer = tensor.prev
        
        return tensor

    def backward(self, error, lr):
        for layer in self.final_layer:
            layer.backward(error, lr)
    
    def train(self):
        for layer in self.final_layer:
            layer.train()
    
    def eval(self):
        for layer in self.final_layer:
            layer.eval()
    
class Linear(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = xavier_init(input_size, output_size)
        self.bias = cp.zeros((output_size,))
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_size={self.weight.shape[0]}, output_size={self.weight.shape[1]})"
    
    def forward(self, x):
        # x is cp array

        self.input = x
        #print(x.shape, self.weight.shape, self.bias.shape)
        x = cp.matmul(x, self.weight) + self.bias
        return x

    def backward_calc(self, error, lr):
        error = self.error_grad(error)
        #print(self.input.shape, error.shape)
        delta_weight = cp.einsum("ij,ik->ijk", self.input, error)
        #print(delta_weight.shape)
        delta_error = cp.matmul(self.weight, error.T)
        
        #print(delta_error.shape)
        
        self.weight -= lr * cp.mean(delta_weight, axis = 0)
        self.bias -= lr * cp.mean(error, axis=0)
        
        return delta_error.T        

class Conv2d(Layer):
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
        
        flattend = self.padded_input[:,:,self.window_i,self.window_j]
        flattend_filter = self.filter.reshape((self.out_channels, self.in_channels, 1, self.kernel_h*self.kernel_w))
        
        # maybe optimze this
        return cp.tensordot(flattend, flattend_filter, axes=([1,3], [1,3])).transpose(0,2,1,3).reshape(batch_size, self.out_channels, output_h, output_w) + self.bias
    
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
        
        flattend_error = error.reshape(batch_size, self.out_channels, -1)
        flattend_error = cp.expand_dims(flattend_error, axis = -1)
        
        delta_filter = cp.sum(sub_matrices * flattend_error,axis = 2).reshape(batch_size, self.out_channels, self.in_channels, self.kernel_h, self.kernel_w)
        
        # calculate delta_error
        flattened_filter = self.filter.reshape(self.out_channels, self.in_channels, 1, -1)
        flattened_filter = cp.expand_dims(flattened_filter, axis = 0)

        flattend_error = cp.expand_dims(flattend_error, axis = 2)

        delta_error = cp.sum(flattend_error * flattened_filter, axis = 1)

        delta_error_base = cp.zeros_like(self.padded_input, dtype=cp.float64)

        cp.add.at(delta_error_base, (slice(None), slice(None), self.window_i, self.window_j), delta_error)
        
        if self.padding > 0:
            delta_error_base = delta_error_base[:,:,self.padding:-self.padding,self.padding:-self.padding]
        
        self.filter -= lr * cp.mean(delta_filter, axis=0)
        self.bias -= lr * cp.mean(error, axis = 0)
        
        return delta_error_base
        
class MaxPool2d(Layer):
    def __init__(self,kernel_size, stride=1, padding=0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        
        if type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
        if type(stride) == int:
            self.stride = (stride, stride)
        else:
            self.stride = stride
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
    
    def forward(self,x):
        batch_size, channel_size = x.shape[:2]
        
        output_height = (x.shape[2] - self.kernel_size[0] + 2 * self.padding)//self.stride[0] + 1
        output_width = (x.shape[3] - self.kernel_size[1] + 2 * self.padding)//self.stride[1] + 1
        
        self.padded_input = cp.pad(x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode="constant", constant_values = -10000)
        #sub_matrices = sliding_window_view(self.padded_input, (batch_size, channel_size, *self.kernel_size))
        #self.flattened = sub_matrices.reshape((-1, batch_size, channel_size, self.kernel_size[0]*self.kernel_size[1])).transpose(1,2,0,3)
        sub_matrices_2 = sliding_window_view_with_strides(self.padded_input, self.kernel_size, self.stride)
        self.flattened = sub_matrices_2.reshape(batch_size, channel_size, output_height*output_width, self.kernel_size[0]*self.kernel_size[1])
        
        return cp.max(self.flattened, axis = -1).reshape(batch_size, channel_size, output_height, output_width)
        
    
    def backward_calc(self, error, lr):
        error = self.error_grad(error)
        
        delta_error = cp.zeros(self.padded_input.shape)
        
        batch_size = error.shape[0]
        channel_size = error.shape[1]
        error_height = error.shape[2]
        error_width = error.shape[3]
        
        max_values = cp.max(self.flattened, axis = -1)
        max_values = cp.expand_dims(max_values, axis = -1)
        bool_matrix = (self.flattened == max_values).astype(int)
        
        #print(bool_matrix)
        
        flattend_error = error.reshape(batch_size, channel_size, -1,1)
        sub_matrix_error = bool_matrix * flattend_error
        
        
        
        sub_matrix_error = sub_matrix_error.reshape(batch_size, channel_size, error_height, error_width, self.kernel_size[0], self.kernel_size[1])
        #print(sub_matrix_error)
        for i in range(0, error_height):
            for j in range(0, error_width):
                delta_error[:,:,i*self.stride[0]:i*self.stride[0]+self.kernel_size[0],j*self.stride[1]:j*self.stride[1]+self.kernel_size[1]] += sub_matrix_error[:,:,i,j]
        
        if self.padding > 0:
            delta_error = delta_error[:,:,self.padding:-self.padding,self.padding:-self.padding]
        
        return delta_error

if __name__ == "__main__":
    cnn = Conv2d(3,5,2)
    err = cnn.forward(cp.arange(10*3*4*4).reshape(10,3,4,4))
    a = cnn.backward_calc(err, 0.01)
    b = cnn.new_backward(err, 0.01)
    print(a[0])
    print("b")
    print(b[0])