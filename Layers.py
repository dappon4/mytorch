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
        # TODO: account for multiple previous layers
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
        self.error_grad = lambda x: x
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
    def __init__(self, input_size, output_size, dropout=0.0):
        super().__init__()
        self.weight = xavier_init(input_size, output_size)
        self.bias = cp.zeros((output_size,))
        
        self.dropout = dropout
        
    def forward(self, x):
        # x is cp array
        if self.training:
            dropout_mask = cp.random.binomial(1, 1-self.dropout, size = x.shape)
            x = x * dropout_mask / (1-self.dropout)

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
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
        if type(stride) == int:
            self.stride = (stride, stride)
        else:
            self.stride = stride

        self.padding = padding
        
        self.filter = he_init_conv2d((out_channels, in_channels, *self.kernel_size))
        self.bias = None
    
    
    def forward(self, x):
        # output size: [(W−K+2P)/S]+1
        batch_size = x.shape[0]
        output_height = (x.shape[2] - self.kernel_size[0] + 2 * self.padding)//self.stride[0] + 1
        output_width = (x.shape[3] - self.kernel_size[1] + 2 * self.padding)//self.stride[1] + 1
        
        if self.bias is None:
            self.bias = cp.zeros((1, self.out_channels, output_height, output_width))
        
        self.padded_input = cp.pad(x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)))
        #sub_matrices = sliding_window_view(self.padded_input, (batch_size, self.in_channels, *self.kernel_size))
        sub_matrices = sliding_window_view_with_strides(self.padded_input, self.kernel_size, self.stride)
        flattend = sub_matrices.reshape(batch_size, self.in_channels, output_height*output_width, self.kernel_size[0]*self.kernel_size[1])
        #flattend = sub_matrices.reshape((-1, batch_size, self.in_channels, self.kernel_size[0]*self.kernel_size[1])).transpose(1,2,0,3)
        
        # flattend is shape (batch_size, in_channels, out_height*out_width, kernel_size[0]*kernel_size[1])
        flattend_filter = self.filter.reshape((self.out_channels, self.in_channels, 1, self.kernel_size[0]*self.kernel_size[1]))
        
        return cp.tensordot(flattend, flattend_filter, axes=([1,3], [1,3])).transpose(0,2,1,3).reshape(batch_size, self.out_channels, output_height, output_width) + self.bias

    # TODO: optimize this
    def backward_calc(self, error, lr):
        error = self.error_grad(error)
        
        # error is cp array of shape (batch_size, out_channels, new_height, new_width)
        batch_size = error.shape[0]
        
        delta_filter = cp.zeros((batch_size,self.out_channels, self.in_channels, *self.kernel_size))
        delta_error = cp.zeros((batch_size, self.in_channels, self.padded_input.shape[2], self.padded_input.shape[3]))
        
        for j in range(0, error.shape[2]): # height dimension
            for k in range(0, error.shape[3]): # width dimension
                delta_filter += error[:,:,j,k].reshape(batch_size,self.out_channels,1,1,1) * self.padded_input[:,:,j*self.stride[0]:j*self.stride[0]+self.kernel_size[0], k*self.stride[1]:k*self.stride[1]+self.kernel_size[1]].reshape(batch_size,1,self.in_channels,self.kernel_size[0],self.kernel_size[1])
                delta_error[:,:,j*self.stride[0]:j*self.stride[0]+self.kernel_size[0],k*self.stride[1]:k*self.stride[1]+self.kernel_size[1]] += cp.sum(error[:,:,j,k].reshape(batch_size, self.out_channels,1,1,1) * self.filter, axis=1)
        #print(delta_error)
        
        self.filter -= lr * cp.mean(delta_filter, axis=0)
        self.bias -= lr * cp.mean(error, axis=0)
        
        if self.padding > 0:
            delta_error = delta_error[:,:,self.padding:-self.padding,self.padding:-self.padding]
        
        return delta_error
    
    def new_backward(self, error, lr):
        error = error * self.error_grad
        
        # error is cp array of shape (batch_size, out_channels, new_height, new_width)
        batch_size = self.padded_input.shape[0]
        channel_size = self.padded_input.shape[1]
        
        error_size = error.shape[2] * error.shape[3]
        
        input_sub_matrices = sliding_window_view_with_strides(self.padded_input, self.kernel_size, self.stride).reshape(batch_size, channel_size, -1, self.kernel_size[0], self.kernel_size[1])
        flattened_error = error.reshape(batch_size, error.size[1], -1, 1).repeat(self.kernel_size[0]*self.kernel_size[1], axis = -1).reshape(batch_size, error.size[1], -1, self.kernel_size[0],self.kernel_size[1])
        delta_filter_flattened = input_sub_matrices * flattened_error
        

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
    pass