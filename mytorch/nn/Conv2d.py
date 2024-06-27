from mytorch.F.Initializer import he_init_conv2d
from mytorch.nn.Module import Module

import cupy as cp

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