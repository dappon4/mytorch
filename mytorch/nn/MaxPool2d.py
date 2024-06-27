from mytorch.nn.Module import Module
import cupy as cp

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
        
        self.window_i = cp.repeat(cp.arange(self.kernel_h, dtype=cp.float32), self.kernel_w).reshape(1,-1) + (cp.repeat(cp.arange(output_h, dtype=cp.float32), output_w)*self.stride_h).reshape(-1,1)
        self.window_j = cp.tile(cp.arange(self.kernel_w, dtype=cp.float32),self.kernel_h).reshape(1,-1) + (cp.tile(cp.arange(output_w, dtype=cp.float32),output_h)*self.stride_w).reshape(-1,1)

        self.padded_input = cp.pad(x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)))
        
        flattened = self.padded_input[:,:,self.window_i,self.window_j]
        self.flattened = flattened
        self.max_indices = cp.argmax(flattened, axis = -1)
        
        return cp.max(flattened, axis = -1, dtype=cp.float32).reshape(batch_size, channel_size, output_h, output_w)

    def backward_calc(self, error, lr):

        batch_size, channel_size = error.shape[:2]
        max_sparse = cp.eye(self.kernel_h * self.kernel_w)[self.max_indices]
        flattened_error = error.reshape(batch_size, channel_size, -1, 1)
        max_vals = max_sparse * flattened_error
        
        delta_error = cp.zeros(self.padded_input.shape, dtype=cp.float32)
        cp.add.at(delta_error, (slice(None), slice(None), self.window_i, self.window_j), max_vals)
        
        if self.padding > 0:
            delta_error = delta_error[:,:,self.padding:-self.padding,self.padding:-self.padding]
        
        return delta_error