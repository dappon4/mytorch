from mytorch.nn.Module import Module
import cupy as cp

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
        
        self.jacobian = self.calculate_jacobian(reshaped_x)
        
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
        
        first_part = (cp.tile(I, (*reshaped_x.shape[:-1],1,1)) - 1) / (N*self.std[...,cp.newaxis])
        
        x_1 = cp.expand_dims(reshaped_x - self.mean, -1) # (..., N, 1)
        x_2 = cp.expand_dims(reshaped_x - self.mean, -2) # (..., 1, N)
        
        second_part = cp.matmul(x_1, x_2) / (N*(self.std[...,cp.newaxis]**3))
        
        return self.gamma[...,cp.newaxis] * (first_part - second_part)