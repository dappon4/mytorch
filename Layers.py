from typing import Any
from Functions import *
import cupy as cp
import time

class Layer:
    def __init__(self):
        self.prev = None
        self.training = True
        
    def __call__(self, inp):
        x, prev = inp
        if not self.prev:
            self.prev = prev
        x = self.forward(x)
        
        return x, self
    
    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError
    
    def train(self):
        self.training = True
        if self.prev:
            self.prev.train()
    
    def eval(self):
        self.training = False
        if self.prev:
            self.prev.eval()

class CompoundLayer(Layer):
    def __init__(self):
        super().__init__()
        self.final_layer = None
    
    def __call__(self, inp):
        _, prev = inp
        if not self.prev:
            self.prev = prev
        x = self.forward(inp)
        if not self.final_layer:
            _, self.final_layer = x
        
        # x is (cp array, self)
        return x
    
    def backward(self, error):
        self.final_layer.backward(error)
    
    def train(self):
        if self.final_layer:
            self.final_layer.train()
    
    def eval(self):
        if self.final_layer:
            self.final_layer.eval()
    
    def predict(self, X):
        y_pred, _ = self((X,None))
        return y_pred
    
class Linear(Layer):
    def __init__(self, input_size, output_size, lr = 0.005, dropout=0.0):
        super().__init__()
        self.lr = lr
        self.weight = xavier_init(input_size, output_size)
        self.bias = cp.zeros((1, output_size))
        
        self.dropout = dropout
        
    def forward(self, x):
        # x is cp array
        if self.training:
            dropout_mask = cp.random.binomial(1, 1-self.dropout, size = x.shape)
            x = x * dropout_mask / (1-self.dropout)

        self.input = x
        x = cp.matmul(x, self.weight) + self.bias
        return x

    def backward(self, error):
        delta_weight = cp.matmul(self.input.transpose(0,2,1), error)
        delta_error = cp.matmul(self.weight, error.transpose(0,2,1))
        
        self.weight -= self.lr * cp.mean(delta_weight, axis = 0)
        self.bias -= self.lr * cp.mean(error, axis=0)
        
        if self.prev:
            self.prev.backward(delta_error.transpose(0,2,1))

class Residual(Layer):
    def __init__(self):
        super().__init__()
        self.prev = []
    
    def __call__(self, inp1, inp2):
        x1, prev1 = inp1
        x2, prev2 = inp2
        if not self.prev:
            self.prev.append(prev1)
            self.prev.append(prev2)
        x = self.forward(x1, x2)
        
        return x, self
    
    def forward(self, x1, x2):
        return x1 + x2
    
    def backward(self, error):
        for prev in self.prev:
            prev.backward(error)
    
    def train(self):
        for prev in self.prev:
            prev.train()
    
    def eval(self):
        for prev in self.prev:
            prev.eval()

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, lr=0.005):
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
        self.lr = lr
        
        self.filter = he_init_conv2d((out_channels, in_channels, *self.kernel_size))
        self.bias = None
    
    def forward(self, x):
        #x is cp array of shape (batch_size, in_channels, height, width)
        # output cp array (batch size, out_channels, new_height, new_width)
        #print((x.shape[0], self.out_channels, x.shape[2] - self.kernel_size[0] + 1, x.shape[3] - self.kernel_size[1] + 1))
        batch_size = x.shape[0]
        if self.bias is None:
            self.bias = cp.zeros((1, self.out_channels, int((x.shape[2] - self.kernel_size[0] + 2 * self.padding)/self.stride[0] + 1), int((x.shape[3] - self.kernel_size[1] + 2 * self.padding)/self.stride[1] + 1)))
        
        out_height = int((x.shape[2] - self.kernel_size[0] + 2 * self.padding)/self.stride[0] + 1)
        out_width = int((x.shape[3] - self.kernel_size[1] + 2 * self.padding)/self.stride[1] + 1)
        
        out = cp.zeros((batch_size, self.out_channels, out_height, out_width))
        self.padded_input = cp.pad(x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)))
        
        for j in range(0, out_height): # height dimension
            for k in range(0, out_width): # width dimension
                start_height = j * self.stride[0]
                start_width = k * self.stride[1]
                cropped = self.padded_input[:, :, start_height:start_height+self.kernel_size[0], start_width:start_width+self.kernel_size[1]]
                # cropped is shape (in channels, kernel height, kernel width)
                out[:, :, j, k] = cp.tensordot(cropped, self.filter, axes=([1,2,3], [1,2,3]))

        out += self.bias
        
        return out,self
    
    def backward(self, error):
        # error is cp array of shape (batch_size, out_channels, new_height, new_width)
        batch_size = error.shape[0]
        
        delta_filter = cp.zeros((batch_size,self.out_channels, self.in_channels, *self.kernel_size))
        delta_error = cp.zeros((batch_size, self.in_channels, self.padded_input.shape[2], self.padded_input.shape[3]))
        
        for j in range(0, error.shape[2]): # height dimension
            for k in range(0, error.shape[3]): # width dimension
                delta_filter += error[:,:,j,k].reshape(batch_size,self.out_channels,1,1,1) * self.padded_input[:,:,j*self.stride[0]:j*self.stride[0]+self.kernel_size[0], k*self.stride[1]:k*self.stride[1]+self.kernel_size[1]].reshape(batch_size,1,self.in_channels,self.kernel_size[0],self.kernel_size[1])
                delta_error[:,:,j*self.stride[0]:j*self.stride[0]+self.kernel_size[0],k*self.stride[1]:k*self.stride[1]+self.kernel_size[1]] += cp.sum(error[:,:,j,k].reshape(batch_size, self.out_channels,1,1,1) * self.filter, axis=1)
    
        self.filter -= self.lr * cp.mean(delta_filter, axis=0)
        self.bias -= self.lr * cp.mean(error, axis=0)
        
        return delta_error[:,:,self.padding:-self.padding,self.padding:-self.padding]

# TODO: Implement MaxPool2d
# TODO: Implement Flatten
# TODO: IMplement BatchNorm2d

class Relu(Layer):
    def __init__(self):
        super().__init__()
        self.input = None
    
    def forward(self, x):
        self.input = x
        return cp.maximum(x, 0)

    def backward(self, error):
        if self.prev:
            self.prev.backward(error * cp.where(self.input > 0, 1, 0))

if __name__ == "__main__":
    network = Conv2d(32, 64, (2,2), padding = 1)
    
    x = cp.random.randn(30,32,28,28)
    x,_ = network.forward(x)
    t1 = time.time()
    x = network.backward(x)
    t2 = time.time()
    print(t2-t1)
    print(x.shape)