## Introduction
This is my attempt to implement a neural network from ground up using numpy and cupy, the GPU version of numpy.  
This includes the basic functionality such as forward pass and backward pass, backpropagation, and activation functions.

## Layers
 - Linear(input_size, output_size, dropout=0.0)
 - Residual(x1, x2)
 - Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
 - MaxPool2d(kernel_size, stride=0, padding=0)
 - Flatten(x)

## Usage
```python
from Functions import *
from Layers import CompoundLayer, Linear, Conv2d, MaxPool2d, Flatten

class Network(CompoundLayer):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Conv2d(1, 16, 3, 1, 1)
        self.conv2 = Conv2d(16, 32, 3, 1, 1)
        
        self.pool1 = MaxPool2d(2, 2)
        self.pool2 = MaxPool2d(2, 2)
        
        self.flatten = Flatten()
        
        self.linear1 = Linear(32 * 7 * 7, 128)
        self.linear2 = Linear(128, 10)
        
    def forward(self, x):
        x = relu(self.conv1(x))
        x = self.pool1(x)
        x = relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        return x
```