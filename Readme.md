## Introduction
This is my attempt to implement a neural network from ground up using numpy and cupy, the GPU version of numpy.  
This includes the basic functionality such as forward pass and backward pass, backpropagation, and activation functions.

## Layers
 - Linear(input_size, output_size)
 - Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
 - MaxPool2d(kernel_size, stride=1, padding=0)

## Activation Functions
 - relu
 - sigmoid
 - softmax

## Initializers
 - Xavier Initializer
 - He Initializer

## General Functions
 - flatten
 - dropout

## About Tensor class
The Tensor class is a wrapper for my numpy arrays.  
It retains information about the previous Layer, as it goes through the network.

## Usage
```python
from Functions import *
from Layers import CompoundLayer, Linear, Conv2d, MaxPool2d
from Trainer import Trainer, load_mnist

class CNN(CompoundLayer):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Conv2d(1, 16, 3, 1, 1)
        self.conv2 = Conv2d(16, 32, 3, 1, 1)
        
        self.pool1 = MaxPool2d(2, 2)
        self.pool2 = MaxPool2d(2, 2)
        
        self.linear1 = Linear(32 * 7 * 7, 128)
        self.linear2 = Linear(128, 10)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = self.pool1(x)
        x = relu(self.conv2(x))
        x = flatten(self.pool2(x))

        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        return x
```

## How to train a model using Trainer
I implemented Trainer class, which can be used to train a model.  
Trainer arguments:
 - X: numpy array, input value
 - y: numpy array, target value
 - batch: int, batch size
 - epochs: int, number of epochs
 - lr: float, learning rate
 - test_size: float, test size
- validation_size: float, validation size
```python
from Trainer import Trainer, load_mnist

X, y = load_mnist(flatten=False)
trainer = Trainer(X, y, batch=256, epochs=5, lr = 0.005, test_size=0.2, validation_size=0.1, loss_func="cross_entropy")

trainer.train(network)
```
You can also visualize the trianing process using these methods:
```python
trainer.accuracy(network)
trainer.visualize_loss()
```

## What's next?
I plan to re-implement this using JAX, once I get my hands on a Linux machine.