## Introduction
This is my attempt to implement a neural network from ground up using numpy and cupy, the GPU version of numpy.  

This include basic functionalities such as forward pass and backward pass, backpropagation, and activation functions.  

I created a package named mytorch, hwich ocntains all the functions and classes required to build a neural network in Pytoch style.

## Layers
 - Linear(input_size, output_size)
 - Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
 - MaxPool2d(kernel_size, stride=1, padding=0)
 - LayerNorm(axis=-1)

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
Here is an example of how to use mytorch package to build a simple neural network. in the same style as Pytorch.    
```python
from mytorch.F.Activation import relu
from mytorch.nn.Module import CompoundModule, Linear

class SimpleFeedForward(CompoundModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        x = relu(x)
        
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
from mytorch.Util.Trainer import Trainer, load_mnist
from mytorch.F.Evaluator import cross_entropy

X, y = load_mnist(flatten=False)
trainer = Trainer(X, y, batch=64, epochs=10, lr=0.001, test_size=0.2, validation_size=0.2, loss_func=cross_entropy)

trainer.train(network)
```
You can also visualize the trianing process using these methods:
```python
trainer.accuracy(network)
trainer.visualize_loss()
```

## What's next?
I plan to re-implement this using JAX, once I get my hands on a Linux machine.