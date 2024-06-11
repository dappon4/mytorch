import cupy as np
import matplotlib.pyplot as plt
from Functions import *

np.cuda.Device(0).use()

class Network:
    def __init__(self, size: list[int], activations, dropout_rate, lr=0.01, loss_func="mean_squared_error") -> None:
        self.lr = lr
        self.weights = []
        self.biases = []
        self.output = []
        self.activations = []
        self.activation_derivatives = []
        self.dropout_rate = dropout_rate
        self.dropout_rate.insert(0,0)
        
        for i in range(len(size)-1):
            self.weights.append(self.xavier_init(size[i],size[i+1]))
            self.biases.append(np.zeros((1,size[i+1])))
        
        for activation in activations:
            if activation == 'sigmoid':
                self.activations.append(sigmoid)
                self.activation_derivatives.append(sigmoid_derivative)
            elif activation == 'softmax':
                self.activations.append(softmax)
                self.activation_derivatives.append(softmax_derivative)
            elif activation == 'relu':
                self.activations.append(relu)
                self.activation_derivatives.append(relu_derivative)
            else:
                raise ValueError("Activation function not recognized")

        if loss_func == "mean_squared_error":
            self.loss = mean_squared_error
            self.loss_derivative = mean_squared_error_derivative
        elif loss_func == "cross_entropy":
            self.loss = cross_entropy
            self.loss_derivative = cross_entropy_derivative

    def xavier_init(self, input_size, output_size):
        return np.random.normal(0.0, np.sqrt(2/(input_size + output_size)), (input_size, output_size))
    
    def forward(self,x,training=True):
        # x shape: (batch_size, 1, input)
        self.output.append(x)
        for w, b, activation, dropout in zip(self.weights, self.biases, self.activations, self.dropout_rate):
            if training:
                dropout_mask = np.random.binomial(1, 1-dropout, size = x.shape)
                x = x * dropout_mask / (1-dropout)
            x = np.matmul(x,w) + b
            x = activation(x)

            self.output.append(x)
        return x
    
    def backward(self, error, layer):
        if layer == 0:
            return
        # derivative of the activation function
        error = error * self.activation_derivatives[layer - 1](self.output[layer])

        # error shape: (1, output_size)
        delta_weight = np.matmul(self.output[layer - 1].transpose(0,2,1), error)
        delta_weight = np.mean(delta_weight,axis = 0)
        
        delta_error = np.matmul(self.weights[layer - 1], error.transpose(0,2,1))

        # update weights and biases
        self.weights[layer - 1] -= self.lr * delta_weight
        self.biases[layer - 1] -= self.lr * np.mean(error, axis=0)
        
        # propagte to the previous layer
        self.backward(delta_error.transpose(0,2,1), layer - 1)
    
    def fit(self, X, y):
        self.output = []
        
        y_pred = self.forward(X)
        error = self.loss_derivative(y_pred, y)
        self.backward(error, len(self.weights))
        
        return self.loss(y_pred, y)
    
    def predict(self, X):
        return self.forward(X, training=False)
    
    # TODO: implement gradient clipping