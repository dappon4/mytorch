from typing import Any
import cupy as cp

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
    
    def xavier_init(self, input_size, output_size):
        return cp.random.normal(0.0, cp.sqrt(2/(input_size + output_size)), (input_size, output_size))

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
        self.weight = self.xavier_init(input_size, output_size)
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