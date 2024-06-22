import cupy as cp
import Tensor

class Loss:
    def __init__(self, value, gradient, prev):
        self.value = value
        self.prev = prev
        self.gradient = gradient
    
    def backward(self, lr):
        for prev in self.prev:
            prev.backward(self.gradient, lr)

def mean_squared_error(y_pred_tensor, y_true):
    y_pred = y_pred_tensor.tensor
    value = cp.mean(cp.sum(cp.power(y_pred - y_true,2),axis=-1)).item()
    gradient = 2 * (y_pred - y_true)
    
    return Loss(value, gradient, y_pred_tensor.prev)

def cross_entropy(y_pred_tensor, y_true):
    # cross entropy and softmax all together
    #y_pred = softmax(y_pred)
    y_pred = y_pred_tensor.tensor
    
    y_pred = cp.exp(y_pred - cp.max(y_pred))
    y_pred = y_pred / cp.sum(y_pred, axis=-1, keepdims=True)
    
    value = (-cp.sum(y_true * cp.log(y_pred)) / len(y_pred)).item()
    gradient = y_pred - y_true
    
    return Loss(value, gradient, y_pred_tensor.prev)