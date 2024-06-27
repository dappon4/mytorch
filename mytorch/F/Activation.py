import cupy as cp
from mytorch.nn import Intermediate
from mytorch.F.Gradient import relu_backward, sigmoid_backward, softmax_backward
from mytorch.Tensor import Tensor

def relu(tensor):
    fil = cp.where(tensor.tensor > 0, 1, 0)

    return Tensor(tensor.tensor * fil, Intermediate(tensor.prev, relu_backward(fil)))

def sigmoid(tensor):
    out = 1/(1+cp.exp(-tensor.tensor))

    return Tensor(out, Intermediate(tensor.prev, sigmoid_backward(out)))

def softmax(tensor, axis=-1):
    x = tensor.tensor.copy()
    x = cp.exp(x - cp.max(x, axis=axis, keepdims=True))
    tensor.tensor = x / cp.sum(x, axis=-axis, keepdims=True)
    
    return Tensor(x / cp.sum(x, axis=-axis, keepdims=True), softmax_backward())