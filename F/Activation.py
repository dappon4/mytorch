import cupy as cp
from mytorch.nn import Intermediate
from mytorch.F.Gradient import relu_backward, sigmoid_backward, softmax_backward, tanh_backward
from mytorch.Tensor import Tensor

def relu(tensor):
    fil = cp.where(tensor.tensor > 0, 1, 0)

    return Tensor((tensor.tensor * fil).astype(cp.float32), {Intermediate(tensor.prev, relu_backward(fil))})

def leaky_relu(tensor, alpha=0.01):
    fil = cp.where(tensor.tensor > 0, 1, alpha)

    return Tensor((tensor.tensor * fil).astype(cp.float32), {Intermediate(tensor.prev, relu_backward(fil))})

def sigmoid(tensor):
    out = 1/(1+cp.exp(-tensor.tensor))

    return Tensor(out, {Intermediate(tensor.prev, sigmoid_backward(out))})

def softmax(tensor, axis=-1):
    x = tensor.tensor.copy()
    x = cp.exp(x - cp.max(x, axis=axis, keepdims=True))
    tensor.tensor = x / cp.sum(x, axis=-axis, keepdims=True)
    
    return Tensor(x / cp.sum(x, axis=-axis, keepdims=True), {Intermediate(tensor.prev, softmax_backward())})

def tanh(tensor):
    out = cp.tanh(tensor.tensor)

    return Tensor(out, {Intermediate(tensor.prev, tanh_backward(out))})