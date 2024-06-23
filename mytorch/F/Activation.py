import cupy as cp

def relu(tensor):
    fil = cp.where(tensor.tensor > 0, 1, 0)
    for prev in tensor.prev:

        f = prev.error_grad
        prev.error_grad = lambda x: f(x * fil)

    tensor.tensor *= fil
    return tensor

def sigmoid(tensor):
    tensor.tensor = 1/(1+cp.exp(-tensor.tensor))
    for prev in tensor.prev:
        f = prev.error_grad
        prev.error_grad = lambda x: f(x * tensor.tensor * (1 - tensor.tensor))
    
    return tensor

def softmax(tensor, axis=-1):
    x = tensor.tensor
    x = cp.exp(x - cp.max(x, axis=axis, keepdims=True))
    tensor.tensor = x / cp.sum(x, axis=-axis, keepdims=True)
    
    return tensor