import cupy as cp
from cupy.lib.stride_tricks import as_strided
from Tensor import Tensor

def softmax_derivative(x):
    return x

def mean_squared_error(y_pred, y_true):
    #print(cp.mean(cp.mean(cp.power(y_pred - y_true,2),axis=-1)))
    return cp.mean(cp.sum(cp.power(y_pred - y_true,2),axis=-1)).item()

def mean_squared_error_derivative(y_pred, y_true):
    #print(cp.mean(y_pred - y_true, axis = 0).shape)
    return 2 * (y_pred - y_true)

def cross_entropy(y_pred, y_true):
    # cross entropy and softmax all together
    #y_pred = softmax(y_pred)
    y_pred = cp.exp(y_pred - cp.max(y_pred))
    y_pred = y_pred / cp.sum(y_pred, axis=-1, keepdims=True)
    return (-cp.sum(y_true * cp.log(y_pred)) / len(y_pred)).item()

def cross_entropy_derivative(y_pred, y_true):
    #print(cp.mean(y_pred - y_true, axis=0))
    y_pred = cp.exp(y_pred - cp.max(y_pred))
    y_pred = y_pred / cp.sum(y_pred, axis=-1, keepdims=True)
    return y_pred - y_true

def xavier_init(input_size, output_size):
    return cp.random.normal(0.0, cp.sqrt(2/(input_size + output_size)), (input_size, output_size))

def he_init_conv2d(filter_shape):
    # filter shape (output_depth, input_depth, filter_height, filter_width)
    fan_in = filter_shape[1] * filter_shape[2] * filter_shape[3] # number of input units, product of: input depth, filter height, filter width
    stddev = cp.sqrt(2.0 / fan_in) # standard deviation of normal distribution

    return cp.random.normal(loc=0, scale=stddev, size=filter_shape)

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

def softmax(tensor):
    x = tensor.tensor
    x = cp.exp(x - cp.max(x))
    tensor.tensor = x / cp.sum(x, axis=-1, keepdims=True)
    
    return tensor

def matmul(tensor1, tensor2):
    # assume the sahpes are length 3
    for prev in tensor1.prev:
        f = prev.error_grad
        prev.error_grad = lambda x: f(cp.einsum("ijk,ilk->ijl",x,tensor2.tensor))
    for prev in tensor2.prev:
        f = prev.error_grad
        prev.error_grad = lambda x: f(cp.einsum("ijk,ijl->ikl",tensor1.tensor,x))
    
    return Tensor(cp.einsum("ijk,ikl->ijl", tensor1.tensor, tensor2.tensor), prev=tensor1.prev.union(tensor2.prev))

def flatten(tensor):
    shape = tensor.tensor.shape
    for prev in tensor.prev:
        f = prev.error_grad
        prev.error_grad = lambda x: f(x.reshape(shape))

    tensor.tensor = tensor.tensor.reshape((shape[0], -1))
    return tensor

def dropout(tensor, p):
    mask = cp.random.binomial(1, 1-p, tensor.tensor.shape)
    for prev in tensor.prev: break
    if prev.training:
        for prev in tensor.prev:
            f = prev.error_grad
            prev.error_grad = lambda x: f(x * mask)
    
        tensor.tensor *= mask
    return tensor

if __name__ == "__main__":
    pass