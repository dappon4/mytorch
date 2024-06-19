import cupy as cp
from cupy.lib.stride_tricks import sliding_window_view, as_strided
from Tensor import Tensor

def softmax(x):
    x = cp.exp(x - cp.max(x))
    return x / cp.sum(x, axis=-1, keepdims=True)

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
    y_pred = softmax(y_pred)
    return (-cp.sum(y_true * cp.log(y_pred)) / len(y_pred)).item()

def cross_entropy_derivative(y_pred, y_true):
    #print(cp.mean(y_pred - y_true, axis=0))
    y_pred = softmax(y_pred)
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


def sliding_window_view_with_strides(matrix, sub_shape, stride):
    batch_size, in_channels, height, width = matrix.shape
    sub_rows, sub_cols = sub_shape

    # Calculate the shape and strides for the submatrices
    out_height = (height - sub_rows) // stride[0] + 1
    out_width = (width - sub_cols) // stride[1] + 1
    shape = (batch_size, in_channels, out_height, out_width, sub_rows, sub_cols)
    new_strides = (matrix.strides[0],matrix.strides[1],matrix.strides[2] * stride[0],matrix.strides[3] * stride[1],matrix.strides[2],matrix.strides[3])

    sub_matrices = as_strided(matrix, shape=shape, strides=new_strides)
    return sub_matrices

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
    
    tensor.tensor.reshape((shape[0], -1))
    return tensor

if __name__ == "__main__":
    x = cp.zeros((2,3,4,4))
    y = sliding_window_view_with_strides(x, (2,2), (2,2))
    z = cp.ones(y.shape)
    y += z
    print(x)
    print(y.shape)