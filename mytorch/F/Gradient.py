import cupy as cp

def mul_backward(num):
    return lambda x: num*x

def div_backward(num):
    return lambda x: x/num

def transpose_backward(*axis):
    counter_index = [0]*len(axis)
    for i, dim in enumerate(axis):
        counter_index[dim] = i
    
    counter_index = tuple(counter_index)
    return lambda x: x.transpose(*counter_index)

def matmul_backward(tensor1, tensor2):
    trasnpose_index = list(range(len(tensor1.shape)))
    trasnpose_index[-1], trasnpose_index[-2] = trasnpose_index[-2], trasnpose_index[-1]
    
    tensor1_T = tensor1.transpose(*trasnpose_index)
    tensor2_T = tensor2.transpose(*trasnpose_index)
    
    return lambda x: (cp.matmul(x, tensor2_T), cp.matmul(tensor1_T, x))

def reshape_backward(curr_shape):
    return lambda x: x.reshape(*curr_shape)

def flatten_backward(shape):
    return lambda x: x.reshape(shape)

def dropout_backward(mask):
    return lambda x: x * mask

def relu_backward(fil):
    return lambda x: x * fil

def sigmoid_backward(output):
    return lambda x: x * output * (1 - output)

def softmax_backward():
    return lambda x: x