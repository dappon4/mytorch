import cupy as cp

def mul_backward(num):
    return lambda x: num*x

def div_backward(num):
    return lambda x: x/num

def transpose_backward(dimension):
    counter_index = [0]*len(dimension)
    for i, dim in enumerate(dimension):
        counter_index[dim] = i
    
    counter_index = tuple(counter_index)
    return lambda x: x.transpose(*counter_index)

def matmul_backward(tensor1, tensor2):
    shape1 = tensor1.tensor.shape
    shape2 = tensor2.tensor.shape
    
    tensor1_T = tensor1.transpose(shape1[:-2], shape1[-1], shape1[-2])
    tensor2_T = tensor2.transpose(shape2[:-2], shape2[-1], shape2[-2])
    
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