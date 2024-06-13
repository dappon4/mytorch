import cupy as cp

def identity(x):
    return x

def identity_derivative(x):
    return x

def sigmoid(x):
    return 1/(1+cp.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def softmax(x):
    # shape of x: (batch_size, 1, output_size)
    # calculate softmax for each row
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