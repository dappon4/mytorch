import cupy as np

def identity(x):
    return x

def identity_derivative(x):
    return x

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def softmax(x):
    # shape of x: (batch_size, 1, output_size)
    # calculate softmax for each row
    x = np.exp(x - np.max(x))
    return x / np.sum(x, axis=-1, keepdims=True)

def softmax_derivative(x):
    return x

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def mean_squared_error(y_pred, y_true):
    #print(np.mean(np.mean(np.power(y_pred - y_true,2),axis=-1)))
    return np.mean(np.sum(np.power(y_pred - y_true,2),axis=-1)).item()

def mean_squared_error_derivative(y_pred, y_true):
    #print(np.mean(y_pred - y_true, axis = 0).shape)
    return 2 * (y_pred - y_true)

def cross_entropy(y_pred, y_true):
    # cross entropy and softmax all together
    y_pred = softmax(y_pred)
    return (-np.sum(y_true * np.log(y_pred)) / len(y_pred)).item()

def cross_entropy_derivative(y_pred, y_true):
    #print(np.mean(y_pred - y_true, axis=0))
    y_pred = softmax(y_pred)
    return y_pred - y_true