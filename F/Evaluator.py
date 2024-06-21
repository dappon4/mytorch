import cupy as cp

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