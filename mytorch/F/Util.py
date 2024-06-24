import cupy as cp
from mytorch.Tensor import Tensor

def matmul(tensor1, tensor2):
    shape1 = tensor1.tensor.shape
    shape2 = tensor2.tensor.shape
    for prev in tensor1.prev:
        f = prev.error_grad
        prev.error_grad = lambda x: f(cp.matmul(x,tensor2.tensor.transpose(*shape2[:-2],shape2[-1],shape2[-2])))
    for prev in tensor2.prev:
        f = prev.error_grad
        prev.error_grad = lambda x: f(cp.matmul(tensor1.tensor.transpose(*shape1[:-2],shape1[-1],shape1[-2]),x))
    
    return Tensor(cp.matmul(tensor1.tensor, tensor2.tensor), prev=tensor1.prev.union(tensor2.prev))

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