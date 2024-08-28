import cupy as cp
from mytorch.Tensor import Tensor
from mytorch.nn import Intermediate, IntermediateSplit
from mytorch.F.Gradient import dropout_backward, matmul_backward

def matmul(tensor1, tensor2):

    return Tensor(cp.matmul(tensor1.tensor, tensor2.tensor), {IntermediateSplit(tensor1.prev, tensor2.prev, matmul_backward(tensor1.tensor, tensor2.tensor))})

def flatten(tensor):
    shape = tensor.tensor.shape

    return tensor.reshape(shape[0], -1)

def dropout(tensor, p):
    mask = cp.random.binomial(1, 1-p, tensor.tensor.shape)
    for prev in tensor.prev: break
    if prev.training:
        return Tensor(tensor.tensor * mask, {Intermediate(tensor.prev, dropout_backward(mask))})
    return tensor