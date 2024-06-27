import cupy as cp
from mytorch.nn import Intermediate
from mytorch.F.Gradient import mul_backward, div_backward, transpose_backward, reshape_backward

class Tensor():
    def __init__(self, x, prev=None):
        if prev is None:
            self.prev = set()
        else:
            self.prev = prev.copy()

        self.tensor = x
        self.shape = x.shape
    
    def __repr__(self) -> str:
        return f"Tensor({self.tensor}, previous={self.prev})"
    
    def __add__(self, other):
        return Tensor(self.tensor + other.tensor, self.prev.union(other.prev))
    
    def __mul__(self, num):

        return Tensor(num * self.tensor, {Intermediate(self.prev, mul_backward(num))})
    
    def __rmul__(self, num):
        return self.__mul__(num)
    
    def __truediv__(self, num):

        return Tensor(self.tensor/num, {Intermediate(self.prev, div_backward(num))})
    
    def transpose(self, *axis):

        return Tensor(self.tensor.transpose(*axis), {Intermediate(self.prev, transpose_backward(*axis))})
    
    def reshape(self, *shape):
        curr_shape = self.tensor.shape

        return Tensor(self.tensor.reshape(*shape), {Intermediate(self.prev, reshape_backward(curr_shape))})

if __name__ == "__main__":
    t1 = Tensor(cp.ones(3))
    t2 = t1
    t1 /= cp.random.rand(3)
    print(t1)
    print(t2)