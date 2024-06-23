import cupy as cp

class Tensor():
    def __init__(self, x, prev=None):
        if prev is None:
            self.prev = set()
        else:
            self.prev = prev.copy()
        self.tensor = x
    
    def __repr__(self) -> str:
        return f"Tensor({self.tensor}, previous={self.prev})"
    
    def __add__(self, other):
        return Tensor(self.tensor + other.tensor, self.prev.union(other.prev))
    
    def __mul__(self, num):
        for prev in self.prev:
            prev.error_grad = lambda x: prev.error_grad(num*x)
        return Tensor(num * self.tensor, self.prev)
    
    def __rmul__(self, num):
        return self.__mul__(num)
    
    def transpose(self, *dimension):
        counter_index = [0]*len(dimension)
        for i, dim in enumerate(dimension):
            counter_index[dim] = i
        
        counter_index = tuple(counter_index)
        for prev in self.prev:
            f = prev.error_grad
            prev.error_grad = lambda x: f(x.transpose(*counter_index))
        
        return Tensor(self.tensor.transpose(*dimension), self.prev)
    
    def reshape(self, *shape):
        curr_shape = self.tensor.shape
        for prev in self.prev:
            f = prev.error_grad
            prev.error_grad = lambda x: f(x.reshape(*curr_shape))

        return Tensor(self.tensor.reshape(*shape), self.prev)

if __name__ == "__main__":
    t1 = Tensor(cp.ones(3))
    t2 = t1
    t1 *= 3
    print(t1)
    print(t2)