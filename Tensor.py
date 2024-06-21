import cupy as cp

class Tensor():
    def __init__(self, x, prev=None):
        if prev is None:
            self.prev = set()
        else:
            self.prev = prev
        self.tensor = x
    
    def __repr__(self) -> str:
        return f"Tensor({self.tensor}, previous={self.prev})"
    
    def __add__(self, other):
        self.prev = self.prev.union(other.prev)
        self.tensor += other.tensor
        
        return self
    
    def __mul__(self, num):
        self.tensor *= num
        for prev in self.prev:
            prev.error_grad = lambda x: prev.error_grad(num*x)
        return self
    
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
        
        self.tensor.transpose(*dimension)
    
    def reshape(self, *shape):
        curr_shape = self.tensor.shape
        for prev in self.prev:
            f = prev.error_grad
            prev.error_grad = lambda x: f(x.reshape(*curr_shape))

        return Tensor(self.tensor.reshape(*shape), self.prev)

if __name__ == "__main__":
    t = Tensor(cp.random.rand(4,5,6,7,8))
    t.transpose(3,0,1,4,2)
    #t.transpose(1,2,0)
    t.transpose(1,2,4,0,3)
    print(t.tensor.shape)