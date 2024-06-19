class Tensor():
    def __init__(self, x, prev=None):
        if prev is None:
            self.prev = set()
        else:
            self.prev = prev
        self.tensor = x
    
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