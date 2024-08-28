class Intermediate:
    def __init__(self, prev, grad_fn):
        self.prev = prev
        self.grad_fn = grad_fn
    
    def backward(self, error, lr):
        error = self.grad_fn(error)
        for prev in self.prev:
            prev.backward(error, lr)
    
    def train(self):
        for prev in self.prev:
            prev.train()
    
    def eval(self):
        for prev in self.prev:
            prev.eval()

class IntermediateSplit(Intermediate):
    def __init__(self, prev1, prev2, grad_fn):
        self.grad_fn = grad_fn
        self.prev1 = prev1
        self.prev2 = prev2
        # TODO: delete all the lambda in the layers
        
    def backward(self, error, lr):
        error_1, error_2 = self.grad_fn(error)
        for prev in self.prev1:
            prev.backward(error_1, lr)
        for prev in self.prev2:
            prev.backward(error_2, lr)