from mytorch.Tensor import Tensor

class Module:
    def __init__(self):
        self.prev = set()
        self.training = True
        
        self.error_grad = lambda x: x
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
        
    def __call__(self, tensor, *args):
        self.prev = tensor.prev.copy()
        return Tensor(self.forward(tensor.tensor.copy(), *args), {self})
    
    def forward(self, x):
        raise NotImplementedError

    def backward_calc(self, error, lr):
        return error
    
    def backward(self, error, lr):
        delta_error = self.backward_calc(error, lr)
        
        if self.prev:
            for prev in self.prev:
                prev.backward(delta_error, lr)
        else:
            return delta_error
    
    def train(self):
        self.training = True
        for prev in self.prev:
            prev.train()
    
    def eval(self):
        self.training = False
        for prev in self.prev:
            prev.eval()