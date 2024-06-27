from mytorch.nn import Module
from mytorch.F.Initializer import xavier_init
import cupy as cp

class Embedding(Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weight = xavier_init(vocab_size, embed_dim)
        self.input = None
    
    def forward(self, x):
        self.input = x
        return cp.take(self.weight, x, axis=0)
    
    def backward_calc(self, error, lr):
        # TODO: maybe find a better way to do this
        one_hot_T = cp.eye(self.vocab_size)[self.input].transpose(0,2,1)
        delta_weight = cp.matmul(one_hot_T, error)
        delta_error = cp.matmul(error, self.weight.T)
        
        self.weight -= lr * delta_weight
        
        return delta_error