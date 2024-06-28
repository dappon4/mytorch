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
        one_hot_T = cp.eye(self.vocab_size, dtype=cp.float32)[self.input].transpose(0,2,1)
        delta_weight = cp.matmul(one_hot_T, error)
        delta_error = cp.matmul(error, self.weight.T)
        
        self.weight -= lr * cp.mean(delta_weight, axis=0)
        
        return delta_error
    
    def get_padding_mask(self):
        mask_base = (self.input==0)[..., cp.newaxis]
        half_mask = cp.repeat(mask_base, self.input.shape[1], axis=-1)
        half_mask_T = cp.moveaxis(half_mask, -1, -2)
        full_mask = half_mask | half_mask_T
        
        return cp.expand_dims(full_mask*(-cp.inf), 1)
    
    def get_causal_mask(self):
        seq_len = self.input.shape[1]
        mask = cp.triu(cp.ones((seq_len, seq_len)), 1)
        return (mask * -cp.inf)[cp.newaxis, cp.newaxis, ...]