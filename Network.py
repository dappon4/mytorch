from mytorch.nn.Transformer import Transformer, PositionalEncoding
from mytorch.Tensor import Tensor
import mytorch.nn as nn
import cupy as cp
import time

if __name__ == "__main__":
    batch_size = 16
    vocab_size = 30000
    seq_size = 100
    embed_dim = 200
    d_ff = 2048
    dummy_input = cp.random.randint(0, vocab_size-1, (batch_size, seq_size))
    dec_input = cp.random.randint(0, vocab_size-1, (batch_size, seq_size))
    transformer = Transformer(vocab_size, vocab_size, num_layers=1, d_model=embed_dim, num_heads=8, d_ff=d_ff)
    transformer.train()
    
    for i in range(5):
        
        res = transformer(Tensor(dummy_input), Tensor(dec_input))
        
        
    final_layer = res.prev
    print(res.shape)
    error = cp.random.rand(*res.shape)
    t1 = time.time()
    for prev in final_layer: break
    prev.backward(error, 0.005)
    t2 = time.time()
    print(t2-t1)
    t1 = time.time()
    for prev in final_layer: break
    prev.backward(error, 0.005)
    t2 = time.time()
    print(t2-t1)
    
    