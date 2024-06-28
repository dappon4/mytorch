from mytorch.nn.Transformer import Transformer, PositionalEncoding
from mytorch.Tensor import Tensor
import mytorch.nn as nn
import cupy as cp

if __name__ == "__main__":
    batch_size = 4
    vocab_size = 1000
    seq_size = 128
    embed_dim = 256
    d_ff = 2048
    dummy_input = cp.random.randint(0, vocab_size-1, (batch_size, seq_size))
    dec_input = cp.random.randint(0, vocab_size-1, (batch_size, seq_size))
    transformer = Transformer(vocab_size, vocab_size, num_layers=1, d_model=embed_dim, num_heads=8, d_ff=d_ff)
    transformer.train()
    
    res = transformer(Tensor(dummy_input), Tensor(dec_input))
    final_layer = res.prev
    print(res.shape)
    error = cp.random.rand(*res.shape)
    
    for prev in final_layer: break
    prev.backward(error, 0.005)
    
    