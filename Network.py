from mytorch.F.Activation import relu
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
    dummy_input = cp.random.rand(batch_size, seq_size, embed_dim)
    dec_input = cp.random.rand(batch_size, seq_size, embed_dim)
    transformer = Transformer(vocab_size, num_layers=2, d_model=embed_dim, num_heads=8, d_ff=d_ff)
    transformer.train()
    
    res = transformer(Tensor(dummy_input), Tensor(dec_input))
    print(res)
    print(res.shape)
    
    