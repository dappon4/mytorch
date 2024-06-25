from mytorch.F.Activation import relu
from mytorch.nn.Transformer import Transformer
from mytorch.Tensor import Tensor
import mytorch.nn as nn
import cupy as cp

class SimpleFeedForward(nn.CompoundModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        x = relu(x)
        
        return x

if __name__ == "__main__":
    batch_size = 16
    vocab_size = 1000
    seq_size = 128
    embed_dim = 256
    dummy_input = cp.random.rand(batch_size, seq_size, embed_dim)
    dec_input = cp.random.rand(batch_size, seq_size, embed_dim)
    transformer = Transformer(vocab_size, num_layers=2, d_model=embed_dim, num_heads=4)
    transformer.train()
    
    print(transformer(Tensor(dummy_input), Tensor(dec_input)))
    
    
    