from Functions import *
from Layers import CompoundLayer, Linear
from Trainer import Trainer, load_mnist

class ScaledDotProductAttention(CompoundLayer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        self.fc_Q = Linear(d_model, d_model)
        self.fc_K = Linear(d_model, d_model)
        self.fc_V = Linear(d_model, d_model)

    def forward(self, x):
        # x is of shape (batch_size, seq_len, d_model)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        
        QK = matmul(Q, K.transpose(0, 2, 1)) / cp.sqrt(self.d_model)
        # QK is of shape (batch, seq_len, seq_len)
        scaled = softmax(QK)
        x = matmul(scaled, V)
        # x is of shape (batch, seq_len, d_model)
        
        return x

if __name__ == "__main__":
    pass