from Functions import *
from Layers import CompoundLayer, Linear, LayerNorm

class MultiHeadAttention(CompoundLayer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.fc_Q = Linear(d_model, d_model)
        self.fc_K = Linear(d_model, d_model)
        self.fc_V = Linear(d_model, d_model)
    
    def forward(self, x):
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        
        batch_size, seq_len = x.shape[:2]
        Q.reshape(batch_size, self.num_heads, seq_len, self.d_model//self.num_heads)
        K.reshape(batch_size, self.num_heads, seq_len, self.d_model//self.num_heads)
        V.reshape(batch_size, self.num_heads, seq_len, self.d_model//self.num_heads)
        
        QK = matmul(Q, K.transpose(0, 1, 3, 2)) / cp.sqrt(self.d_model//self.num_heads)
        scaled = softmax(QK, axis=-2)
        
        x = matmul(scaled, V)
        x = x.reshape(batch_size, seq_len, self.d_model)
        
        return x

class EncoderLayer(CompoundLayer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
    
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.layernorm1 = LayerNorm()
        self.layernorm2 = LayerNorm()
        
        self.linear = Linear(d_model, d_model)
    
    def forward(self, x):
        x_res = x
        x = self.multi_head_attention(x)
        x = LayerNorm(x + x_res)
        
        x_res = x
        x = self.linear(x)
        x = self.layernorm2(x + x_res)
        
        return x

class Encoder(CompoundLayer):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = [EncoderLayer(512, 8) for _ in range(num_layers)]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x