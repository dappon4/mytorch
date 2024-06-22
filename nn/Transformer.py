from F.Activation import softmax
from F.Util import matmul
from Module import CompoundModule, Linear, LayerNorm

import cupy as cp

class MultiHeadAttention(CompoundModule):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
    
    def forward(self, Q, K, V):
        
        batch_size, seq_len = x.shape[:2]
        Q.reshape(batch_size, self.num_heads, seq_len, self.d_model//self.num_heads)
        K.reshape(batch_size, self.num_heads, seq_len, self.d_model//self.num_heads)
        V.reshape(batch_size, self.num_heads, seq_len, self.d_model//self.num_heads)
        
        QK = matmul(Q, K.transpose(0, 1, 3, 2)) / cp.sqrt(self.d_model//self.num_heads)
        scaled = softmax(QK, axis=-2)
        
        x = matmul(scaled, V)
        x = x.reshape(batch_size, seq_len, self.d_model)
        
        return x

class EncoderLayer(CompoundModule):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
    
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.layernorm1 = LayerNorm()
        self.layernorm2 = LayerNorm()
        
        self.fc_Q = Linear(d_model, d_model)
        self.fc_K = Linear(d_model, d_model)
        self.fc_V = Linear(d_model, d_model)
        
        self.linear = Linear(d_model, d_model)
    
    def forward(self, x):
        x_res = x
        
        self.Q = self.fc_Q(x)
        self.K = self.fc_K(x)
        self.V = self.fc_V(x)
        
        x = self.multi_head_attention(self.Q,self.K,self.V)
        x = LayerNorm(x + x_res)
        
        x_res = x
        x = self.linear(x)
        x = self.layernorm2(x + x_res)
        
        return x

class Encoder(CompoundModule):
    def __init__(self, num_layers=6, d_model=512, num_heads=8):
        super().__init__()
        self.layers = [EncoderLayer(d_model, num_heads) for _ in range(num_layers)]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DecoderLayer(CompoundModule):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.fc_Q_1 = Linear(d_model, d_model)
        self.fc_K_1 = Linear(d_model, d_model)
        self.fc_V_1 = Linear(d_model, d_model)
        
        self.encoder_K = None
        self.encoder_V = None
        self.fc_Q_2 = Linear(d_model, d_model)
        
        self.linear = Linear(d_model, d_model)
        
        self.layernorm1 = LayerNorm()
        self.layernorm2 = LayerNorm()
        self.layernorm3 = LayerNorm()
        
        self.multi_head_attention1 = MultiHeadAttention(d_model, num_heads)
        self.multi_head_attention2 = MultiHeadAttention(d_model, num_heads)
    
    def forward(self, x):
        x_res = x
        
        Q = self.fc_Q_1(x)
        K = self.fc_K_1(x)
        V = self.fc_V_1(x)
        
        x = self.multi_head_attention1(Q,K,V)
        x = self.layernorm1(x + x_res)
        
        x_res = x
        Q = self.fc_Q_2(x)
        
        x = self.multi_head_attention2(Q, self.encoder_K, self.encoder_V)
        x = self.layernorm2(x + x_res)
        
        x_res = x
        x = self.linear(x)
        
        x = self.layernorm3(x + x_res)
        
        return x
        

class Decoder(CompoundModule):
    def __init__(self, num_layers=6, d_model=512, num_heads=8):
        super().__init__()
        self.layers = [DecoderLayer(d_model, num_heads) for _ in range(num_layers)]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def set_encoder_KV(self, encoder_K, encoder_V):
        for layer in self.layers:
            layer.encoder_K = encoder_K
            layer.encoder_V = encoder_V

class Transformer(CompoundModule):
    def __init__(self, vocab_size, num_layers=6, d_model=512, num_heads=8):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads)
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads)
        
        self.fc_encoder_K = Linear(512, 512)
        self.fc_encoder_V = Linear(512, 512)
        
        self.output_linear = Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.encoder(x)
        
        encoder_K = self.fc_encoder_K(x)
        encoder_V = self.fc_encoder_V(x)
        
        self.decoder.set_encoder_KV(encoder_K, encoder_V)
        
        # ???????????????????
        x = self.decoder(x)
        
        # lienar layer for output (d_model, vocab_size)
        x = self.output_linear(x)
        x = softmax(x)
        
        return x