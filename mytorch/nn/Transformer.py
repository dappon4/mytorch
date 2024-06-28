from mytorch.F.Activation import relu, softmax
from mytorch.F.Util import matmul
from mytorch.nn import CompoundModule, Linear, LayerNorm, Module, Embedding

import cupy as cp

# TODO: create mask for MHA
# TODO: create embedding model

class PositionalEncoding(Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, x):
        seq_len = x.shape[1]
        
        denom = cp.power(10000, cp.repeat(cp.arange(self.d_model/2), 2)/ self.d_model)[cp.newaxis, cp.newaxis, ...]
        num = cp.arange(seq_len)[cp.newaxis, ..., cp.newaxis]
        
        x[:,:,::2] += cp.sin((num/denom)[:,:,::2])
        x[:,:,1::2] += cp.cos((num/denom)[:,:,1::2])
        
        return x
        

class MultiHeadAttention(CompoundModule):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
    
    def forward(self, Q, K, V, mask):
        
        batch_size, seq_len, _ = Q.shape
        Q = Q.reshape(batch_size, self.num_heads, seq_len, self.d_model//self.num_heads)
        K = K.reshape(batch_size, self.num_heads, seq_len, self.d_model//self.num_heads)
        V = V.reshape(batch_size, self.num_heads, seq_len, self.d_model//self.num_heads)
        
        QK = matmul(Q, K.transpose(0, 1, 3, 2)) / cp.sqrt(self.d_model//self.num_heads, dtype=cp.float32)
        if mask is not None:
            QK = QK + mask
        scaled = softmax(QK)
        
        x = matmul(scaled, V)
        x = x.reshape(batch_size, seq_len, self.d_model)
        
        return x

class EncoderLayer(CompoundModule):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.num_heads = num_heads
    
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.layernorm1 = LayerNorm()
        self.layernorm2 = LayerNorm()
        
        self.fc_Q = Linear(d_model, d_model)
        self.fc_K = Linear(d_model, d_model)
        self.fc_V = Linear(d_model, d_model)
        
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
    
    def forward(self, x, mask):
        x_res = x
        
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        
        x = self.multi_head_attention(Q, K, V, mask)
        x = self.layernorm1(x + x_res)
        
        x_res = x
        x = self.linear1(x)
        x = relu(x)
        x = self.linear2(x)
        
        x = self.layernorm2(x + x_res)

        return x

class Encoder(CompoundModule):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super().__init__()
        self.layers = [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderLayer(CompoundModule):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.fc_Q_1 = Linear(d_model, d_model)
        self.fc_K_1 = Linear(d_model, d_model)
        self.fc_V_1 = Linear(d_model, d_model)
        
        self.fc_Q_2 = Linear(d_model, d_model)
        self.fc_K_2 = Linear(d_model, d_model)
        self.fc_V_2 = Linear(d_model, d_model)
        
        self.linear1 = Linear(d_model, d_ff)
        self.lienar2 = Linear(d_ff, d_model)
        
        self.layernorm1 = LayerNorm()
        self.layernorm2 = LayerNorm()
        self.layernorm3 = LayerNorm()
        
        self.multi_head_attention1 = MultiHeadAttention(d_model, num_heads)
        self.multi_head_attention2 = MultiHeadAttention(d_model, num_heads)
    
    def forward(self, x, encoder_output, mask):
        x_res = x
        
        Q = self.fc_Q_1(x)
        K = self.fc_K_1(x)
        V = self.fc_V_1(x)
        
        x = self.multi_head_attention1(Q, K, V, mask)
        x = self.layernorm1(x + x_res)
        
        x_res = x
        Q = self.fc_Q_2(x)
        K = self.fc_K_2(encoder_output)
        V = self.fc_V_2(encoder_output)
        
        x = self.multi_head_attention2(Q, K, V, mask)
        x = self.layernorm2(x + x_res)
        
        x_res = x
        
        x = self.linear1(x)
        x = relu(x)
        x = self.lienar2(x)
        
        x = self.layernorm3(x + x_res)
        
        return x
        

class Decoder(CompoundModule):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super().__init__()
        self.layers = [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
    
    def forward(self, x, decoder_input, mask=None):
        encoder_output = x
        for layer in self.layers:
            decoder_input = layer(decoder_input, encoder_output, mask)
        return decoder_input

class Transformer(CompoundModule):
    def __init__(self, src_vocab_size, tgt_vocab_size, num_layers=6, d_model=512, num_heads=8, d_ff=2048):
        super().__init__()
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        self.src_embedding = Embedding(src_vocab_size, d_model)
        self.tgt_embedding = Embedding(tgt_vocab_size, d_model)
        
        self.positiolnal_encoder_src = PositionalEncoding(d_model)
        self.positiolnal_encoder_tgt = PositionalEncoding(d_model)
        
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff)
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff)
        
        self.fc_encoder_K = Linear(512, 512)
        self.fc_encoder_V = Linear(512, 512)
        
        self.output_linear = Linear(d_model, tgt_vocab_size)
        
    def forward(self, encoder_input, decoder_input):
        
        x = self.src_embedding(encoder_input)
        x = self.positiolnal_encoder_src(x)
        x = self.encoder(x, self.src_embedding.get_padding_mask())

        decoder_input = self.tgt_embedding(decoder_input)
        decoder_input = self.positiolnal_encoder_tgt(decoder_input)
        x = self.decoder(x, decoder_input, self.tgt_embedding.get_causal_mask() + self.tgt_embedding.get_padding_mask())
        
        # lienar layer for output (d_model, vocab_size)
        x = self.output_linear(x)
        x = softmax(x)
        
        return x
    
    def predict(self, x):
        pass
    
    def get_mask(self, x):
        pass