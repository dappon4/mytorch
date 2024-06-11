import cupy as cp

class Layer:
    def __init__(self, input_size, output_size, lr):
        self.lr = lr
        self.input_size = input_size
        self.output_size = output_size
        
        self.weight = None
        self.bias = None
        
        self.input = None

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError
    
    def xavier_init(self, input_size, output_size):
        return cp.random.normal(0.0, cp.sqrt(2/(input_size + output_size)), (input_size, output_size))

class Dense(Layer):
    def __init__(self, input_size, output_size, lr, dropout=0.0):
        super().__init__(input_size, output_size, lr)
        self.weight = cp.random.rand(input_size, output_size) - 0.5
        self.bias = cp.random.rand(1, output_size) - 0.5
        
        self.dropout = dropout

    def forward(self, x, training=True):
        if training:
            dropout_mask = cp.random.binomial(1, 1-self.dropout, size = x.shape)
            x = x * dropout_mask / (1-self.dropout)
            
        self.input = x
        x = cp.matmul(x, self.weight) + self.bias
        return x

    def backward(self, error):
        delta_weight = cp.matmul(self.input.transpose(0,2,1), error)
        delta_error = cp.matmul(self.weight, error.transpose(0,2,1))
        
        self.weight -= self.lr * cp.mean(delta_weight, axis = 0)
        self.bias -= self.lr * cp.mean(error, axis=0)
        
        return delta_error.transpose(0,2,1)

class Relu(Layer):
    def forward(self, x):
        self.input = x
        return cp.maximum(x, 0)

    def backward(self, error):
        return error * cp.where(self.input > 0, 1, 0)