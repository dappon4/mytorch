from mytorch.nn.Transformer import Transformer, PositionalEncoding
from mytorch.Tensor import Tensor
from mytorch.F.Evaluator import cross_entropy
from mytorch.F.Activation import relu
import mytorch.nn as nn
from mytorch.Util.Trainer import Trainer, load_mnist

class FeedForward(nn.CompoundModule):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        return x
        


if __name__ == "__main__":
    #network = nn.CNN()
    network = FeedForward()
    
    X, y = load_mnist(flatten=True)
    trainer = Trainer(X,y, batch=32, epochs = 20, lr=0.005, test_size=0.2, validation_size=0.1, loss_func=cross_entropy)
    trainer.train(network)
    trainer.accuracy(network)
    trainer.visualize_loss()
    network.save("model/SimpleFeedForward.pkl")