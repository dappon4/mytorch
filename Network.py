from F.Activation import relu
from F.Evaluator import cross_entropy
from nn.Module import CompoundModule, Linear
from Util.Trainer import Trainer, load_mnist

class SimpleFeedForward(CompoundModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        x = relu(x)
        
        return x

if __name__ == "__main__":
    X, y = load_mnist()
    network = SimpleFeedForward(784, 128, 10)
    trainer = Trainer(X, y, batch=64, epochs=10, lr=0.001, test_size=0.2, validation_size=0.2, loss_func=cross_entropy)
    trainer.train(network)
    trainer.accuracy(network)
    trainer.visualize_loss()