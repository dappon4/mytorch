from Functions import *
from Layers import CompoundLayer, Relu, Linear
from Trainer import Trainer

class Network(CompoundLayer):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = Linear(784, 256)
        self.linear2 = Linear(256, 128)
        self.linear3 = Linear(128, 10)
        
        self.relu1 = Relu()
        self.relu2 = Relu()
        self.relu3 = Relu()
        
    def forward(self, x, training=True):
        # x is (cp array, prev layer instance)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        
        return x

if __name__ == "__main__":
    trainer = Trainer(batch=128, epochs=20, test_size=0.2, validation_size=0.1, loss_func="cross_entropy")

    network = Network()
    #network = Network([784, 64, 32, 10],["sigmoid", "sigmoid", "sigmoid"],lr=0.01, loss_func="mean_squared_error")

    trainer.train(network)
    trainer.accuracy(network)
    trainer.visualize_loss()