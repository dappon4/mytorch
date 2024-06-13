from Functions import *
from Layers import CompoundLayer, Relu, Linear, Residual
from Trainer import Trainer

class Network(CompoundLayer):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = Linear(784, 128)
        self.linear2 = Linear(128, 64)
        self.linear3 = Linear(64, 128)
        self.linear4 = Linear(128, 10)
        
        self.relu1 = Relu()
        self.relu2 = Relu()
        self.relu3 = Relu()
        self.relu4 = Relu()
        self.relu5 = Relu()
        
        self.res1 = Residual()
        
    def forward(self, x):
        # x is (cp array, prev layer instance)
        x = self.linear1(x)
        x = self.relu1(x)
        x_res = x
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.res1(x, x_res)
        x = self.relu4(x)
        x = self.linear4(x)
        x = self.relu5(x)
        
        return x

if __name__ == "__main__":
    trainer = Trainer(batch=128, epochs=20, test_size=0.2, validation_size=0.1, loss_func="cross_entropy", dastaset="MNIST")

    network = Network()
    #network = Network([784, 64, 32, 10],["sigmoid", "sigmoid", "sigmoid"],lr=0.01, loss_func="mean_squared_error")

    trainer.train(network)
    trainer.accuracy(network)
    trainer.visualize_loss()