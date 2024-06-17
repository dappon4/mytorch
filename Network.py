from Functions import *
from Layers import CompoundLayer, Linear, Residual, Conv2d, MaxPool2d, Flatten
from Trainer import Trainer

class Network(CompoundLayer):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = Linear(784, 256)
        self.linear2 = Linear(256, 128)
        self.linear3 = Linear(128, 10)
        
    def forward(self, x):
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        x = relu(self.linear3(x))
        return x

if __name__ == "__main__":
    trainer = Trainer(batch=64, epochs=40, lr = 0.005, test_size=0.2, validation_size=0.1, loss_func="cross_entropy", dastaset="MNIST", flatten = True)

    network = Network()
    #network = Network([784, 64, 32, 10],["sigmoid", "sigmoid", "sigmoid"],lr=0.01, loss_func="mean_squared_error")

    trainer.train(network)
    trainer.accuracy(network)
    trainer.visualize_loss()