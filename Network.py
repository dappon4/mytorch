from Functions import *
from Layers import CompoundLayer, Linear, Residual, Conv2d, MaxPool2d, Flatten
from Trainer import Trainer

class Network(CompoundLayer):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Conv2d(1, 16, 3, 1, 1)
        self.conv2 = Conv2d(16, 32, 3, 1, 1)
        
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)
        
        self.linear1 = Linear(32 * 7 * 7, 128)
        self.linear2 = Linear(128, 64)
        self.linear3 = Linear(64, 10)
        
        self.flatten = Flatten()
        
    def forward(self, x):
        # x is (cp array, prev layer instance)
        x = self.conv1(x)
        x = self.pool1(x) # output shape: (batch_size, 16, 14, 14)
        
        x = self.conv2(x) # output shape: (batch_size, 32, 14, 14)
        x = self.pool2(x) # output shape: (batch_size, 32, 7, 7)
        
        x = self.flatten(x)
        
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        x = relu(self.linear3(x))
        
        return x

if __name__ == "__main__":
    trainer = Trainer(batch=128, epochs=10, test_size=0.2, validation_size=0.1, loss_func="cross_entropy", dastaset="MNIST", flatten = False)

    network = Network()
    #network = Network([784, 64, 32, 10],["sigmoid", "sigmoid", "sigmoid"],lr=0.01, loss_func="mean_squared_error")

    trainer.train(network)
    trainer.accuracy(network)
    trainer.visualize_loss()