from Functions import *
from Layers import CompoundLayer, Linear, Conv2d, MaxPool2d
from Trainer import Trainer, load_mnist

class CNN(CompoundLayer):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Conv2d(1, 16, 3, 1, 1)
        self.conv2 = Conv2d(16, 32, 3, 1, 1)
        
        self.pool1 = MaxPool2d(2, 2)
        self.pool2 = MaxPool2d(2, 2)
        
        self.linear1 = Linear(32 * 7 * 7, 128)
        self.linear2 = Linear(128, 10)
        
    def forward(self, x):
        x = relu(self.conv1(x))
        x = self.pool1(x)
        x = relu(self.conv2(x))
        x = flatten(self.pool2(x))

        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        return x

class SimpleFeedForward(CompoundLayer):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(784, 256)
        self.linear2 = Linear(256, 64)
        self.linear3 = Linear(64, 10)

    def forward(self, x):
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        x = relu(self.linear3(x))
        return x

class Test(CompoundLayer):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(784, 256)
        self.linear2 = Linear(256, 128)
        self.linear3 = Linear(128, 10)

    def forward(self, x):
        x = relu(self.linear1(x))
        x_ = x
        x_ = dropout(x_, 0.2)
        x = relu(self.linear2(x))
        x_ = dropout(x_, 0.2)
        x = relu(self.linear3(x))
        return x

if __name__ == "__main__":
    
    cp.random.seed(0)
    network = CNN()
    #network = Network([784, 64, 32, 10],["sigmoid", "sigmoid", "sigmoid"],lr=0.01, loss_func="mean_squared_error")
    X, y = load_mnist(flatten=False)
    trainer = Trainer(X, y, batch=256, epochs=5, lr = 0.005, test_size=0.2, validation_size=0.1, loss_func="cross_entropy")
    trainer.train(network)
    trainer.accuracy(network)
    trainer.visualize_loss()