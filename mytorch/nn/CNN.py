from mytorch.nn.CompoundModule import CompoundModule
from mytorch.nn import Conv2d, Linear, MaxPool2d
from mytorch.F.Activation import relu
from mytorch.F.Util import flatten

class CNN(CompoundModule):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 32, 3, 1, 1)
        self.pool1 = MaxPool2d(2, 1, 0)
        
        self.conv2 = Conv2d(32, 64, 3, 1, 1)
        self.pool2 = MaxPool2d(2, 1, 0)
        
        self.linear1 = Linear(64*7*7, 128)
        self.lienar2 = Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = relu(x)
        x = self.pool2(x)
        
        x = flatten(x)
        x = self.linear(x)
        x = relu(x)
        
        x = self.linear2(x)
        x = relu(x)
        
        return x