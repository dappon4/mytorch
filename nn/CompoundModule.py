from mytorch.nn.Module import Module

class CompoundModule(Module):
    def __init__(self):
        super().__init__()
    
    def __call__(self, tensor, *args):

        tensor = self.forward(tensor, *args)
        
        return tensor