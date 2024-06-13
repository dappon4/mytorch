from Network import Network
from Trainer import Trainer

trainer = Trainer(batch=128, epochs=20, test_size=0.2, validation_size=0.1, loss_func="cross_entropy")

#network = Network()
network = Network([784, 256, 128, 10],["relu", "relu", "relu"],dropout_rate=[0.2,0.2], lr=0.005)

trainer.train(network)
trainer.accuracy(network)
trainer.visualize_loss()
