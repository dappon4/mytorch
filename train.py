from Network import Network
from Trainer import Trainer

trainer = Trainer(batch=128, epochs=20, test_size=0.2, validation_size=0.1)

network = Network(size=[784, 512, 256, 128, 10], activations=["relu", "relu", "relu", "relu"], dropout_rate=[0.2, 0.2, 0.2], lr=0.005, loss_func="cross_entropy")
#network = Network([784, 64, 32, 10],["sigmoid", "sigmoid", "sigmoid"],lr=0.01, loss_func="mean_squared_error")

trainer.train(network)
trainer.accuracy(network)
trainer.visualize_loss()
