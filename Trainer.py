from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from Functions import *
import cupy as cp
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

cp.cuda.Device(0).use()

class Trainer:
    def __init__(self, batch, epochs, lr, test_size, validation_size, loss_func ="mean_squared_error" ,dastaset="MNIST", flatten=True) -> None:
        self.batch = batch
        self.epochs = epochs
        self.lr = lr
        self.test_size = test_size
        self.validation_size = validation_size
        self.loss = None
        self.loss_derivative = None
        self.train_losses = []
        self.validation_losses = []
        
        if dastaset == "MNIST":
            self.load_MNIST(flatten)
        else:
            raise ValueError("Dataset not recognized")

        if loss_func == "mean_squared_error":
            self.loss = mean_squared_error
            self.loss_derivative = mean_squared_error_derivative
        elif loss_func == "cross_entropy":
            self.loss = cross_entropy
            self.loss_derivative = cross_entropy_derivative
    
    def load_MNIST(self, flatten):
        print("loading MNIST...")
        digits = fetch_openml('mnist_784')
        
        target = pd.get_dummies(digits.target).astype(int)
        target = cp.array(target)
        
        data = cp.array(digits.data)
        data = data / 255
        if not flatten:
            data = data.reshape(-1, 1, 28, 28)
        

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, target, test_size=self.test_size, random_state=42, stratify=np.argmax(cp.asnumpy(target), axis=-1))
        self.X_validation, self.X_test, self.y_validation, self.y_test = train_test_split(self.X_test, self.y_test, test_size=self.validation_size, random_state=42, stratify=np.argmax(cp.asnumpy(self.y_test), axis=-1))
        print("MNIST loaded")
    
    def calculate_error(self, y_pred, y_true):
        return self.loss_derivative(y_pred, y_true)
    
    def train(self, network):
        print("training...")

        for epoch in range(self.epochs):
            t1 = time.time()
            # Randomly permute data and label together
            permutation = cp.random.permutation(len(self.X_train))
            self.X_train = self.X_train[permutation]
            self.y_train = self.y_train[permutation]
            
            train_loss = 0
            validation_loss = 0
            
            network.train()
            for i in range(0, len(self.X_train), self.batch):
                batch_X = self.X_train[i:i+self.batch]
                batch_y = self.y_train[i:i+self.batch]
                y_pred = network.predict(batch_X)
                train_loss += self.loss(y_pred, batch_y)
                network.backward(self.calculate_error(y_pred, batch_y), self.lr)
            
            network.eval()
            for i in range(0, len(self.X_validation), self.batch):
                batch_X = self.X_validation[i:i+self.batch]
                batch_y = self.y_validation[i:i+self.batch]
                y_pred = network.predict(batch_X)
                validation_loss += self.loss(y_pred, batch_y)
            
            t2 = time.time()
            average_train_loss = train_loss / (len(self.X_train) // self.batch)
            average_validation_loss = validation_loss / (len(self.X_validation) // self.batch)
            
            self.train_losses.append(average_train_loss)
            self.validation_losses.append(average_validation_loss)
            
            print(f'Epoch {epoch+1}/{self.epochs} Training Loss: {average_train_loss:.4f} Validation Loss: {average_validation_loss:.4f} Time: {t2-t1:.3f} seconds')
    
    def visualize_loss(self):
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.validation_losses, label="Validation Loss")
        plt.legend()
        plt.show()
    
    def accuracy(self, network):
        batch = 100
        score = 0
        
        network.eval()
        
        for i in range(0, len(self.X_test), batch):
            X_batch = self.X_test[i:i+batch]
            y_batch = self.y_test[i:i+batch]
            y_pred = network.predict(X_batch)
            
            score += cp.sum(cp.argmax(y_pred, axis=-1) == np.argmax(y_batch, axis=-1)).item()
        
        print(f"Accuracy: {score / len(self.X_test) * 100:.2f}%")

        