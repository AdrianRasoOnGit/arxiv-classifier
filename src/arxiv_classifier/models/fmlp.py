#!/usr/bin/python3

import numpy as np
from scipy.sparse import issparse

from ..utils.adam_gradient import AdamOptimizer


class NeuralNetwork:

    # The neural network itself, composed of vectors provided by numpy. The notation is quite straightforward, but in any case, W references to a vector of weights, and b to a vector of biases
    def __init__(self, layer_dims, lr = 0.001, dropout_rate = 0.0, lam = 1e-4):
        self.L = len(layer_dims) - 1
        self.params = {}
        for l in range(1, self.L + 1):
            self.params[f"W{l}"] = np.random.randn(layer_dims[l - 1], layer_dims[l]) * np.sqrt(2 / layer_dims[l - 1]) # He initilization
            self.params[f"b{l}"] = np.zeros((1, layer_dims[l]))
            
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.lam = lam
        self.optimizer = AdamOptimizer(self.params, lr = self.lr)

        # Normalization
        self.mean = None
        self.std = None

    # Here we introduce a nn builder: given the amount of layers desired, we can design an appropiate nn. Great to tweak around, so we don't commit to any particular size
    @classmethod
    def setup(cls, input_dim, output_dim, config: dict):

        hidden_layers = config["hidden_layers"]
        hidden_dim = config["hidden_dim"]
        lr = config["learning_rate"]
        dropout_rate = config["dropout_rate"]
        lam = config.get("lam", 1e-4)
        
        layer_dims = [input_dim] + [hidden_dim] * hidden_layers + [output_dim]
        print(f"Preparing network with {layer_dims} layers. Learning rate is at {lr}, dropout rate at {dropout_rate}, and lam at {lam}.")
        return cls(layer_dims, lr = lr, dropout_rate = dropout_rate, lam = lam)

     # Activation functions
    def relu(self, z):
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis = 1, keepdims = True))
        return exp_z / np.sum(exp_z, axis = 1, keepdims = True)

    # Loss function (L2 weight decay)
    def loss(self, Y_hat, Y):
        m = Y.shape[0]
        ce = -np.mean(np.log(Y_hat[np.arange(m), Y] + 1e-9))
        reg = (self.lam / (2 * m)) * sum(np.sum(W**2) for W in self.params.values() if W.ndim == 2)
        return ce + reg

    
    # Forward and backward movements. Clarification: as usual, Z are the logits, A the layers
    def forward(self, X, dropout = True):
        cache = {"A0": X}
        for l in range(1, self.L):
            Z = cache[f"A{l-1}"] @ self.params[f"W{l}"] + self.params[f"b{l}"]
            A = self.relu(Z)
            
            # cache[f"Z{l}"] = Z
            # cache[f"A{l}"] = self.relu(Z)

            if dropout and self.dropout_rate > 0:
                D = (np.random.rand(*A.shape) > self.dropout_rate).astype(float)
                A *= D
                A /= (1.0 - self.dropout_rate)
                cache[f"D{l}"] = D

            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A

        # Output layer
        ZL = cache[f"A{self.L-1}"] @ self.params[f"W{self.L}"] + self.params[f"b{self.L}"]
        cache[f"Z{self.L}"] = ZL
        cache[f"A{self.L}"] = self.softmax(ZL)
        return cache

    def backward(self, cache, Y):
        grads = {}
        m = Y.shape[0]
        A_final = cache[f"A{self.L}"]
        dZ = A_final.copy()
        dZ[np.arange(m), Y] -= 1
        dZ /= m

        for l in reversed(range(1, self.L + 1)):
            A_prev = cache[f"A{l - 1}"]
            grads[f"dW{l}"] = A_prev.T @ dZ
            grads[f"db{l}"] = np.sum(dZ, axis = 0, keepdims = True)

            if l > 1:
                dA_prev = dZ @ self.params[f"W{l}"].T
                dZ = dA_prev * self.relu_deriv(cache[f"Z{l - 1}"])

                if f"D{l - 1}" in cache:
                    dZ *= cache[f"D{l - 1}"]
                    dZ /= (1.0 - self.dropout_rate)

        for g in grads.values():
            np.clip(g, -5, 5, out = g)

        self.optimizer.step(self.params, grads)


    # Training
    def train(self, X, Y, config):

        # Ensure sparse matrices are in CSR format, and force float32 to improve performance
        if issparse(X):
            X = X.astype(np.float32)
        else:
            X = X.astype(np.float32)

        # Define shuffle samples globally
        n_samples = X.shape[0]

        epochs = config["epochs"]
        batch_size = config["batch_size"]
        losses_per_epoch = []
        accs_per_epoch = []
        
        # Training without early stop
        
        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)

            losses, accs = [], []

            for i in range(0, n_samples, batch_size):
                Xb = X[idx[i : i + batch_size]]
                Yb = Y[idx[i : i + batch_size]]

                cache = self.forward(Xb, dropout=True)
                loss = self.loss(cache[f"A{self.L}"], Yb)
                self.backward(cache, Yb)

                preds = np.argmax(cache[f"A{self.L}"], axis=1)
                acc = np.mean(preds == Yb)
                losses.append(loss)
                accs.append(acc)

            epoch_loss = np.mean(losses)
            epoch_acc = np.mean(accs)
            losses_per_epoch.append(epoch_loss)
            accs_per_epoch.append(epoch_acc)

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.3f}"
            )
        return losses_per_epoch, accs_per_epoch

