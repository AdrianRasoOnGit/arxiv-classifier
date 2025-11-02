#!usr/bin/python3
import numpy as np

# Activation functions
    def relu(self, z):
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis = 1, keepdims = True))
        return exp_z / np.sum(exp_z, axis = 1, keepdims = True)
