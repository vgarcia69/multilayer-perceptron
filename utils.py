import numpy as np
import math

def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def relu(z):
	return np.maximum(0, z)


def relu_derivative(Z):
    return (Z > 0).astype(int)


def softmax(z):
	exps = np.exp(z - np.max(z, axis=1, keepdims=True))
	total = np.sum(exps, axis=1, keepdims=True)

	return (exps / total)


def binary_cross_entropy(p, Y):
	p_clipped = np.clip(p, 1e-7, 1-1e-7)

	losses = -(Y * np.log(p_clipped) + (1 - Y)*(np.log(1 - p_clipped)))

	return np.mean(losses)


def accuracy(p, Y):
	return np.mean((p >= 0.5) == Y)