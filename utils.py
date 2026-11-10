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

	return (np.max(exps, axis=1, keepdims=True) / total)


def standard_deviation(values, mean):
	var = 0
	v_size = len(values)
	if (v_size <= 1):
		return var 
	for v in values:
		ecart = v - mean
		var = var + ecart * ecart
	var = var / (v_size - 1)
	return math.sqrt(var)