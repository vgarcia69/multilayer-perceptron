import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
	@abstractmethod
	def forward(self, X):
		return X

	@abstractmethod
	def backward(self, X):
		return 1

class Sigmoid(Activation):
	def forward(self, X):
		return 1 / (1 + np.exp(-np.clip(X, -500, 500)))

	def backward(self, X):
		sigmoid_x = self.forward(X)
		return sigmoid_x * (1 - sigmoid_x)

class ReLU(Activation):
	def forward(self, X):
		return np.maximum(0, X)

	def backward(self, X):
		return (X > 0).astype(float)

class Softmax(Activation):
	def forward(self, X):
		exps = np.exp(X - np.max(X, axis=1, keepdims=True))
		return exps / np.sum(exps, axis=1, keepdims=True)

	def backward(self, X):
		raise NotImplementedError("Softmax backward pass is typically handled with the loss function.")
