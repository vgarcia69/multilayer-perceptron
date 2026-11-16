import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
	@abstractmethod
	def forward(self, y_pred, y_true):
		pass

	@abstractmethod
	def backward(self, y_pred, y_true):
		pass


class BinaryCrossEntropy(Loss):
	def forward(self, y_pred, y_true):
		p_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

		losses = -(y_true * np.log(p_clipped) + (1 - y_true)*(np.log(1 - p_clipped)))

		return np.mean(losses)

	def backward(self, y_pred, y_true):
		p_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
		return -(y_true / p_clipped - (1 - y_true) / (1 - p_clipped)) / y_pred.shape[0]


class CategoricalCrossEntropy(Loss):
	def forward(self, y_pred, y_true):
		p_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

		losses = -np.sum(y_true * np.log(p_clipped), axis=1)

		return np.mean(losses)

	def backward(self, y_pred, y_true):
		return (y_pred - y_true) / y_pred.shape[0]
