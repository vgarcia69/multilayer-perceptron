import numpy as np

class Accuracy:
	def __init__(self):
		pass

	def forward(self, y_pred, y_true):
		predictions = np.argmax(y_pred, axis=1)
		labels = np.argmax(y_true, axis=1)
		return np.mean(predictions == labels)