from classes.CallBacks.CallBack import CallBack
import classes.Loss as l
import numpy as np

class Metrics(CallBack):
	def __init__(self, X, Y, metrics_required=["accuracy", "loss"], prefix="", positive=1):
		self.X = X
		self.Y = Y
		self.metrics = metrics_required
		self.positive = positive
		self.prefix = prefix

	def on_epoch_end(self, epoch, logs=None):
		A = self.model.predict(self.X)

		pred = np.argmax(A, axis=1) == self.positive
		true = np.argmax(self.Y, axis=1) == self.positive
		true_positives = np.sum(pred & true)
		false_positives = np.sum(pred & ~true)
		false_negatives = np.sum(~pred & true)
  
		loss = self.model.loss.forward(A, self.Y)
		accuracy = np.mean(pred == true)
		recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.000
		precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.000

		available_metrics = {
			"accuracy": accuracy,
			"recall": recall,
			"precision": precision,
			"loss": loss,
		}

		for metric in self.metrics:
			if metric in available_metrics:
				key = f"{self.prefix}{metric}"
				logs[key] = available_metrics[metric]
