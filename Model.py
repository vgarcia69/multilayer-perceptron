from DenseLayer import DenseLayer
from colorama import Fore
import numpy as np
import utils as u
import Loss as l
import Accuracy as a
import json

loss_functions = {
	"binary_cross_entropy": l.BinaryCrossEntropy,
	"categorical_cross_entropy": l.CategoricalCrossEntropy
}

class Model():

	def __init__(self, layers=None):
		self.layers = layers
		self.accuracy = a.Accuracy()
		return

	def train(self, X, Y, loss="binary_cross_entropy",  n_batch=20, n_epoch=10000, learning_rate=0.01):
		self.X = X
		self.Y = Y
		self.mean = np.mean(X, axis=0)
		self.std = np.std(X, axis=0)
		self.std[self.std == 0] = 1
		self.loss_name = loss
		self.loss = loss_functions[loss]()
		self.n_batch = n_batch
		self.epoches = n_epoch
		self.learning_rate = learning_rate

		X_norm = (self.X - self.mean) / self.std
		for i in range(0, self.epoches):
			X_curr, Y_curr = u.shuffle_order(X_norm, self.Y)
			batches = u.split_in_batches(X_curr, Y_curr, self.n_batch)
			acc = []
			errors = []
			for batch in batches:
				X_b, Y_b = batch[0], batch[1]
				A = self.forward(X_b)
				errors.append(self.loss.forward(A, Y_b))
				acc.append(self.accuracy.forward(A, Y_b))
				patch = self.loss.backward(A, Y_b)
				self.backward(patch, Y_b)

			print(f"epoch {i+1:02d}/{self.epoches:02d} - accuracy: {np.mean(acc):.4f} - loss: {np.mean(errors):.4f}")
			
		return 

	def forward(self, X):
		curr_input = X
		for layer in self.layers:
			curr_input = layer.forward(curr_input)
		return curr_input

	def backward(self, A, batch_y):
		for layer in reversed(self.layers):
			A = layer.backward(A, self.learning_rate, batch_y)

	def save(self, filename="trained_model.json"):
		data = {
		"loss": self.loss_name,
		"mean": self.mean.tolist(),
		"std": self.std.tolist(),
		"layers": [layer.save() for layer in self.layers],
		}

		with open(filename, "w") as f:
			json.dump(data, f, indent=2)
		return


	@staticmethod
	def load(filename="trained_model.json"):
		with open(filename, "r") as f:
			data = json.load(f)
		layers = [DenseLayer.load(d) for d in data["layers"]]
		model = Model(layers)
		model.mean = np.array(data["mean"])
		model.std = np.array(data["std"])
		model.loss = loss_functions[data["loss"]]()
		return model

	def predict(self, X):
		x_norm =  (X - self.mean) / self.std
		A = self.forward(x_norm)
		return A

	def __str__(self):
		lines = []
		for i, layer in enumerate(self.layers):
			lines.append(Fore.YELLOW + f"===== Layer {i + 1} : {layer.n_inputs} -> {layer.n_neurons} =====")
			lines.append(Fore.GREEN + f"weights {layer.weights.shape} :")
			lines.append(Fore.WHITE + str(layer.weights))
			lines.append(Fore.GREEN + f"bias {layer.bias.shape} :")
			lines.append(Fore.WHITE + str(layer.bias))
			lines.append("")
		return "\n".join(lines)