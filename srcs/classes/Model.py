from classes.DenseLayer import DenseLayer
import classes.Loss as l

from colorama import Fore
import numpy as np
import utils.train_helper as u
import json

loss_functions = {
	"binary_cross_entropy": l.BinaryCrossEntropy,
	"categorical_cross_entropy": l.CategoricalCrossEntropy
}


class Model():

	def __init__(self, layers=None):
		self.layers = layers
		return

	def train(self, X, Y, callbacks=None, 
	        loss="binary_cross_entropy", learning_rate=0.01, batch_size=20, epochs=10000):

		self.set_train_parameters(X, Y, loss, batch_size, epochs, learning_rate)

		X_norm = (self.X - self.mean) / self.std

		self.callback(callbacks, 'set_model', self)
		self.callback(callbacks, 'on_train_begin', self.epochs)

		for epoch in range(0, self.epochs):
			X_curr, Y_curr = u.shuffle_order(X_norm, self.Y)
			batches = u.split_in_batches(X_curr, Y_curr, self.batch_size)

			for batch in batches:
				X_b, Y_b = batch[0], batch[1]
				A = self.forward(X_b)

				patch = self.loss.backward(A, Y_b)
				self.backward(patch, Y_b)

			logs = {}

			if self.callback(callbacks, 'on_epoch_end', epoch, logs):
				break

		self.callback(callbacks, 'on_train_end')


	def forward(self, X):
		curr_input = X
		for layer in self.layers:
			curr_input = layer.forward(curr_input)
		return curr_input

	def backward(self, dA, Y):
		for layer in reversed(self.layers):
			dA = layer.backward(dA, Y, self.learning_rate)

	def predict(self, X):
		x_norm =  (X - self.mean) / self.std
		A = self.forward(x_norm)
		return A

	def evaluate(self, X, Y):
		A = self.predict(X)
		binary = loss_functions["binary_cross_entropy"]()
		loss = binary.forward(A, Y)
		return loss

	def save(self, filename="data/trained_model.json"):
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
	def load(filename="data/trained_model.json"):
		with open(filename, "r") as f:
			data = json.load(f)
		layers = [DenseLayer.load(d) for d in data["layers"]]
		model = Model(layers)
		model.mean = np.array(data["mean"])
		model.std = np.array(data["std"])
		model.loss = loss_functions[data["loss"]]()
		return model

	def callback(self, callbacks, event, *args, **kwargs):
		if callbacks:
			for cb in callbacks:
				method = getattr(cb, event)
				flag = method(*args, **kwargs)
				if flag:
					return True
		return False

	def set_train_parameters(self, X, Y, loss, batch_size, epochs, learning_rate):
		if batch_size < 1 or epochs < 1 or learning_rate <= 0:
			raise ValueError(f"invalid hyperparameters: batch_size={batch_size}, epochs={epochs}, learning_rate={learning_rate} (all must be > 0)")
		self.X = X
		self.Y = Y
		self.mean = np.mean(X, axis=0)
		self.std = np.std(X, axis=0)
		self.std[self.std == 0] = 1
		self.loss_name = loss
		self.loss = loss_functions[loss]()
		self.batch_size = batch_size
		self.epochs = epochs
		self.learning_rate = learning_rate

	def get_weights(self):
		return [layer.get_weights() for layer in self.layers]

	def get_bias(self):
		return [layer.get_bias() for layer in self.layers]

	def set_weights(self, weights):
		if len(weights) != len(self.layers):
			raise ValueError(f"weights length mismatch: expected {len(self.layers)}, got {len(weights)}")
		for layer, w in zip(self.layers, weights):
			layer.set_weights(w)

	def set_bias(self, bias):
		if len(bias) != len(self.layers):
			raise ValueError(f"bias length mismatch: expected {len(self.layers)}, got {len(bias)}")
		for layer, b in zip(self.layers, bias):
			layer.set_bias(b)

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