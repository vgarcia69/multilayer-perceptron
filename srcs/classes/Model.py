from classes.DenseLayer import DenseLayer
import classes.Loss as l
import classes.Accuracy as a

from colorama import Fore
import numpy as np
import utils.train_helper as u
import json
import matplotlib.pyplot as plt

loss_functions = {
	"binary_cross_entropy": l.BinaryCrossEntropy,
	"categorical_cross_entropy": l.CategoricalCrossEntropy
}

class Model():

	def __init__(self, layers=None):
		self.layers = layers
		self.accuracy = a.Accuracy()
		return

	def train(self, X, Y, X_val=None, Y_val=None, loss="binary_cross_entropy", learning_rate=0.01, batch_size=20, epochs=10000):
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
		self.accuracy_val = []
		self.loss_val = []
		self.accuracy_train = []
		self.loss_train = []

		X_norm = (self.X - self.mean) / self.std

		for i in range(0, self.epochs):
			X_curr, Y_curr = u.shuffle_order(X_norm, self.Y)
			batches = u.split_in_batches(X_curr, Y_curr, self.batch_size)
			acc = []
			los = []
			for batch in batches:
				X_b, Y_b = batch[0], batch[1]
				A = self.forward(X_b)
				los.append(self.loss.forward(A, Y_b))
				acc.append(self.accuracy.forward(A, Y_b))
				patch = self.loss.backward(A, Y_b)
				self.backward(patch, Y_b)

			if X_val is not None and Y_val is not None:
				A_val = self.predict(X_val)
				self.accuracy_val.append(self.accuracy.forward(A_val, Y_val))
				self.loss_val.append(self.loss.forward(A_val, Y_val))
				print(f"{Fore.WHITE}epoch {i+1:02d}/{self.epochs:02d} - {Fore.CYAN}accuracy: {np.mean(acc):.4f} - loss: {np.mean(los):.4f} - {Fore.GREEN}accuracy (val): {np.mean(self.accuracy_val[-1]):.4f} - loss (val): {np.mean(self.loss_val[-1]):.4f}")
			else:
				print(f"{Fore.WHITE}epoch {i+1:02d}/{self.epochs:02d} - {Fore.CYAN}accuracy: {np.mean(acc):.4f} - loss: {np.mean(los):.4f} - {Fore.GREEN}accuracy (val): N/A - loss (val): N/A")
			self.accuracy_train.append(np.mean(acc))
			self.loss_train.append(np.mean(los))
		return 

	def forward(self, X):
		curr_input = X
		for layer in self.layers:
			curr_input = layer.forward(curr_input)
		return curr_input

	def backward(self, dA, Y):
		for layer in reversed(self.layers):
			dA = layer.backward(dA, Y, self.learning_rate)

	def show_learning_curve(self):
		plt.figure(figsize=(8, 5))
		plt.title("Loss")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.plot(self.loss_train, label="Training", color="blue", linewidth=2)
		plt.plot(self.loss_val, label="Validation", color="red", linewidth=2)
		plt.legend()


		plt.figure(figsize=(8, 5))
		plt.title("Learning Curve")
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy")
		plt.plot(self.accuracy_train, label="Training", color="blue", linewidth=2)
		plt.plot(self.accuracy_val, label="Validation", color="red", linewidth=2)
		plt.legend()

		plt.show()

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