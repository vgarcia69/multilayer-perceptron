from DenseLayer import DenseLayer
from colorama import Fore
import numpy as np
import utils as u
import Loss as l
import Accuracy as a

loss_functions = {
	"binary_cross_entropy": l.BinaryCrossEntropy,
	"categorical_cross_entropy": l.CategoricalCrossEntropy
}

class Model():

	def __init__(self, layers):

		if len(layers) < 3:
			print("ERREUR")
			return

		self.layers = layers
		return


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

	def set(self, X, Y, loss="binary_cross_entropy",  n_batch=10, n_epoch=10000, learning_rate=0.01):
		self.X = X
		self.Y = Y
		self.mean = np.mean(X, axis=0)
		self.std = np.std(X, axis=0)
		self.std[self.std == 0] = 1
		self.loss = loss_functions[loss]()
		self.n_batch = n_batch
		self.epoches = n_epoch
		self.learning_rate = learning_rate
		self.accuracy = a.Accuracy()

		if self.loss is None:
			print("ERREUR")
			return

		if self.n_batch < 1:
			print("ERREUR")
			return

		if self.epoches < 1:
			print("ERREUR")
			return

		if self.learning_rate <= 0:
			print("ERREUR")
			return


	def train(self):
		X_norm = (self.X - self.mean) / self.std
		for i in range(0, self.epoches):
			X_curr, Y_curr = u.shuffle_order(X_norm, self.Y)
			batches = u.split_in_batches(X_curr, Y_curr, self.n_batch)
			acc = []
			errors = []
			for batch in batches:
				X, Y = batch[0], batch[1]
				A = self.forward(X)
				errors.append(self.loss.forward(A, Y))
				acc.append(self.accuracy.forward(A, Y))
				patch = self.loss.backward(A, Y)
				self.backward(patch, Y)
			print(f"epoch {i+1:02d}/{self.epoches:02d} - accuracy: {np.mean(acc):.4f}, loss: {np.mean(errors):.4f}")
			
		return 


	def forward(self, X):
		curr_input = X
		for layer in self.layers:
			curr_input = layer.forward(curr_input)
		return curr_input

	def backward(self, A, batch_y):
		for layer in reversed(self.layers):
			A = layer.backward(A, self.learning_rate, batch_y)

	def predict(self, X):

		# pour predict on fait passer un tour dans le neural layers la data comme dans la fonction train mais sans le backprop
		pass
