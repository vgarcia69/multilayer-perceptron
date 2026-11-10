from DenseLayer import DenseLayer
from colorama import Fore
import numpy as np
import utils as u

class NeuralNetwork():

	def __init__(self, layers_size):

		if len(layers_size) < 4:
			return

		self.layers_size = layers_size
		self.network = []

		self.create()
		return


	def __str__(self):
		lines = []
		for i, layer in enumerate(self.network):
			lines.append(Fore.YELLOW + f"===== Layer {i + 1} : {layer.n_inputs} -> {layer.n_neurons} =====")
			lines.append(Fore.GREEN + f"weights {layer.weights.shape} :")
			lines.append(Fore.WHITE + str(layer.weights))
			lines.append(Fore.GREEN + f"bias {layer.bias.shape} :")
			lines.append(Fore.WHITE + str(layer.bias))
			lines.append("")
		return "\n".join(lines)


	def create(self):
		input_size = 0
		for layer_size in self.layers_size:
			if input_size != 0:
				self.network.append(DenseLayer(input_size, layer_size))
			input_size = layer_size
		return

	def train(self, X, Y):
		# need to btach X here
		return self.forward(X, Y)

	def forward(self, X, Y):
		self.mean = np.mean(X, axis=0)
		self.std = np.std(X, axis=0)
		self.std[self.std == 0] = self.mean[self.std == 0]

		curr_input = X - self.mean / self.std

		for i in range(0, len(self.network) - 1):
			curr_input = self.network[i].forward(curr_input, u.relu)

		p = self.network[-1].forward(curr_input, u.softmax)

		p_binary = p[:, 1]

		print(u.accuracy(p_binary, Y))

		return self.loss(p_binary, Y, u.binary_cross_entropy)

	def loss(self, p, Y, loss_func):
		return loss_func(p, Y)

	def predict(self, X):

		# pour predict on fait passer un tour dans le neural network la data comme dans la fonction train mais sans le backprop
		pass
