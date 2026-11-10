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

		self.createNetwork()
		return


	def __str__(self):
		lines = []
		for i, layer in enumerate(self.network):
			lines.append(Fore.YELLOW + f"===== Layer {i + 1} : {layer.input_size} -> {layer.current_size} =====")
			lines.append(Fore.GREEN + f"weights {layer.weights.shape} :")
			lines.append(Fore.WHITE + str(layer.weights))
			lines.append(Fore.GREEN + f"bias {layer.bias.shape} :")
			lines.append(Fore.WHITE + str(layer.bias))
			lines.append("")
		return "\n".join(lines)


	def createNetwork(self):
		firstLayer = False
		input_size = 0
		for layer_size in self.layers_size:
			if firstLayer is False:
				firstLayer = True
			else:
				self.network.append(DenseLayer(input_size, layer_size))
			input_size = layer_size
		return


	def trainNetwork(self, X):
		self.mean = X.mean(axis=0)
		self.std = X.std(axis=0)
		self.std[self.std == 0] = 1

		val = self.normalize(X)

		# should batch it next
		for i in range(0, len(self.network) - 1):
			val = self.network[i].forward(val, u.relu)

		return self.network[-1].forward(val, u.softmax)

	def normalize(self, X):
		return (X - self.mean) / self.std
