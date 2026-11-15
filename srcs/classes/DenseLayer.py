import numpy as np
import classes.Activation as act

activation_functions = {
	"sigmoid": act.Sigmoid,
	"relu": act.ReLU,
	"softmax": act.Softmax
}

weights_initializers = {
	"random": lambda n_inputs, n_neurons: np.random.randn(n_inputs, n_neurons) * 0.1,
	"he_uniform": lambda n_inputs, n_neurons: np.random.randn(n_inputs, n_neurons) * 0.1, # a changer
}


class DenseLayer:
	""" Hidden Layers + Final Layer """

	def __init__(self, n_inputs, n_neurons, activation, weights_initializer="random"):
		"""TEST"""
		self.n_neurons = n_neurons
		self.n_inputs = n_inputs
		self.weights = weights_initializers[weights_initializer](n_inputs, n_neurons)
		self.activation_name = activation
		self.activation = activation_functions[activation]()
		self.bias = np.zeros((1, self.n_neurons))
		return

	def __len__(self):
		return self.n_neurons


	def forward(self, X):
		self.X = X
		self.Z = self.X @ self.weights + self.bias
		self.A = self.activation.forward(self.Z)

		return self.A


	def backward(self, dA, learning_rate, Y=0):
		# backward activation, (derivee de la fonction)
		if type(self.activation) is act.Softmax:
			dZ = (self.A - Y) / self.X.shape[0]
		else:
			dZ = dA * self.activation.backward(self.Z)

		# backward weights, bias and input
		dW = self.X.T @ dZ
		dB = np.sum(dZ, axis=0, keepdims=True)
		dX = dZ @ self.weights.T

		self.weights -= dW * learning_rate
		self.bias -= dB * learning_rate

		return dX

	def save(self):
		return {
			"weights": self.weights.tolist(),
			"bias": self.bias.tolist(),
			"activation": self.activation_name,
			"n_neurons": self.n_neurons,
			"n_inputs": self.n_inputs
		}

	@staticmethod
	def load(data):
		layer = DenseLayer(data["n_inputs"], data["n_neurons"], activation=data["activation"])
		layer.weights = np.array(data["weights"])
		layer.bias = np.array(data["bias"])
		return layer
