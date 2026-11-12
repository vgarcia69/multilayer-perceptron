import numpy as np
import Activation as act
from colorama import Fore, Style

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
	"""DESNE LAYERE"""

	def __init__(self, n_inputs, n_neurons, activation=None, weights_initializer="random"):
		"""TEST"""
		self.n_neurons = n_neurons
		self.n_inputs = n_inputs
		self.weights = weights_initializers[weights_initializer](n_inputs, n_neurons)
		self.activation = activation_functions[activation]()
		self.bias = np.zeros((1, self.n_neurons))
		return


	def __len__(self):
		return self.n_neurons

	def __str__(self):
		"""Affiche la couche et, si un forward a eu lieu, ce qui s'y est passe"""
		R = Style.RESET_ALL
		GRIS = Fore.LIGHTBLACK_EX
		forme = lambda a: Fore.CYAN + str(a.shape) + R

		lignes = []
		lignes.append(Fore.YELLOW + Style.BRIGHT
			+ f"\u250c\u2500 DenseLayer {self.n_inputs} \u2192 {self.n_neurons} " + "\u2500" * 20 + R)
		lignes.append(f"\u2502 W {forme(self.weights)}   b {forme(self.bias)}")

		if not hasattr(self, "A"):
			lignes.append("\u2502 " + GRIS + "(pas encore de forward)" + R)
			lignes.append(Fore.YELLOW + "\u2514" + "\u2500" * 40 + R)
			return "\n".join(lignes)

		X, Z, A = self.X, self.Z, self.A

		# Les formes sont-elles celles attendues ?
		attendu = (X.shape[0], self.n_neurons)
		if Z.shape == attendu and A.shape == attendu:
			verdict = Fore.GREEN + "\u2713" + R
		else:
			verdict = Fore.RED + Style.BRIGHT + f"\u2717 attendu {attendu}" + R
		lignes.append(f"\u2502 X {forme(X)} @ W + b \u2192 Z {forme(Z)} \u2192 A {forme(A)}  {verdict}")

		# Quelques chiffres pour voir si les valeurs restent raisonnables
		for nom, M in (("Z", Z), ("A", A)):
			if np.isnan(M).any():
				lignes.append(f"\u2502 {nom}  " + Fore.RED + Style.BRIGHT + "contient des NaN !" + R)
			else:
				lignes.append(f"\u2502 {nom}  " + GRIS + "min " + R + f"{M.min():8.3f}"
					+ GRIS + "  moy " + R + f"{M.mean():8.3f}"
					+ GRIS + "  max " + R + f"{M.max():8.3f}")

		# La sortie du premier patient, en petites barres
		barres = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
		patient = A[0][:30]
		plus_grand = np.abs(patient).max()
		dessin = ""
		for v in patient:
			niveau = 0 if plus_grand == 0 else int(abs(v) / plus_grand * (len(barres) - 1))
			couleur = GRIS if v == 0 else (Fore.GREEN if v > 0 else Fore.MAGENTA)
			dessin += couleur + barres[niveau] + R
		valeurs = np.array2string(patient, precision=2, suppress_small=True, max_line_width=1000)
		lignes.append(f"\u2502 patient 0 : {dessin}  " + GRIS + valeurs + R)

		# Neurones qui sortent 0 pour tous les patients
		morts = int((A == 0).all(axis=0).sum())
		couleur = Fore.GREEN if morts == 0 else Fore.YELLOW
		lignes.append("\u2502 " + couleur + f"neurones toujours a 0 : {morts}/{self.n_neurons}" + R)

		lignes.append(Fore.YELLOW + "\u2514" + "\u2500" * 40 + R)
		return "\n".join(lignes)


	def forward(self, X):
		if self.activation is None:
			return X

		self.X = X
		self.Z = self.X @ self.weights + self.bias
		self.A = self.activation.forward(self.Z)

		return self.A


	def backward(self, A, learning_rate, Y=0):
		if type(self.activation) is act.Softmax:
			dZ = (self.A - Y) / self.X.shape[0]
		else:
			dZ = A * self.activation.backward(self.Z)

		dW = self.X.T @ dZ
		dB = np.sum(dZ, axis=0, keepdims=True)
		dX = dZ @ self.weights.T
		self.weights -= dW * learning_rate
		self.bias -= dB * learning_rate

		return dX