import sys
import pandas as pd
from data_reader import Dataset
from NeuralNetwork import NeuralNetwork
import matplotlib as mpl


def main():
	#generer le training/ validation
	df = pd.read_csv(sys.argv[1], header=None)
	if df is None:
		return

	dataset = Dataset(df)

	dataset.rand_test_split()

	# dataset sera remplace par le fichier de training et le main sera split en 3 parties
	# generer le file de weight + biais entraine
	X = dataset.data[:, 2:].astype(float)
	Y = (dataset.data[:, 1] == "M").astype(int)

	network = NeuralNetwork([X.shape[1], 4, 4, 2])

	print(network.train(X, Y))
	# predict


if __name__ == "__main__":
	try:
		main()
	except (EOFError, KeyboardInterrupt):
		print()