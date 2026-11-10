import sys
import pandas as pd
from data_reader import Dataset
from NeuralNetwork import NeuralNetwork
import matplotlib as mpl


def main():
	df = pd.read_csv(sys.argv[1], header=None)
	if df is None:
		return

	dataset = Dataset(df)

	dataset.rand_test_split()

	data_shape = dataset.shape()

	network = NeuralNetwork([data_shape[1], 4, 4, 2])

	print(network.trainNetwork(dataset.X)[:10])

	# print(network)


if __name__ == "__main__":
	try:
		main()
	except (EOFError, KeyboardInterrupt):
		print()