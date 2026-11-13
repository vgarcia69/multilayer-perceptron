import sys
import pandas as pd
from data_reader import Dataset
from Model import Model
from DenseLayer import DenseLayer
import matplotlib as mpl
import numpy as np


def main():
	# generer le file de weight + biais entraine
	df = pd.read_csv("training", header=None)
	if df is None:
		return
	dataset = Dataset(df)

	X = dataset.data[:, 2:].astype(float)
	is_m = dataset.data[:, 1] == "M"
	Y = np.c_[~is_m, is_m].astype(int)

	input_shape = X.shape[1]
	hl1_shape = 4
	hl2_shape = 4
	output_shape = 2

	model = Model([
		DenseLayer(input_shape, hl1_shape, activation='relu', weights_initializer='he_uniform'),
		DenseLayer(hl1_shape, hl2_shape, activation='relu', weights_initializer='he_uniform'),
		DenseLayer(hl2_shape, output_shape, activation='softmax', weights_initializer='he_uniform')
	])

	model.train(X,Y, learning_rate=0.1, n_epoch=100)

	print(model)
	model.save()


if __name__ == "__main__":
	try:
		main()
	except (EOFError, KeyboardInterrupt):
		print()