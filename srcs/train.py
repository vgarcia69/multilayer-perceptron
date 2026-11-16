import pandas as pd
from classes.Dataset import Dataset
from classes.Model import Model
from classes.DenseLayer import DenseLayer
import numpy as np
import matplotlib.pyplot as plt

def main():
	df = pd.read_csv("data/training.csv", header=None)
	if df is None:
		return
	dataset = Dataset(df)

	X = dataset.data[:, 2:].astype(float)
	is_m = dataset.data[:, 1] == "M"
	Y = np.c_[~is_m, is_m].astype(int)

	df = pd.read_csv("data/validation.csv", header=None)
	if df is None:
		return
	dataset = Dataset(df)

	X_val = dataset.data[:, 2:].astype(float)
	is_m_val = dataset.data[:, 1] == "M"
	Y_val = np.c_[~is_m_val, is_m_val].astype(int)

	input_shape = X.shape[1]
	hl1_shape = 24
	hl2_shape = 24
	hl3_shape = 24
	output_shape = 2

	model = Model([
		DenseLayer(input_shape, hl1_shape, activation='sigmoid', weights_initializer='he_uniform'),
		DenseLayer(hl1_shape, hl2_shape, activation='sigmoid', weights_initializer='he_uniform'),
		DenseLayer(hl2_shape, hl3_shape, activation='sigmoid', weights_initializer='he_uniform'),
		DenseLayer(hl3_shape, output_shape, activation='softmax', weights_initializer='he_uniform')
	])

	model.train(X, Y, X_val=X_val, Y_val=Y_val, loss="categorical_cross_entropy", learning_rate=0.0314, batch_size=8, epochs=100)

	model.save()

	model.show_learning_curve()


if __name__ == "__main__":
	try:
		main()
	except (EOFError, KeyboardInterrupt):
		print()