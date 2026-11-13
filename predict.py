import pandas as pd
from data_reader import Dataset
from Model import Model
from DenseLayer import DenseLayer
import matplotlib as mpl
import numpy as np


def main():
	df = pd.read_csv("validation", header=None)
	if df is None:
		return
	dataset = Dataset(df)

	X = dataset.data[:, 2:].astype(float)
	is_m = dataset.data[:, 1] == "M"
	Y = np.c_[~is_m, is_m].astype(int)

	model = Model.load()

	result= model.predict(X)

	print(model.accuracy.forward(result, Y))


if __name__ == "__main__":
	try:
		main()
	except (EOFError, KeyboardInterrupt):
		print()