import sys
from utils.io_helper import load_csv, to_float, check_labels
from classes.Dataset import Dataset
from classes.Model import Model
from classes.DenseLayer import DenseLayer
from classes.CallBack import ConsoleLogger, Graphics, EarlyStopping, History 
import numpy as np

def main():
	df = load_csv("data/training.csv")
	dataset = Dataset(df)

	X = to_float(dataset.data[:, 2:], "data/training.csv")
	check_labels(dataset.data[:, 1], "data/training.csv")
	is_m = dataset.data[:, 1] == "M"
	Y = np.c_[~is_m, is_m].astype(int)

	df = load_csv("data/validation.csv")
	dataset = Dataset(df)

	X_val = to_float(dataset.data[:, 2:], "data/validation.csv")
	check_labels(dataset.data[:, 1], "data/validation.csv")
	is_m_val = dataset.data[:, 1] == "M"
	Y_val = np.c_[~is_m_val, is_m_val].astype(int)

	if X_val.shape[1] != X.shape[1]:
		raise OSError(f"Error: 'data/validation.csv' has {X_val.shape[1]} features, 'data/training.csv' has {X.shape[1]}")

	input_shape = X.shape[1]
	hl1_shape = 24
	hl2_shape = 24
	hl3_shape = 24
	output_shape = 2

	history = History()

	model = Model([
		DenseLayer(input_shape, hl1_shape, activation='sigmoid', weights_initializer='he_uniform', optimizer='adam'),
		DenseLayer(hl1_shape, hl2_shape, activation='sigmoid', weights_initializer='he_uniform', optimizer='adam'),
		DenseLayer(hl2_shape, hl3_shape, activation='sigmoid', weights_initializer='he_uniform', optimizer='adam'),
		DenseLayer(hl3_shape, output_shape, activation='softmax', weights_initializer='he_uniform', optimizer='adam')
	])

	model.train(X, Y, X_val=X_val, Y_val=Y_val,
				callbacks=[ConsoleLogger(), history, Graphics(history), EarlyStopping()],
				loss="categorical_cross_entropy",
	            learning_rate=0.0001,
				batch_size=8,
				epochs=100)

	model.save()
	history.save()

if __name__ == "__main__":
	try:
		main()
	except (EOFError, KeyboardInterrupt, OSError) as e:
		print(e, file=sys.stderr)