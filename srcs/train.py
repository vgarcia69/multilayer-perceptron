import sys
from utils.io_helper import load_csv, to_float, check_labels
from classes.Dataset import Dataset
from classes.Model import Model
from classes.DenseLayer import DenseLayer
from classes.CallBacks.ConsoleLogger import ConsoleLogger
from classes.CallBacks.Graphics import Graphics 
from classes.CallBacks.EarlyStopping import EarlyStopping
from classes.CallBacks.History import History
from classes.CallBacks.Metrics import Metrics
import numpy as np

def get_XY(path):
	df = load_csv(path)
	dataset = Dataset(df)

	X = to_float(dataset.data[:, 2:], path)
	check_labels(dataset.data[:, 1], path)
	is_m = dataset.data[:, 1] == "M"
	Y = np.c_[~is_m, is_m].astype(int)

	return X, Y


def main():
	X, Y = get_XY("data/training.csv")
	X_val, Y_val = get_XY("data/validation.csv")

	if X_val.shape[1] != X.shape[1]:
		raise OSError(f"Error: validation has {X_val.shape[1]} features, training has {X.shape[1]}")

	input_shape = X.shape[1]
	hl1_shape = 24
	hl2_shape = 24
	output_shape = 2
	optimizer = "gradient_descent"

	metrics = Metrics(X, Y, metrics_required=["accuracy", "loss", "recall", "precision"])
	metrics_val = Metrics(X_val, Y_val, prefix="val_")
	history = History()
	console_logger = ConsoleLogger()
	graphics = Graphics(history)
	early_stopping = EarlyStopping()

	model = Model([
		DenseLayer(input_shape, hl1_shape, activation='sigmoid', weights_initializer='he_uniform', optimizer=optimizer),
		DenseLayer(hl1_shape, hl2_shape, activation='sigmoid', weights_initializer='he_uniform', optimizer=optimizer),
		DenseLayer(hl2_shape, output_shape, activation='softmax', weights_initializer='he_uniform', optimizer=optimizer)
	])

	model.train(X, Y,
				callbacks=[metrics, metrics_val, console_logger, history, graphics, early_stopping],
				loss="categorical_cross_entropy",
	            learning_rate=0.001,
				batch_size=1,
				epochs=1000)

	model.save()

	history.save(f"data/history_{optimizer}.json")

if __name__ == "__main__":
	try:
		main()
	except (EOFError, KeyboardInterrupt, OSError) as e:
		print(e, file=sys.stderr)