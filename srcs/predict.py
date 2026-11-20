import sys
import json
from utils.io_helper import load_csv, to_float, check_labels
from classes.Dataset import Dataset
from classes.Model import Model
import numpy as np


def main():
	df = load_csv("data/validation.csv")
	dataset = Dataset(df)

	X = to_float(dataset.data[:, 2:], "data/validation.csv")
	check_labels(dataset.data[:, 1], "data/validation.csv")
	is_m = dataset.data[:, 1] == "M"
	Y = np.c_[~is_m, is_m].astype(int)

	try:
		model = Model.load()
	except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as e:
		print(f"Error: cannot load model: {e}", file=sys.stderr)
		sys.exit(1)

	if X.shape[1] != model.layers[0].n_inputs:
		print(f"Error: 'data/validation.csv' has {X.shape[1]} features, model expects {model.layers[0].n_inputs}", file=sys.stderr)
		sys.exit(1)

	loss = model.evaluate(X, Y)
	print(f"Validation loss: {loss:.4f}")


if __name__ == "__main__":
	try:
		main()
	except (EOFError, KeyboardInterrupt):
		print()