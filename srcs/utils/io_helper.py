import sys
import json
import numpy as np
import pandas as pd
from classes.DenseLayer import activation_functions, weights_initializers, optimizer_functions
from classes.Model import loss_functions
from classes.Dataset import Dataset


def load_csv(path):
	try:
		return pd.read_csv(path, header=None)
	except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
		print(f"Error: cannot read '{path}': {e}", file=sys.stderr)
		sys.exit(1)


def load_json(path):
	try:
		return pd.read_json(path)
	except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
		print(f"Error: cannot read '{path}': {e}", file=sys.stderr)
		sys.exit(1)


def to_float(values, path):
	if values.shape[1] == 0:
		print(f"Error: '{path}' has no feature columns (expected: id, label, features...)", file=sys.stderr)
		sys.exit(1)
	try:
		X = values.astype(float)
	except ValueError as e:
		print(f"Error: '{path}' contains non-numeric features: {e}", file=sys.stderr)
		sys.exit(1)

	bad_lines = np.where(~np.isfinite(X).all(axis=1))[0] + 1
	if len(bad_lines) > 0:
		print(f"Error: '{path}' has missing or infinite features at line(s): {bad_lines[:5].tolist()}", file=sys.stderr)
		sys.exit(1)
	return X


def check_labels(labels, path):
	bad_lines = np.where(~np.isin(labels, ["M", "B"]))[0] + 1
	if len(bad_lines) > 0:
		print(f"Error: '{path}' has invalid labels (expected 'M' or 'B') at line(s): {bad_lines[:5].tolist()}", file=sys.stderr)
		sys.exit(1)


def get_XY(path):
	df = load_csv(path)
	dataset = Dataset(df)

	X = to_float(dataset.data[:, 2:], path)
	check_labels(dataset.data[:, 1], path)
	is_m = dataset.data[:, 1] == "M"
	Y = np.c_[~is_m, is_m].astype(int)

	return X, Y


def parse_config(path="data/config.json"):
	expected = {
		"layer": list,
		"activations": str,
		"loss": str,
		"weights_initializer": str,
		"optimizer": str,
		"epochs": int,
		"learning_rate": (int, float),
		"batch_size": int,
		"optionnal_metrics": list,
		"optionnal_callbacks": list
	}

	try:
		with open(path) as f:
			config = json.load(f)
	except json.JSONDecodeError as e:
		raise OSError(f"Error: '{path}' is not valid JSON: {e}")

	if not isinstance(config, dict):
		raise OSError(f"Error: '{path}' must contain a JSON object")

	for key, kind in expected.items():
		if key not in config:
			raise OSError(f"Error: '{path}' is missing key '{key}'")
		if not isinstance(config[key], kind) or isinstance(config[key], bool):
			raise OSError(f"Error: '{path}': '{key}' has wrong type")

	if not config["layer"] or not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in config["layer"]):
		raise OSError(f"Error: '{path}': 'layer' must be a non-empty list of positive integers")
	if config["epochs"] <= 0 or config["batch_size"] <= 0 or config["learning_rate"] <= 0:
		raise OSError(f"Error: '{path}': 'epochs', 'batch_size' and 'learning_rate' must be positive")

	choices = {
		"activations": (config["activations"].lower(), activation_functions),
		"weights_initializer": (config["weights_initializer"], weights_initializers),
		"optimizer": (config["optimizer"], optimizer_functions),
		"loss": (config["loss"], loss_functions),
	}
	for key, (value, allowed) in choices.items():
		if value not in allowed:
			raise OSError(f"Error: '{path}': '{key}' must be one of {list(allowed)}")

	if "softmax" in config["activations"]:
		raise OSError(f"Error: only relu and sigmoid are supported in hidden layers")

	allowed_callbacks = ["console_logger", "graphics", "early_stopping"]
	for name in config["optionnal_callbacks"]:
		if name not in allowed_callbacks:
			raise OSError(f"Error: '{path}': 'optionnal_callbacks' entries must be in {allowed_callbacks}")

	return config