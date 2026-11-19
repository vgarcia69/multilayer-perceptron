import sys
import numpy as np
import pandas as pd


def load_csv(path):
	try:
		return pd.read_csv(path, header=None)
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
