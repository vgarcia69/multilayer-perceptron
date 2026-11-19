import numpy as np

def split_in_batches(X, Y, batch_size):
	batches = []
	for start in range(0, len(X), batch_size):
		end = start + batch_size
		batches.append((X[start:end], Y[start:end]))
	return batches

def shuffle_order(X, Y):
	order = np.random.permutation(len(X))
	return X[order], Y[order]
