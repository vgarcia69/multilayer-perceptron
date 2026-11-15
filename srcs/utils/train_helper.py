import numpy as np

def split_in_batches(X, Y, batch_size):
	"""Decoupe X et Y en morceaux de batch_size lignes.
	Renvoie une liste de paires (X_batch, Y_batch), chacune etant un tableau numpy 2D."""
	batches = []
	for start in range(0, len(X), batch_size):
		end = start + batch_size
		batches.append((X[start:end], Y[start:end]))
	return batches

def shuffle_order(X, Y):
	"""Melange les lignes de X et Y dans le meme ordre (chaque patient garde son label)."""
	order = np.random.permutation(len(X))
	return X[order], Y[order]
