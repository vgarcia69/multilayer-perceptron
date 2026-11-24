import sys
import numpy as np

from utils.io_helper import load_csv, to_float, check_labels, parse_config, get_XY

from classes.Model import Model
from classes.DenseLayer import DenseLayer
from classes.CallBacks.ConsoleLogger import ConsoleLogger
from classes.CallBacks.Graphics import Graphics 
from classes.CallBacks.EarlyStopping import EarlyStopping
from classes.CallBacks.History import History
from classes.CallBacks.Metrics import Metrics

def main():
	if len(sys.argv) != 2:
		print(f"Usage: python {sys.argv[0]} <config.json>")
		sys.exit(1)

	config = parse_config(sys.argv[1])

	X, Y = get_XY("data/training.csv")
	X_val, Y_val = get_XY("data/validation.csv")

	if X_val.shape[1] != X.shape[1]:
		raise OSError(f"Error: validation has {X_val.shape[1]} features, training has {X.shape[1]}")

	opt = config["optimizer"]
	init = config["weights_initializer"]
	act  = config["activations"].lower()
	shapes = [X.shape[1]] + config["layer"]

	layers = [
		DenseLayer(n_in, n_out, activation=act, weights_initializer=init, optimizer=opt)
		for n_in, n_out in zip(shapes[:-1], shapes[1:])
	]
	layers.append(DenseLayer(shapes[-1], Y.shape[1], activation="softmax", weights_initializer=init, optimizer=opt))
	model = Model(layers)

	metrics = Metrics(X, Y, metrics_required=["accuracy", "loss"] + config["optionnal_metrics"])
	metrics_val = Metrics(X_val, Y_val, prefix="val_")
	history = History()

	optionnal = config["optionnal_callbacks"]
	callbacks = [metrics, metrics_val]
	if "console_logger" in optionnal:
		callbacks.append(ConsoleLogger())
	callbacks.append(history)
	if "graphics" in optionnal:
		callbacks.append(Graphics(history))
	if "early_stopping" in optionnal:
		callbacks.append(EarlyStopping())

	model.train(X, Y,
				callbacks=callbacks,
				loss=config["loss"],
				learning_rate=config["learning_rate"],
				batch_size=config["batch_size"],
				epochs=config["epochs"])

	model.save()

	history.save(f"data/history_{optimizer}.json")

if __name__ == "__main__":
	try:
		main()
	except (EOFError, KeyboardInterrupt, OSError) as e:
		print(e, file=sys.stderr)