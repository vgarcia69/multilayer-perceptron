import os
import sys
import matplotlib.pyplot as plt

from utils.io_helper import load_json


def read_json(df):
	metrics = {
		"accuracy": df.get("accuracy"),
		"loss": df.get("loss"),
		"val_accuracy": df.get("val_accuracy"),
		"val_loss": df.get("val_loss")
	}
	if metrics["accuracy"] is None or metrics["loss"] is None:
		raise OSError("Error: history file must contain 'accuracy' and 'loss'")
	return metrics


def model_name(path):
	name = os.path.splitext(os.path.basename(path))[0]
	return name.removeprefix("history_")


def plot_metric(ax, to_compare, metric, title):
	for i, (name, metrics) in enumerate(to_compare):
		color = f"C{i}"
		epochs = range(1, len(metrics[metric]) + 1)

		ax.plot(epochs, metrics[metric], color=color, linestyle="-", label=f"{name} (train)")
		if metrics[f"val_{metric}"] is not None:
			ax.plot(epochs, metrics[f"val_{metric}"], color=color, linestyle="--", label=f"{name} (validation)")

	ax.set_title(title)
	ax.set_xlabel("Epochs")
	ax.set_ylabel(title)
	ax.grid(alpha=0.3)
	ax.legend()


def load_graph(to_compare):
	fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))

	plot_metric(ax_acc, to_compare, "accuracy", "Accuracy")
	plot_metric(ax_loss, to_compare, "loss", "Loss")

	fig.suptitle("Model comparison")
	fig.tight_layout()
	plt.show()


def main():
	if len(sys.argv) < 2:
		print("Usage: python compare.py history_a.json [history_b.json ...]", file=sys.stderr)
		sys.exit(1)

	to_compare = []
	for path in sys.argv[1:]:
		df = load_json(path)
		to_compare.append((model_name(path), read_json(df)))

	load_graph(to_compare)


if __name__ == "__main__":
	try:
		main()
	except (EOFError, KeyboardInterrupt, OSError) as e:
		print(e, file=sys.stderr)
