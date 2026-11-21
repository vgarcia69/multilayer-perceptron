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
	return metrics

def load_graph(to_compare):
	for i, metrics in enumerate(to_compare):
		epochs = range(1, len(metrics["accuracy"]) + 1)

		plt.figure(figsize=(12, 5))

		plt.subplot(1, 2, 1)
		plt.plot(epochs, metrics["accuracy"], label='Training Accuracy')
		if metrics["val_accuracy"] is not None:
			plt.plot(epochs, metrics["val_accuracy"], label='Validation Accuracy')
		plt.title(f'Model {i+1} Accuracy')
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
		plt.legend()

		plt.subplot(1, 2, 2)
		plt.plot(epochs, metrics["loss"], label='Training Loss')
		if metrics["val_loss"] is not None:
			plt.plot(epochs, metrics["val_loss"], label='Validation Loss')
		plt.title(f'Model {i+1} Loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()

		plt.tight_layout()
		plt.show()

def main():
	to_compare = []

	for i in range(1, len(sys.argv)):
		df = load_json(sys.argv[i])
		to_compare.append(read_json(df))

	load_graph(to_compare)

if __name__ == "__main__":
	try:
		main()
	except (EOFError, KeyboardInterrupt, OSError) as e:
		print(e, file=sys.stderr)