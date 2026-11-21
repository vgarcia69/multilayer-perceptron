from matplotlib import pyplot as plt
from classes.CallBacks.CallBack import CallBack


class Graphics(CallBack):
	def __init__(self, history):
		self.history = history

	def on_train_end(self, logs=None):
		train_loss = self.history["loss"]
		train_accuracy = self.history["accuracy"]
		val_accuracy = self.history["val_accuracy"]
		val_loss = self.history["val_loss"]

		if val_loss is None or val_accuracy is None:
			return

		epochs = range(1, len(train_loss) + 1)

		plt.figure(figsize=(12, 5))

		plt.subplot(1, 2, 1)
		plt.plot(epochs, train_accuracy, label='Training Accuracy')
		plt.plot(epochs, val_accuracy, label='Validation Accuracy')
		plt.title('Model Accuracy')
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
		plt.legend()

		plt.subplot(1, 2, 2)
		plt.plot(epochs, train_loss, label='Training Loss')
		plt.plot(epochs, val_loss, label='Validation Loss')
		plt.title(f'Model Loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()

		plt.tight_layout()
		plt.show()