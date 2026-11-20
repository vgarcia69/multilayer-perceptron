from colorama import Fore, Style
from matplotlib import pyplot as plt


class CallBack():
	def set_model(self, model):
		self.model = model
	def on_train_begin(self, logs=None):
		return
	def on_epoch_end(self, epoch, logs=None):
		return
	def on_train_end(self, logs=None):
		return

class ConsoleLogger(CallBack):
	def __init__(self, every=1, colors=True):
		if every < 1:
			raise ValueError(f"ConsoleLogger: 'every' must be >= 1 (got {every})")
		self.every = every
		self.colors = colors

	def on_train_begin(self, epochs):
		self.epochs = epochs
		self.width = len(str(epochs))

	def on_epoch_end(self, epoch, metrics):
		parts = [self.paint(Fore.WHITE, f"epoch {epoch + 1:0{self.width}d}/{self.epochs}")]
		value = metrics["accuracy"]
		parts.append(self.paint(Fore.CYAN, f'{"accuracy"}: {value:.4f}'))
		value = metrics["loss"]
		parts.append(self.paint(Fore.CYAN, f'{"loss"}: {value:.4f}'))

		if "val_loss" in metrics and "val_accuracy" in metrics:
			value_val = metrics["val_loss"]
			parts.append(self.paint(Fore.GREEN, f'{"val_loss"}: {value_val:.4f}'))
			value_val = metrics["val_accuracy"]
			parts.append(self.paint(Fore.GREEN, f'{"val_accuracy"}: {value_val:.4f}'))

		print(" - ".join(parts))


	def paint(self, color, text):
		if not self.colors:
			return text
		return f"{color}{text}{Style.RESET_ALL}"

class History(CallBack):
	pass

class Graphics(CallBack):
	def __init__(self):
		pass

	def on_train_begin(self, epochs):
		self.epochs = epochs
		self.accuracy_train = []
		self.loss_train = []
		self.accuracy_val = []
		self.loss_val = []

	def on_epoch_end(self, epoch, metrics):
		self.accuracy_train.append(metrics["accuracy"])
		self.loss_train.append(metrics["loss"])
		if "val_accuracy" in metrics and "val_loss" in metrics:
			self.accuracy_val.append(metrics["val_accuracy"])
			self.loss_val.append(metrics["val_loss"])

	def on_train_end(self, logs=None):
		plt.figure(figsize=(8, 5))
		plt.title("Loss")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.plot(self.loss_train, label="Training", color="blue", linewidth=2)
		plt.plot(self.loss_val, label="Validation", color="red", linewidth=2)
		plt.legend()

		plt.figure(figsize=(8, 5))
		plt.title("Learning Curve")
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy")
		plt.plot(self.accuracy_train, label="Training", color="blue", linewidth=2)
		plt.plot(self.accuracy_val, label="Validation", color="red", linewidth=2)
		plt.legend()

		plt.show()

class EarlyStopping(CallBack):
	def __init__(self, patience=10):
		self.patience = patience

	def on_train_begin(self, epochs):
		self.best_loss = float('inf')
		self.wait = 0
		self.best_weights = None
		self.best_bias = None

	def on_epoch_end(self, epoch, logs=None):
		current_loss = logs.get('val_loss')
		if current_loss is None:
			return False

		if current_loss < self.best_loss:
			self.best_loss = current_loss
			self.wait = 0
			self.best_weights = self.model.get_weights()
			self.best_bias = self.model.get_bias()
		else:
			self.wait += 1
			if self.wait >= self.patience:
				print(f"Early stopping at epoch {epoch + 1}")
				self.model.set_weights(self.best_weights)
				self.model.set_bias(self.best_bias)
				return True
		return False
