from colorama import Fore, Style
from classes.CallBacks.CallBack import CallBack


class ConsoleLogger(CallBack):
	def __init__(self, colors=True):
		self.colors = colors

	def on_train_begin(self, epochs):
		self.epochs = epochs
		self.width = len(str(epochs))

	def on_epoch_end(self, epoch, metrics):
		parts = [self.paint(Fore.WHITE, f"epoch {epoch + 1:0{self.width}d}/{self.epochs}")]

		if "accuracy" in metrics and "loss" in metrics: 
			value = metrics["accuracy"]
			parts.append(self.paint(Fore.CYAN, f'{"accuracy"}: {value:.4f}'))
			value = metrics["loss"]
			parts.append(self.paint(Fore.CYAN, f'{"loss"}: {value:.4f}'))

		if "val_loss" in metrics and "val_accuracy" in metrics:
			value_val = metrics["val_loss"]
			parts.append(self.paint(Fore.GREEN, f'{"val_loss"}: {value_val:.4f}'))
			value_val = metrics["val_accuracy"]
			parts.append(self.paint(Fore.GREEN, f'{"val_accuracy"}: {value_val:.4f}'))
		
		for k, v in metrics.items():
			if k not in ("accuracy", "loss", "val_loss", "val_accuracy"):
				parts.append(self.paint(Fore.MAGENTA, f"{k}: {v:.4f}"))


		print(" - ".join(parts))


	def paint(self, color, text):
		if not self.colors:
			return text
		return f"{color}{text}{Style.RESET_ALL}"
