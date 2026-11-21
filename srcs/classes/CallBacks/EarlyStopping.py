from classes.CallBacks.CallBack import CallBack


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
