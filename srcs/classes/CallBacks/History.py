import json
from classes.CallBacks.CallBack import CallBack


class History(CallBack):
	def on_train_begin(self, epochs):
		self.history = {}
		self.epoch = []

	def on_epoch_end(self, epoch, logs=None):
		self.epoch.append(epoch + 1)
		for name, value in logs.items():
			self.history.setdefault(name, []).append(float(value))

	def save(self, filename="data/history.json"):
		data = {}
		data.update(self.history)

		try:
			with open(filename, "w") as f:
				json.dump(data, f, indent=2)
		except PermissionError:
			print(f"Error : history has not been saved, openning {filename} failed")

	def __getitem__(self, key):
		return self.history[key]
