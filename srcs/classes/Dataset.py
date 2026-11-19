import numpy as np
import sys


class Dataset:
	def __init__(self, df):
		self.df = df

		self.data = df.to_numpy()

		if len(self.data) < 1:
			print("Dataset too small to be used")
			sys.exit(1)


	def randomize_order(self):
		self.data = self.df.sample(frac=1, random_state=1).to_numpy()

	def rand_test_split(self):
		self.randomize_order()

		size = len(self.data)
		limit = int(np.round(size * 0.8))

		training_datas = self.data[0:limit]
		validation_datas = self.data[limit: size]

		np.savetxt("data/validation.csv", validation_datas, "%s", ',')
		np.savetxt("data/training.csv", training_datas, "%s", ',')