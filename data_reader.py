import pandas as pd
import numpy as np
import utils as u
import sys


class Dataset:
	""" READ THE CSV FILE
	AND DO THINGS """
	def __init__(self, df):
		self.df = df

		self.data = df.to_numpy()

		if len(self.data) < 1:
			print("ERREUR")
			sys.exit(1)


	def shape(self):
		return self.data.shape

	def randomize_order(self):
		self.data = self.df.sample(frac=1, random_state=1).to_numpy()

	def rand_test_split(self):
		""" Split le data set en 2 de 20 et 80 % pour analyser la precision de l'algo """

		self.randomize_order()

		size = len(self.data)
		limit = int(np.round(size * 0.8))

		training_datas = self.data[0:limit]
		validation_datas = self.data[limit: size]

		np.savetxt("validation", validation_datas, "%s", ',')
		np.savetxt("training", training_datas, "%s", ',')
		

	def patch_null(self, mean):
		"""Convertir les NaN avec la moyenne donnee"""
		np.nan_to_num(self.data[:, :2], nan=mean, copy=False)

