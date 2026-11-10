import pandas as pd
import numpy as np
import utils as u
import sys


class Dataset:
	""" READ THE CSV FILE
	AND DO THINGS """
	def __init__(self, df):
		self.df = df

		self.X = df.iloc[:, 2:].to_numpy()
		self.Y = (df[1] == "M").to_numpy(dtype=int)

		if len(self.X) < 1 or len(self.Y) < 1:
			print("ERREUR")
			sys.exit(1)


	def shape(self):
		return self.X.shape

	def randomize_order(self):
		sampled_df = self.df.sample(frac=1, random_state=1)

		self.X = sampled_df.iloc[:, 2:].to_numpy()
		self.Y = (sampled_df[1] == "M").to_numpy(dtype=int)



	def rand_test_split(self):
		""" Split le data set en 2 de 20 et 80 % pour analyser la precision de l'algo """

		self.randomize_order()

		size = len(self.X)
		limit = int(np.round(size * 0.8))

		training_datas = self.X[0:limit]
		validation_datas = self.X[limit: size]

		np.savetxt("validation", validation_datas, "%g", ',')
		np.savetxt("training", training_datas, "%g", ',')


		# il va falloir sauvegarde le Y aussi ici, retourner a df plutot que X ?
		

	def patch_null(self, mean):
		"""Convertir les NaN avec la moyenne donnee"""
		np.nan_to_num(self.X, nan=mean, copy=False)

