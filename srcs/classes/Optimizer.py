import numpy as np

class Optimizer:
	def update(self, dW, dB, W, B, learning_rate):
		W -= dW * learning_rate
		B -= dB * learning_rate

class Adam(Optimizer):
	def __init__(self):
		self.t = 0
		self.beta1 = 0.9
		self.beta2 = 0.999
		self.momentW = 0 
		self.momentB = 0
		self.varianceW = 0
		self.varianceB = 0

	def update(self, dW, dB, W, B, learning_rate):
		epsylon = 1e-8

		self.t += 1

		self.momentW = self.beta1 * self.momentW + (1 - self.beta1) * dW
		self.varianceW = self.beta2 * self.varianceW + (1 - self.beta2) * dW ** 2

		self.momentB = self.beta1 * self.momentB + (1 - self.beta1) * dB
		self.varianceB = self.beta2 * self.varianceB + (1 - self.beta2) * dB ** 2
		
		mcW = self.momentW / (1 - self.beta1 ** self.t)
		vcW = self.varianceW / (1 - self.beta2 ** self.t)

		mcB = self.momentB / (1 - self.beta1 ** self.t)
		vcB = self.varianceB / (1 - self.beta2 ** self.t)
	
		W -= learning_rate * mcW / (np.sqrt(vcW) + epsylon)
		B -= learning_rate * mcB / (np.sqrt(vcB) + epsylon)
