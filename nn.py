import numpy as np
from data import *

class BCE:
	def cost(self, predictions, target):
		return -np.sum( target*np.log(predictions) + (1-target)*np.log(1-predictions) )/predictions.shape[0]

	def d_cost(self, predictions, target):
		return -(target/predictions - (1-target)/(1-predictions))
	
class NeuralNetwork:
	def __init__(self, cuda=False):
		self.layers = list()

		#binary cross entropy
		self.bce = BCE()
		self.cuda = cuda

	def addLayer(self,layer):
		self.layers.append(layer)

	def init_layers(self):
		for layer in self.layers:
			layer.reset()

	def forward(self, Z):
		for layer in self.layers:
			Z = layer.forward(Z, self.cuda)
		return Z

	def backprop(self, predictions, target):
		error = self.bce.d_cost(predictions, target)
		for layer in self.layers[::-1]:
			error = layer.backprop(error, self.cuda)
		return error

	def get_layer(self, name):
		for layer in self.layers:
			if name == layer.name:
				return layer

	def train(self, data, epochs = 400, log = False):
		for i in range(0,epochs):
			cost = 0
			for batch in data:
				Y_hat = self.forward(batch["x"])
				self.backprop(Y_hat, batch["y"])
				cost += self.bce.cost(Y_hat, batch["y"])
			if log:
				print("Epoch={:d}, cost={:.2f}".format(i,cost))

	def predict(self, data):
		Y_hat = self.forward(data["x"])
		acc = np.sum(to_label(Y_hat) == data["y"])/len(data["y"])
		return Y_hat, acc