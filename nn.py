import numpy as np

class BCE:
	def cost(self, predictions, target):
		return -np.sum( target*np.log(predictions) + (1-target)*np.log(1-predictions) )/predictions.shape[0]

	def d_cost(self, predictions, target):
		return -(target/predictions - (1-target)/(1-predictions))
	
class NeuralNetwork:
	def __init__(self):
		self.layers = list()
		self.bce = BCE()

	def addLayer(self,layer):
		self.layers.append(layer)

	def init_layers(self):
		for layer in self.layers:
			layer.reset()

	def forward(self, Z):
		for layer in self.layers:
			Z = layer.forward(Z)
		return Z

	def backprop(self, predictions, target):
		error = self.bce.d_cost(predictions, target)
		for layer in self.layers[::-1]:
			error = layer.backprop(error)
		return error

	def get_layer(self, name):
		for layer in self.layers:
			if name == layer.name:
				return layer