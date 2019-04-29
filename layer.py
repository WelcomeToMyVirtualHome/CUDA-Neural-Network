import numpy as np
from numba import cuda

class Layer:
	def __init__(self, name):
		self.name = name

	def forward(self, A, cuda = False):
		raise NotImplemented

	def backprop(self, dZ, cuda = False):
		raise NotImplemented

	def reset(self):
		raise NotImplemented

class SigmoidLayer(Layer):
	def __init__(self, name):
		super().__init__(name)
	
	def sigmoid(self, x):
		return 1./(1+np.exp(-x))

	def cuda_activation_forward(self, Z, A):
		index = cuda.blockId.x * cuda.blockDim.x + cuda.threadId.x
		if index < Z.shape[0] * Z.shape[1]:
			A[index] = self.sigmoid(Z[index])

	def cuda_activation_backprop(self, Z, A, dZ):
		index = cuda.blockId.x * cuda.blockDim.x + cuda.threadId.x
		if index < dZ.shape[0] * dZ.shape[1]:
			dZ[index] = dA[index] * self.sigmoid(Z[index]) * (1 - self.sigmoid(Z[index]))

	def activation_forward(self):
		self.A = self.sigmoid(self.Z)
	
	def activation_backprop(self):
		self.dZ = self.dA * self.sigmoid(self.Z) * (1 - self.sigmoid(self.Z))

	def forward(self, Z, cuda = False):
		self.Z = Z
		if cuda:
			pass
		else:
			self.activation_forward()
		return self.A

	def backprop(self, dA, cuda = False):
		self.dA = dA
		if cuda:
			pass
		else:
			self.activation_backprop()
		return self.dZ	

	def reset(self):
		pass

class ReLULayer(Layer):
	def __init__(self, name):
		super().__init__(name)
	
	def activation_forward(self):
		self.A = np.copy(self.Z)
		self.A[self.A <= 0] = 0

	def activation_backprop(self):
		self.dZ = np.zeros(self.dA.shape)
		self.dZ[self.Z > 0] = self.dA[self.Z > 0]
		
	def forward(self, Z ,cuda = False):
		self.Z = Z
		if cuda:
			pass
		else:
			self.activation_forward()
		return self.A

	def backprop(self, dA, cuda = False):
		self.dA = dA
		if cuda:
			pass
		else:
			self.activation_backprop()
		return self.dZ

	def reset(self):
		pass

class LinearLayer(Layer):
	def __init__(self, name, input_size, output_size):
		super().__init__(name)
		self.input_size = input_size
		self.output_size = output_size
		self.learning_rate = 0.001
		
	def reset(self):
		self.W = np.random.normal(loc=0.0, scale=1, size=(self.input_size,self.output_size))
		self.b = np.random.normal(loc=0.0, scale=1, size=self.output_size)
	
	def activation_forward(self, A):
		return np.matmul(self.A, self.W) + self.b

	def activation_backprop(self, dZ):
		return np.matmul(dZ, np.transpose(self.W))

	def update_weights(self, dZ):
		self.dW = np.matmul(np.transpose(self.A), dZ)
		self.W -= self.learning_rate * self.dW

	def update_bias(self, dZ):
		self.b -= self.learning_rate * np.sum(dZ, axis=0) / dZ.shape[0] 

	def forward(self, A ,cuda = False):
		self.A = A
		if cuda:
			pass
		else:
			return self.activation_forward(A)
			
	def backprop(self, dZ, cuda = False):
		if cuda:
			pass
		else:
			self.update_bias(dZ)
			dZout = self.activation_backprop(dZ)	
			self.update_weights(dZ)
			return dZout