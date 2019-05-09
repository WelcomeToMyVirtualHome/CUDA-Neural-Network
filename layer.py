import numpy as np
from numba import cuda

"""
TODO
cuda implementation
"""

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

	@cuda.jit
	def activation_forward_cuda(self, Z, A):
		index = cuda.grid(1)
		pass

	def activation_forward_cuda_wrapper(self):
		pass

	@cuda.jit
	def activation_backprop_cuda(self):
		index = cuda.grid(1)
		pass
	
	def activation_backprop_cuda_wrapper(self):
		pass

	def activation_forward(self):
		self.A = self.sigmoid(self.Z)
	
	def activation_backprop(self):
		self.dZ = self.dA * self.sigmoid(self.Z) * (1 - self.sigmoid(self.Z))

	def forward(self, Z, cuda = False):
		self.Z = Z
		if cuda:
			self.activation_forward_cuda_wrapper()
		else:
			self.activation_forward()
		return self.A

	def backprop(self, dA, cuda = False):
		self.dA = dA
		if cuda:
			self.activation_backprop_cuda_wrapper()
		else:
			self.activation_backprop()
		return self.dZ	

	def reset(self):
		pass

class ReLULayer(Layer):
	def __init__(self, name):
		super().__init__(name)
	
	@cuda.jit
	def activation_forward_cuda(self):
		cuda.grid(1)
		pass

	def activation_forward_cuda_wrapper(self):
		pass

	@cuda.jit
	def activation_backprop_cuda(self):
		cuda.grid(1)
		pass

	def activation_backprop_cuda_wrapper(self):
		pass

	def activation_forward(self):
		self.A = np.copy(self.Z)
		self.A[self.A <= 0] = 0

	def activation_backprop(self):
		self.dZ = np.zeros(self.dA.shape)
		self.dZ[self.Z > 0] = self.dA[self.Z > 0]
		
	def forward(self, Z ,cuda = False):
		self.Z = Z
		if cuda:
			self.activation_forward_cuda_wrapper()
		else:
			self.activation_forward()
		return self.A

	def backprop(self, dA, cuda = False):
		self.dA = dA
		if cuda:
			self.activation_backprop_cuda_wrapper()
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
	
	@cuda.jit
	def activation_forward_cuda(self, A):
		row, col = cuda.grid(2)
		pass

	def activation_forward_cuda_wrapper(self, A):
		pass

	@cuda.jit
	def activation_backprop_cuda(self, dZ):
		row, col = cuda.grid(2)
	
		pass

	def activation_backprop_cuda_wrapper(self, dZ):
		griddim = 1, 1
		blockdim = 10, 10
		
		pass

	def activation_forward(self, A):
		return np.matmul(self.A, self.W) + self.b

	def activation_backprop(self, dZ):
		return np.matmul(dZ, np.transpose(self.W))

	def update_weights(self, dZ):
		self.dW = np.matmul(np.transpose(self.A), dZ)
		self.W -= self.learning_rate * self.dW

	def update_bias(self, dZ):
		self.b -= self.learning_rate * np.sum(dZ, axis=0) / dZ.shape[0] 

	def forward(self, A, cuda = False):
		self.A = A
		if cuda:
			return self.activation_forward_cuda_wrapper(A)
		else:
			return self.activation_forward(A)
			
	def backprop(self, dZ, cuda = False):
		if cuda:
			self.update_bias(dZ)
			dZout = self.activation_backprop_cuda_wrapper(dZ)
			self.update_weights(dZ)
			return dZout
		else:
			self.update_bias(dZ)
			dZout = self.activation_backprop(dZ)	
			self.update_weights(dZ)
			return dZout