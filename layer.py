import numpy as np
from numba import cuda

"""
TODO
cuda implementation - flattening matrices for cuda threads/different cuda matrix multiplication logic
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

	def activation_forward_cuda(self):
		index = cuda.blockId.x * cuda.blockDim.x + cuda.threadId.x
		if index < self.Z.shape[0] * self.Z.shape[1]:
			self.A[index] = self.sigmoid(Z[index])
		self.dA = dA

	def activation_backprop_cuda(self):
		index = cuda.blockId.x * cuda.blockDim.x + cuda.threadId.x
		if index < self.dZ.shape[0] * self.dZ.shape[1]:
			dZ[index] = self.dA[index] * self.sigmoid(Z[index]) * (1 - self.sigmoid(Z[index]))
		self.dZ = dZ

	def activation_forward(self):
		self.A = self.sigmoid(self.Z)
	
	def activation_backprop(self):
		self.dZ = self.dA * self.sigmoid(self.Z) * (1 - self.sigmoid(self.Z))

	def forward(self, Z, cuda = False):
		self.Z = Z
		if cuda:
			self.activation_forward_cuda()
		else:
			self.activation_forward()
		return self.A

	def backprop(self, dA, cuda = False):
		self.dA = dA
		if cuda:
			self.activation_backprop_cuda()
		else:
			self.activation_backprop()
		return self.dZ	

	def reset(self):
		pass

class ReLULayer(Layer):
	def __init__(self, name):
		super().__init__(name)
	
	def activation_forward_cuda(self):
		index = cuda.blockId.x * cuda.blockDim.x + cuda.threadId.x
		if index < self.A.shape[0] * self.A.shape[1]:
			self.A[index] = np.max(self.Z[index],0)

	def activation_backprop_cuda(self):
		index = cuda.blockId.x * cuda.blockDim.x + cuda.threadId.x
		if index < self.A.shape[0] * self.A.shape[1]:
			if self.Z[index] > 0:
				self.dZ[index] = self.dA[index]
			else:
				self.dZ[index] = 0

	def activation_forward(self):
		self.A = np.copy(self.Z)
		self.A[self.A <= 0] = 0

	def activation_backprop(self):
		self.dZ = np.zeros(self.dA.shape)
		self.dZ[self.Z > 0] = self.dA[self.Z > 0]
		
	def forward(self, Z ,cuda = False):
		self.Z = Z
		if cuda:
			self.activation_forward_cuda()
		else:
			self.activation_forward()
		return self.A

	def backprop(self, dA, cuda = False):
		self.dA = dA
		if cuda:
			self.activation_backprop_cuda()
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
	
	def activation_forward_cuda(self, A):
		row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
		col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

		Z_x_dim = self.A.shape[0]
		Z_y_dim = self.W.shape[1]

		Z_value = 0;
		if row < Z_y_dim and col < Z_x_dim:
			for i in range(0, self.W.shape[0]):
				Z_value += W[row * self.W.shape[0] + i] * A[i * A.shape[0] + col]
			Z[row * Z_x_dim + col] = Z_value + b[row]

	def activation_backprop_cuda(self):
		row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
		col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
		dA_x_dim = dZ_x_dim
		dA_y_dim = W_x_dim

		dA_value = 0.0

		if row < dA_y_dim and col < dA_x_dim:
			for i in range(0,W_y_dim):
				dA_value += W[i * W_x_dim + row] * dZ[i * dZ_x_dim + col]
			dA[row * dA_x_dim + col] = dA_value;

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
			return self.activation_forward_cuda()
		else:
			return self.activation_forward(A)
			
	def backprop(self, dZ, cuda = False):
		if cuda:
			self.update_bias(dZ)
			dZout = self.activation_backprop_cuda(dZ)
			self.update_weights(dZ)
			return dZout
		else:
			self.update_bias(dZ)
			dZout = self.activation_backprop(dZ)	
			self.update_weights(dZ)
			return dZout