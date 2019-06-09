import numpy as np
import math
from cuda_func import *

def sigmoid(x):
		return 1. / (1 + np.exp(-x))

class Layer():
	def __init__(self, name):
		self.name = name

	def forward(self, A, cuda = False):
		raise NotImplementedError

	def backprop(self, dZ, cuda = False):
		raise NotImplementedError

	def reset(self):
		raise NotImplementedError

class SigmoidLayer(Layer):
	def __init__(self, name):
		super().__init__(name)

	def activation_forward_cuda(self, Z):
		block = (4, 4)
		grid = (Z.shape[0] // block[0] if Z.shape[0] % block[0] == 0 
            else Z.shape[0] // block[0] + 1,
        int(Z.shape[0] // block[1] if Z.shape[1] % block[1] == 0 
            else Z.shape[1] // block[1] + 1))
		out = np.zeros(Z.shape)
		sigmoid_activation_forward[block, grid](Z, out)
		return out

	def activation_backprop_cuda(self, dA):
		block = (4, 4)
		grid = (dA.shape[0] // block[0] if dA.shape[0] % block[0] == 0 
            else dA.shape[0] // block[0] + 1,
        int(self.Z.shape[0] // block[1] if dA.shape[1] % block[1] == 0 
            else dA.shape[1] // block[1] + 1))
		out = np.zeros(dA.shape)
		# sigmoid_activation_backprop[block, grid](dA, self.Z, out)
		sigmoid_activation_backprop[block, grid](self.Z, dA, out)
		return out

	def activation_forward(self, Z):
		return sigmoid(Z)
	
	def activation_backprop(self, dA):
		return dA * sigmoid(self.Z) * (1 - sigmoid(self.Z))

	def forward(self, Z, cuda = False):
		self.Z = Z
		if cuda:
			return self.activation_forward_cuda(Z)
		else:
			return self.activation_forward(Z)
	
	def backprop(self, dA, cuda = False):
		if cuda:
			return self.activation_backprop_cuda(dA)
		else:
			return self.activation_backprop(dA)
		
	def reset(self):
		pass

class ReLULayer(Layer):
	def __init__(self, name):
		super().__init__(name)
		
	def activation_forward_cuda(self, Z):
		block = (4, 4)
		grid = (Z.shape[0] // block[0] if Z.shape[0] % block[0] == 0 
            else Z.shape[0] // block[0] + 1,
        int(Z.shape[0] // block[1] if Z.shape[1] % block[1] == 0 
            else Z.shape[1] // block[1] + 1))
		out = np.zeros(Z.shape)
		relu_activation_forward[block, grid](Z, out)
		return out

	def activation_backprop_cuda(self, dA):
		block = (4, 4)
		grid = (dA.shape[0] // block[0] if dA.shape[0] % block[0] == 0 
            else dA.shape[0] // block[0] + 1,
        int(dA.shape[0] // block[1] if dA.shape[1] % block[1] == 0 
            else dA.shape[1] // block[1] + 1))
		out = np.zeros(dA.shape)
		relu_activation_backprop[block, grid](self.Z, dA, out)
		return out

	def activation_forward(self, Z):
		A = np.copy(Z)
		A[A <= 0] = 0
		return A

	def activation_backprop(self, dA):
		dZ = np.zeros(dA.shape)
		dZ[self.Z > 0] = dA[self.Z > 0]
		return dZ
		
	def forward(self, Z ,cuda = False):
		self.Z = Z
		if cuda:
			return self.activation_forward_cuda(Z)
		else:
			return self.activation_forward(Z)
	
	def backprop(self, dA, cuda = False):
		if cuda:
			return self.activation_backprop_cuda(dA)
		else:
			return self.activation_backprop(dA)
		
	def reset(self):
		pass

class LinearLayer(Layer):
	def __init__(self, name, input_size, output_size, learning_rate=0.001):
		super().__init__(name)
		self.input_size = input_size
		self.output_size = output_size
		self.learning_rate = learning_rate
	
	def reset(self):
		self.W = np.random.normal(loc=0.0, scale=1, size=(self.input_size,self.output_size))
		self.b = np.random.normal(loc=0.0, scale=1, size=self.output_size)
		
	def activation_forward_cuda(self, dA):
		block = (4, 4)
		grid = (dA.shape[0] // block[0] if dA.shape[0] % block[0] == 0 
            else dA.shape[0] // block[0] + 1,
        int(dA.shape[0] // block[1] if dA.shape[1] % block[1] == 0 
            else dA.shape[1] // block[1] + 1))

		out = np.zeros((dA.shape[0], self.W.shape[1]))
		linear_activation_forward[block, grid](dA, self.W, self.b, out)

		# threadsperblock = (4, 4)
		# blockspergrid_x = int(math.ceil(dA.shape[0] / threadsperblock[1]))
	
		# blockspergrid_y = int(math.ceil(self.W.shape[1] / threadsperblock[0]))
		# blockspergrid = (blockspergrid_x, blockspergrid_y)

		# out = np.zeros((dA.shape[0], self.W.shape[1]))

		# dA_global_mem = cuda.to_device(dA)
		# W_global_mem = cuda.to_device(self.W)
		# out_global_mem = cuda.to_device(out)
		# B_global_mem = cuda.to_device(self.b)

		# linear_activation_forward[threadsperblock, blockspergrid](dA_global_mem,
		# 							 W_global_mem, B_global_mem, out_global_mem)

		# out = out_global_mem.copy_to_host()
		return out

	def activation_backprop_cuda(self, Z):
		block = (4, 4)
		grid = (Z.shape[0] // block[0] if Z.shape[0] % block[0] == 0 
            else Z.shape[0] // block[0] + 1,
        int(Z.shape[0] // block[1] if Z.shape[1] % block[1] == 0 
            else Z.shape[1] // block[1] + 1))
	
		out = np.zeros((Z.shape[0], self.W.shape[0]))
		linear_activation_backprop[block, grid](Z, self.W, out)

		# threadsperblock = (4, 4)
		# blockspergrid_x = int(math.ceil(Z.shape[0] / threadsperblock[1]))
		# #not sure here have to check later
		# # blockspergrid_y = int(math.ceil(self.W.shape[1] / threadsperblock[0]))
		# blockspergrid_y = int(math.ceil(self.W.shape[0] / threadsperblock[0]))
		# blockspergrid = (blockspergrid_x, blockspergrid_y)

		# #self.W.shape[0] means that we take into account the transpositoin
		# out = np.zeros((Z.shape[0], self.W.shape[0]))

		# Z_global_mem = cuda.to_device(Z)
		# W_global_mem = cuda.to_device(self.W)
		# out_global_mem = cuda.to_device(out)

		# linear_activation_backprop[threadsperblock, blockspergrid](Z_global_mem, W_global_mem, out_global_mem)
		# out = out_global_mem.copy_to_host()
		return out
	
	def activation_forward(self, A):
		return np.matmul(A, self.W) + self.b

	def activation_backprop(self, dZ):

		return np.matmul(dZ, np.transpose(self.W))

	def update_weights(self, dZ):
		self.dW = np.matmul(np.transpose(self.A), dZ)
		self.W -= self.learning_rate * self.dW

	def update_bias(self, dZ):
		self.b -= self.learning_rate * np.sum(dZ, axis=0) / dZ.shape[0] 

	def forward(self, A, cuda = False):
		self.A = A
		
		# if np.allclose(self.activation_forward_cuda(A), self.activation_forward(A), rtol=0.01):
		# 	print("forward")
		if cuda:
			return self.activation_forward_cuda(A)
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