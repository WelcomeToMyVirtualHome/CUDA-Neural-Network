import numpy as np
from layer import *

size = 8
a = b = c = np.random.random((size,size))
out = np.zeros((size,size), dtype=np.float32)

block = (4, 4)
grid = (a.shape[0] // block[0] if a.shape[0] % block[0] == 0 
            else a.shape[0] // block[0] + 1,
        int(a.shape[0] // block[1] if a.shape[1] % block[1] == 0 
            else a.shape[1] // block[1] + 1))

print(block)
print(grid)
rtol = 1e-06
sig = SigmoidLayer("Sigmoid")
sigmoid_activation_forward[block, grid](a, out)
out_n = sig.forward(a)
print("Sigmoid forward check = {:}".format(np.allclose(out, out_n, rtol=rtol)))

sigmoid_activation_backprop[block, grid](a, b, out)
out_n = sig.backprop(a)
print("Sigmoid backprop check = {:}".format(np.allclose(out, out_n, rtol=rtol)))

relu = ReLULayer("Relu")
relu_activation_forward[block, grid](a, out)
out_n = relu.forward(a)
print("ReLU forward check = {:}".format(np.allclose(out, out_n, rtol=rtol)))

relu_activation_backprop[block, grid](a, b, out)
out_n = relu.backprop(a)
print("ReLU forward check = {:}".format(np.allclose(out, out_n, rtol=rtol)))

linear = LinearLayer("Linear", size, size)
linear.W = b
linear.b = c
linear_activation_forward[block, grid](a, b, c, out)
out_n = linear.forward(a)
print("Linear forward check = {:}".format(np.allclose(out, out_n, rtol=rtol)))

linear_activation_backprop[block, grid](a, b, out)
out_n = linear.backprop(a)
print("Linear forward check = {:}".format(np.allclose(out, out_n, rtol=rtol)))