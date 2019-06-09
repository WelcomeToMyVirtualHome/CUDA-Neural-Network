import numpy as np
from layer import *

TPB = 4
a_x = 20
a_y = b_x = 55
b_y = 70
f_y = a_y 
f_x = 33

a = np.random.random((a_x,a_y))
g = np.random.random((a_x,a_y))
d = np.random.random(b_y )
b = np.random.random((b_x,b_y))
out = np.zeros((a_x,b_y))
f = np.random.random((f_x, f_y))

A_global_mem = cuda.to_device(a)
B_global_mem = cuda.to_device(b)
C_global_mem = cuda.device_array((a_x, b_y))
D_global_mem = cuda.to_device(d)
E_global_mem = cuda.device_array((a_x, f_x))
F_global_mem = cuda.to_device(f)



# Configure the blocks
threadsperblock = (TPB, TPB)
blockspergrid_x = int(math.ceil(a.shape[0] / threadsperblock[1]))
blockspergrid_y = int(math.ceil(b.shape[1] / threadsperblock[0]))
blockspergrid = (blockspergrid_x, blockspergrid_y)


rtol = 1e-05

sig = SigmoidLayer("Sigmoid")
D_global_mem = cuda.device_array((a_x, a_y))
sigmoid_activation_forward[blockspergrid, threadsperblock](A_global_mem, D_global_mem)
out = D_global_mem.copy_to_host()
out_n = sig.forward(a)
print("Sigmoid forward check = {:}".format(np.allclose(out, out_n, rtol=rtol)))

sigmoid_activation_backprop[blockspergrid, threadsperblock](A_global_mem, g, D_global_mem)
out = D_global_mem.copy_to_host()
out_n = sig.backprop(g)
print("Sigmoid backprop check = {:}".format(np.allclose(out, out_n, rtol=rtol)))

relu = ReLULayer("Relu")
relu_activation_forward[blockspergrid, threadsperblock](A_global_mem, D_global_mem)
out = D_global_mem.copy_to_host()
out_n = relu.forward(a)
print("ReLU forward check = {:}".format(np.allclose(out, out_n, rtol=rtol)))

relu_activation_backprop[blockspergrid, threadsperblock](A_global_mem, A_global_mem, D_global_mem)
out = D_global_mem.copy_to_host()
out_n = relu.backprop(a)
print("ReLU backprop check = {:}".format(np.allclose(out, out_n, rtol=rtol)))

linear = LinearLayer("Linear", a_x, a_y)
linear.W = b
linear.b = d
D_global_mem = cuda.to_device(d)
linear_activation_forward[blockspergrid, threadsperblock](A_global_mem, B_global_mem, D_global_mem, C_global_mem)
out = C_global_mem.copy_to_host()
out_n = linear.forward(a)
print("Linear forward check = {:}".format(np.allclose(out, out_n, rtol=rtol)))

linear_activation_backprop[blockspergrid, threadsperblock](A_global_mem, F_global_mem, E_global_mem)
out = E_global_mem.copy_to_host()
linear.W = f
out_n = linear.activation_backprop(a)
print("Linear backprop check = {:}".format(np.allclose(out, out_n, rtol=rtol))) 

matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
out = C_global_mem.copy_to_host()
out_n = a.dot(b)

print("Matrix mulplication = {:}".format(np.allclose(out, out_n, rtol=rtol)))
print(np.sum(out), np.sum(out_n))

