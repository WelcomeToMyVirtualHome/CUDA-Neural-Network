import cProfile
import pstats

from nn import *
from data import *
from layer import *

def init_nn(cuda = False):
	nn = NeuralNetwork(cuda)
	nn.learning_rate = 0.01
	nn.addLayer(LinearLayer("Linear1",2,10))
	nn.addLayer(ReLULayer("ReLU1"))
	nn.addLayer(LinearLayer("Linear2",10,6))
	nn.addLayer(ReLULayer("ReLU2"))
	nn.addLayer(LinearLayer("Linear3",6,3))
	nn.addLayer(ReLULayer("ReLU3"))
	nn.addLayer(LinearLayer("Linear4",3,1))
	nn.addLayer(SigmoidLayer("Sigmoid"))
	nn.init_layers()
	return nn

def train_test(filename, nn, epochs, data, log = False, cuda = False):
	cProfile.run('nn.train(data, epochs = epochs, log = True)', filename = filename )


# nn = init_nn()
epochs = 20
n_batches = 50
n_points_array = [10, 20, 100, 200, 500, 1000, 2000, 5000, 10000]
filename0 = 'gpu_stats'
for n in n_points_array:
	filename = filename0 + str(n)
	nn = init_nn(True)
	data = [prepare_data(n) for i in range(0, n_batches)]
	train_test(filename, nn, epochs, data, log = False, cuda = False)


p = pstats.Stats(filename)
p.strip_dirs().sort_stats('cumulative').print_stats(10)
p.strip_dirs().sort_stats('time').print_stats(10)