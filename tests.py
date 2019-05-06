import cProfile
import pstats

from nn import *
from data import *
from layer import *

def init_nn():
	nn = NeuralNetwork()
	nn.learning_rate = 0.001
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
	cProfile.run('nn.train(data, epochs, False, False)', filename = filename)


nn = init_nn()
epochs = 100
n_points = 200
n_batches = 50
data = [prepare_data(n_points) for i in range(0, n_batches)]

filename = 'stats'
train_test(filename, nn, epochs, data, log = False, cuda = False)


p = pstats.Stats(filename)
p.sort_stats('cumulative').print_stats(10)