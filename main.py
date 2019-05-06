from layer import *
from nn import *
from data import *
import numpy as np

if __name__ == '__main__':
	np.random.seed(2137)
	np.set_printoptions(suppress=True)
	
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

	epochs = 10
	n_points = 200
	n_batches = 50
	data = [prepare_data(n_points) for i in range(0, n_batches)]

	nn.train(data, epochs = epochs, log = True, cuda = False)
	
	test_data = prepare_data(n_points)
	_, acc = nn.predict(test_data)
	print("Accuracy={:.2f}".format(acc))