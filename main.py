from layer import *
from nn import *
from data import *
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='CUDANN')
parser.add_argument('--cuda', dest='feature', action='store_true')
parser.add_argument('--no-cuda', dest='feature', action='store_false')
parser.set_defaults(feature=False)
args = parser.parse_args()

if __name__ == '__main__':
	np.random.seed(2137)
	#These options determine the way floating point numbers, arrays and
    #other NumPy objects are displayed.
	np.set_printoptions(suppress=True)
	
	#args.feature determines whether program should use cuda funtions
	nn = NeuralNetwork(args.feature)
	nn.learning_rate = 0.01

	#layers are not created per say, just specifying the name and input/output sizes
	#the input size of the first layer is arbitrary (depends on the type of data provided)
	#the input size of all other layers has to match the output size of the prevoius layers
	nn.addLayer(LinearLayer("Linear1",2,10))
	#specifies the name of funtion to forward through 
	nn.addLayer(ReLULayer("ReLU1"))
	nn.addLayer(LinearLayer("Linear2",10,6))
	nn.addLayer(ReLULayer("ReLU2"))
	nn.addLayer(LinearLayer("Linear3",6,3))
	nn.addLayer(ReLULayer("ReLU3"))
	nn.addLayer(LinearLayer("Linear4",3,1))
	nn.addLayer(SigmoidLayer("Sigmoid"))

	#it creates the matices for all the layers (with given sizes)
	#it initiates the matrices with random numbers (normal distribution)
	nn.init_layers()

	epochs = 10
	n_points = 200
	n_batches = 20
	
	#data is a list (batches) of dictionaries (x and y arrays) 
	data = [prepare_data(n_points) for i in range(0, n_batches)]

	#back propagation algorithm
	nn.train(data, epochs = epochs, log = True)
	
	test_data = prepare_data(n_points)
	_, acc = nn.predict(test_data)
	print("Accuracy={:.2f}".format(acc))