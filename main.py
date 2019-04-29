from layer import *
from nn import *
import numpy as np

def prepare_data(size, dim = 2):
	size = int(size/2)
	X1 = np.random.normal(loc=[1.0,1.0], scale=0.5, size=(size,dim))
	Y1 = np.full((size,1),0)
	X2 = np.random.normal(loc=[0.1,0.1], scale=0.3, size=(size,dim))
	Y2 = np.full((size,1),1)
	return {"data":np.concatenate((X1,X2)),"labels":np.concatenate((Y1,Y2))}

def to_label(labels, threshold = 0.5):
	labels[labels > threshold] = 1
	labels[labels <= threshold] = 0
	return labels

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

	epochs = 400
	batch_size = 200
	n_batches = 50
	data = [prepare_data(batch_size) for i in range(0, n_batches)]
	for i in range(0,epochs):
		cost = 0
		for b, batch, in zip(range(0,n_batches),data):
			Y_hat = nn.forward(batch["data"])
			nn.backprop(Y_hat, batch["labels"])
			cost += nn.bce.cost(Y_hat, batch["labels"])
			labels = to_label(Y_hat)
		print("Epoch={:d}, cost={:.2f}".format(i,cost))
	
	test_data = prepare_data(batch_size)
	Y_hat = nn.forward(test_data["data"])
	acc = np.sum(to_label(Y_hat) == test_data["labels"])/len(test_data["labels"])
	print("Accuracy={:.2f}".format(acc))

