import numpy as np

def prepare_data(size, dim = 2):
	size = int(size/2)
	X1 = np.random.normal(loc=[1.0,1.0], scale=0.5, size=(size,dim))
	Y1 = np.full((size,1),0)
	X2 = np.random.normal(loc=[0.1,0.1], scale=0.3, size=(size,dim))
	Y2 = np.full((size,1),1)
	return { "x":np.concatenate((X1,X2)), "y":np.concatenate((Y1,Y2)) }

def to_label(labels, threshold = 0.5):
	labels[labels > threshold] = 1
	labels[labels <= threshold] = 0
	return labels