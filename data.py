import numpy as np

def prepare_data(size, dim = 2):
	'''
	Funtion creates sample points from two classes
	class 1 - points from normal distribution with params <x> = [1, 1] stddev = [0.5, 0.5]
	class 2 - points from normal distribution with params <x> = [0.1, 0.1] stddev = [0.3, 0.3]
	'''
	size = int(size/2)
	X1 = np.random.normal(loc=[1.0,1.0], scale=0.5, size=(size,dim))
	Y1 = np.full((size,1),0)
	X2 = np.random.normal(loc=[0.1,0.1], scale=0.3, size=(size,dim))
	Y2 = np.full((size,1),1)
	#concatenates matrices/arrays and returnes it as a dictionary (key - name of variable,
	# value - matrix/array associated with that variale)
	return { "x":np.concatenate((X1,X2)), "y":np.concatenate((Y1,Y2)) }

def to_label(labels, threshold = 0.5):
	labels[labels > threshold] = 1
	labels[labels <= threshold] = 0
	return labels