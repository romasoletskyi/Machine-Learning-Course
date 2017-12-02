def sum(x):
	s=0
	for i in x: s+=i

	return s

def max(x):
	x_max=x[0]
	for i in x:
		if (i>x_max): 
			x_max=i
	
	return x_max

def argmax(x):
	x_max=x[0]
	i_max=0
	for i in range(len(x)):
		if (x[i]>x_max):
			x_max=x[i]
			i_max=i

	return i_max

def min(x):
	x_min=x[0]
	for i in x:
		if (i<x_min): 
			x_min=i
	
	return x_min

def argmin(x):
	x_min=x[0]
	i_min=0
	for i in range(len(x)):
		if (x[i]<x_min):
			x_min=x[i]
			i_min=i

	return i_min

	 
