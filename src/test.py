import numpy as np
import sys

def generate_dataset(shape, linear = True):
	y = np.zeros((shape[0], 1))
	x = np.zeros((shape[0],shape[1]))
	gt = np.zeros((shape[0],1))
	
	t = 3.14*np.random.randn(shape[0]).T
	
	coeff = np.random.randint(1, 9, size=shape[1])
	print "coefficients used to construct y are: {}".format(coeff)
	noise1 = np.random.randint(50, 200, size = shape[0]//2).reshape(shape[0]//2, 1) # for class 1
	noise2 = -1*np.random.randint(50, 200, size = shape[0]//2).reshape(shape[0]//2, 1) # for class 2
	
	for col in range(shape[1]):
		x[:,col] = t
		y[:] += coeff[col]*x[:,col].reshape(shape[0], 1)
		if not linear: y[:] += coeff[col]*x[:,col].reshape(shape[0], 1)**2
		
	y[:int(0.5*shape[0])] = y[:int(0.5*shape[0])] + noise1
	y[int(0.5*shape[0]):] = y[int(0.5*shape[0]):] + noise2
	
	# for two classes only
	index, _ = np.where((np.matmul(x,(coeff.reshape(shape[1], 1))) - y) >= 0)
	gt[index] = 1
	index = np.random.randint(0, shape[0], size=shape[0])
	y, x, gt = y[index], x[index], gt[index]
	
	dataX, dataY = np.hstack([x,y]), gt
	return (dataX[int(shape[0]*0.7):], dataY[int(shape[0]*0.7):]),\
			(dataX[:int(shape[0]*0.7)], dataY[:int(shape[0]*0.7)])


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_out2deriv(out):
	return out * (1 - out)

def cost_fn(pred, labels, smooth = 0.001):
	loss = abs(labels*np.log(pred + smooth) + (1-labels)*np.log(1 - pred + smooth))
	return loss

def get_weights(mu, sd):
	# print mu, sd
	weights = np.random.normal(mu, 2*sd)
	return weights

def accuracy(pred, labels, thresh = 0.5):
	pred[np.where(pred >= thresh)] = 1.0
	pred[np.where(pred < thresh)[0]] = 0.0
	# print np.where((pred - labels) == 0)[0].shape[0], labels.shape[0]
	acc = np.where((pred - labels) == 0)[0].shape[0] / np.array(labels.shape[0], dtype="float")
	return acc

# normal distribution function for probabilistic nature 
def fx(inx, mu, sd, smooth = 1e-3):
	in_dim, out_dim = mu.shape
	updated = np.zeros((batch_size, out_dim))
	for i in range(batch_size):
		x = np.array([inx[i], ]*out_dim).T
		fx = -0.5*((mu-x)/(sd + smooth))**2 
		fx = np.exp(fx)/(sd+smooth)
		fx = np.sum(fx, axis=0)
		updated[i] = fx
	return updated

def fx_deriv_mu(output_delta, x, mu, sd, smooth=1e-3):
	# implement for batch
	in_dim, out_dim = mu.shape
	lp = output_delta
	fp = fx(x,mu, sd) 
	sigmap = sigmoid(fp)*(1 - sigmoid(fp))
	der = np.zeros((batch_size, out_dim))
	for i in range(batch_size):
		der[i] = lp[i] * sigmap[i] * fp[i] * np.sum((np.array([x[i], ]*out_dim).T-mu)/(sd + smooth)**2, axis = 0) 
	return der

def fx_deriv_sd(output_delta, x, mu, sd, smooth = 1e-3):
	in_dim, out_dim = mu.shape
	lp = output_delta
	fp = fx(x,mu, sd) 
	sigmap = sigmoid(fp)*(1 - sigmoid(fp))
	der = np.zeros((batch_size, out_dim))
	for i in range(batch_size):
		der[i] = lp[i] * sigmap[i] * fp[i] * np.sum((np.array([x[i], ]*out_dim).T- mu - sd**2)/(sd + smooth)**3, axis = 0) 
	return der

#################################################################################


class Layer(object):
	
	def __init__(self,input_dim, output_dim,nonlin,nonlin_deriv, alpha = 0.1):
		self.mean = np.random.randn(input_dim, output_dim)
		self.sd = abs(np.random.randn(input_dim, output_dim))
		# self.biase = np.random.randn(output_dim)
		self.nonlin = nonlin
		self.nonlin_deriv = nonlin_deriv
		self.alpha = alpha

	def forward(self, input):
		self.input = input
		self.weights = get_weights(self.mean, self.sd)
		self.fx = self.input.dot(self.weights)
		self.output = self.nonlin(self.fx) 
		return self.output

	def backward(self, output_delta_mean, output_delta_sd):
		self.mean_output_delta = fx_deriv_mu(output_delta_mean, self.input, self.mean, self.sd)
		# print self.mean_output_delta
		self.sd_output_delta = fx_deriv_sd(output_delta_sd, self.input, self.mean, self.sd)
		return self.mean_output_delta.dot(self.mean.T), self.sd_output_delta.dot(self.sd.T)

	def update(self):
		self.mean -= self.input.T.dot(self.mean_output_delta) * self.alpha +0.0001
		self.sd -= self.input.T.dot(self.sd_output_delta) * self.alpha  + 0.0001
		self.sd[self.sd < 0] = 0

	def stats(self):
		return (self.mean[0][0], self.sd[0][0])
		

num_examples = 100
output_dim = 1
iterations = 100000

(x,y), _ = generate_dataset((500, 1))
print x.shape, y.shape

batch_size = 1
input_dim = 2
layer_1_dim = 3
# layer_2_dim = 1


# layer building

layer_1 = Layer(input_dim, layer_1_dim, sigmoid, sigmoid_out2deriv)
# layer_2 = Layer(layer_1_dim, layer_2_dim, sigmoid,sigmoid_out2deriv)
output_layer = Layer(layer_1_dim, output_dim, sigmoid, sigmoid_out2deriv)



# graph building...
for iter in range(iterations):
	error = 0
	index = np.random.randint(0, x.shape[0], size=x.shape[0])
	y, x = y[index], x[index]
	acc = []
	for batch_i in range(int(len(x) / batch_size)):
		batch_x = x[(batch_i * batch_size):(batch_i+1)*batch_size]
		batch_y = y[(batch_i * batch_size):(batch_i+1)*batch_size]
		
		batch_x.shape
		layer_1_out = layer_1.forward(batch_x)
		output_layer_out = output_layer.forward(layer_1_out)

		output_layer_delta = cost_fn(output_layer_out, batch_y)

		layer_1_delta_mean, layer_1_delta_sd = output_layer.backward(output_layer_delta, output_layer_delta)
		layer_1.backward(layer_1_delta_mean, layer_1_delta_sd)

		layer_1.update()
		output_layer.update()
		
		acc.append(accuracy(output_layer_out, batch_y))
		error = np.mean(np.abs(output_layer_delta))


	# print status...
	if(iter % 10 == 0):
		sys.stdout.write("\rIter:" + str(iter) + " Loss:" + str(error) + " Acc: "+ str(np.mean(acc)) + " Data check : " + str(layer_1.stats()) )
		print("")
	if(iter % 100 == 0):
		print("\n")
		print("*"*50)   
		print(batch_y, output_layer_out)


class Network(object):
	def __init__(self, layers):
		self.layers = layers
		pass

	def compile(self):
		# forward
		temp = inputX
		for layer in self.layers:
			temp = layer.forward(temp)

		
	def update(self):
		pass

	def check_SNR(self):
		pass

	def status(self):
		pass

	def train(self):
		pass
