"""
Used to run and test MLP.py.
Author: Marius Havgar
"""

#File to run calculations. 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import numpy as np
from MLP import neuralNetwork #Import neural network class
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
#import relevant datasets
from sklearn.datasets import load_breast_cancer
from  sklearn.preprocessing import scale
cancer = load_breast_cancer()
from sklearn.neural_network import MLPRegressor as MLPClassifier
from sklearn.model_selection import train_test_split

def CreateDesignMatrix(x, y, n = 5):
	"""
	Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
	Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
	"""
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)
	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))
	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = x**(i-k) * y**k
	return X

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	noise = np.random.normal(loc=0,scale=0.2,size=len(x))
	return term1 + term2 + term3 + term4 + noise


def createFrankeData(num_points):
	l = np.linspace(0, 1, num_points)
	x_, y_ = np.meshgrid(l, l)
	x, y = np.ravel(x_), np.ravel(y_)
	X= []
	for i in range(len(x)):
		X.append(np.asarray([x[i], y[i]]))
	X = np.asarray(X)	
	z = FrankeFunction(x, y)
	return X, z

num_points = 20
X, z = createFrankeData(num_points)
X_train, X_test, y_train, y_test = train_test_split(X, z, test_size = 0.2)

nn = neuralNetwork(dimensions = [2, 512,  1], activation= "tanh", loss_function="MSE", learning_rate=0.2, model = "regressor", initialization="xavier")

# nn.batch_train(X_train, y_train, X_test, y_test, plot_training_results=True ,verbose= True, batch_size = 100, epochs = 10)
# data = [nn.predict(x) for x in X]
# plt.imshow(np.reshape(data, (num_points, num_points)))
# plt.colorbar()
# plt.figure()
# plt.imshow(np.reshape(z, (num_points, num_points)))
# plt.colorbar()
# plt.show()
# data = [nn.predict(x) for x in X]
# plt.imshow(np.reshape(data, (num_points, num_points)))
# plt.figure()
# plt.imshow(np.reshape(z, (num_points, num_points)))
# plt.show()
for i in range(1,5):
	print("Starting on epoch number ",50*(i-1))
	nn.train(X_train, y_train, X_test, y_test, plot_training_results=True, epochs= 50 ,verbose= True)

	data = [nn.predict(x) for x in X]
	plt.imshow(np.reshape(data, (num_points, num_points)))
	plt.colorbar()
	plt.figure()
	plt.imshow(np.reshape(z, (num_points, num_points)))
	plt.colorbar()
	plt.show()
	
