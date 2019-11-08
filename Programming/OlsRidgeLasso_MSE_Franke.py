"""
Code to compare MSE from OLS/Ridge with the MLP model.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoLarsIC, RidgeCV
from sklearn.model_selection import train_test_split
import numpy as np

def olsRidge(X, y, test):
    alphas = np.arange(10001)/10000
    alphas[0] = 0.0000001   # Approximately OLS
    clf = RidgeCV(alphas=alphas)
    clf.fit(X, y)
    y_pred = clf.predict(test)
    return y_pred, clf.alpha_, clf.coef_

def lasso(X, y):
    clf = LassoLarsIC(criterion='aic')
    clf.fit(X, y)
    y_pred = clf.predict(X)
    return y_pred, clf.alpha_, clf.coef_

def CreateDesignMatrix(x, y, n = 10):
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
    X = CreateDesignMatrix(x,y)	
    z = FrankeFunction(x, y)
    return X, z

if __name__ == '__main__':
    # Initialization
    num_points = 20
    X, y = createFrankeData(num_points)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    # Perform Ridge and find best Ridge coefficient
    y_pred, alpha, coeff = olsRidge(X_train, y_train, X_test)
    # yhat = (y_pred >= 0.5).astype(int)
    print("RIDGE: Lambda = %f, MSE = %f" %(alpha, mean_squared_error(y_test, y_pred)))
