import numpy as np
from sklearn.linear_model import LogisticRegression

# seed that was used to produce the calculations in the report
# np.random.seed(20)

def sigmoid(z):
    ''' The logistic function '''

    return 1/(1+np.exp(-z))

def indicator_function(y_true, y_pred):
    ''' Indicator function.

        Args:
            y_true (arraylike): target values
            y_pred (arraylike): predicted values
        Returns:
            1, if y_true equals y_pred
            0, else
    '''

    assert y_true.shape == y_pred.shape
    return y_true == y_pred

def accuracy_score(y_true, y_pred):
    ''' Function that computes the accuracy score '''

    indicator_vec = indicator_function(y_true.ravel(), y_pred.ravel())
    accuracy = np.sum(indicator_vec)/indicator_vec.shape[0]
    return accuracy


class LogReg():
    ''' Class implementing logistic regression with standard gradient descent. '''

    def __init__(self):
        self.beta = None
        self.y_pred = None

    def cost_function(self, X, y, beta):
        ''' The cost-function for logistic regression '''
        pred = sigmoid(X @ beta)
        # use that 1 - simoid(x) = sigmoid(-x)
        return - np.sum(y.T @ np.log(pred) + (1-y).T @ np.log(sigmoid(-X@beta)))

    def fit(self, X, y, num_epochs=100, eta=0.01):
        ''' Function that fits the model according to given training data.

            Parameters are found with standard gradient descent optimizer.

            Args:
                X (matrix): design matrix
                y (arraylike): target values
                num_epochs (int): number of iterations
                eta (float): learning rate
            Returns:
                beta (arraylike): regression coefficients
                beta_history (arraylike): coefficients for each iteration
                cost_history (arraylike): cost for each iterations
        '''

        # initialize weights with random numbers
        self.beta = np.random.uniform(size=(X.shape[1],1))

        # arrays to save calculations in the loop
        self.cost_history = np.zeros(num_epochs)
        self.beta_history = np.zeros((num_epochs,X.shape[1]))

        for epoch in range(num_epochs):
            pred = X @ self.beta
            pred = sigmoid(pred)

            # derivative of the cost function wrt beta
            gradient = X.T @ (pred - y[:,np.newaxis])

            self.beta_history[epoch,:] = self.beta.T
            self.cost_history[epoch]  = self.cost_function(X, y, self.beta)

            self.beta = self.beta - eta * gradient

        return self.beta, self.beta_history, self.cost_history


    def predict(self, X, threshold = 0.5):
        ''' Function that returns the predicted values as discrete values

            Args:
                X (matrix): test values
                threshold (float): cut-off threshold

            Returns:
                y_pred (arraylike): predicted values
        '''

        pred = sigmoid(X @ self.beta)
        self.y_pred = (pred >= threshold).astype(int)   # to return values 0 and 1
        return self.y_pred
