import numpy as np 
import sklearn as skl
from sklearn import model_selection
from sklearn.preprocessing import scale
from sklearn.metrics import  mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier #As a benchmark
import pickle # To export the model 
import copy

class neuralNetwork:
    def __init__(self, dimensions, activation = "exponential", learning_rate = 0.04, loss_function = "accuracy", model = "classifier", initialization = "standard"):
    #Constructor method
        self._name = activation
        self.model_type = "regressor"
        self.best_network = 0 #Keep a copy of the best performing network
        self.best_accuracy = 0 #The best accuracy score recorded. 
        self.learning_rate = learning_rate #Default = 0.04    
        self.num_layers = len(dimensions) 
        bias =[0]*(self.num_layers-1)
        if(initialization == "xavier"):
            bias = [0]*len(dimensions)
        else:
            for i in range(self.num_layers-1):
                bias[i] = [np.random.random()]*dimensions[i+1]                 
        self.bias = bias
        self.construct_network(dimensions, initialization)
        self.loss_function = self.accuracy_score
        if(loss_function == "MSE"):
            self.loss_function=self.mean_squared_error
        elif(loss_function == "accuracy"):
            self.loss_function = self.accuracy_score    

        if (activation == "linear"):
            self.activation_function = self.linear_activation
            self.dfdx = self.linear_derivative
        elif (activation == "relu"):
            self.activation_function = self.relu_activation
            self.dfdx = self.relu_derivative
        elif (activation == "leakyrelu"):
            self.activation_function = self.leaky_relu_activation
            self.dfdx = self.leaky_relu_derivative
        elif (activation == "elu"):
            self.activation_function = self.elu_activation
            self.dfdx = self.elu_derivative
        elif (activation == "arctan"):
            self.activation_function = self.arctan_activation
            self.dfdx = self.arctan_derivative
        elif (activation == "tanh"):
            self.activation_function = self.tanh_activation
            self.dfdx = self.tanh_derivative
        else:
            self.activation_function = self.exponential_activation
            self.dfdx = self.exponential_derivative
        self.scaler = False

        if(model == "classifier"):
            self.model_type = "classifier"
            self.output_activation = self.exponential_activation
            self.outut_derivative = self.exponential_derivative
        else:
            self.output_activation = self.linear_activation
            self.outut_derivative = self.linear_derivative

    def load_data(self, X, y, partition_sizes = [0.7, 0.2, 0.1]):
        self.train_data, self.train_targets, self.test_data, self.test_targets, self.validation_data, self.validation_targets = self.split_data(X, y, partition_sizes)


    def change_network(self):
        self.network = copy.deepcopy(self.best_network)
    #Switch out the current weights for the best performing network. 
    

    def linear_activation(self, z):
        """
        Linear function f(x) = x.
        """
        return z

    def linear_derivative(self, z):
        """
        The derivative of a linear function f(x) = x.
        """
        return 1


    def exponential_activation(self, z):
        """
        The exponential sigmoid.
        """
        z = np.asarray(z)
        return 1/(1 + np.exp(-z))

    def exponential_derivative(self, z):
        """
        The derivative of the exponential Sigmoid function.
        """
        z = self.exponential_activation(z)
        return z*(1-z)

    def relu_activation(self, z):
        z = np.asarray(z)
        a = (z > 0).astype(int)
        return a*z

    def relu_derivative(self, z):
        z = np.asarray(z)
        a = (z > 0).astype(int)
        return a

    def leaky_relu_activation(self, z):
        z = np.asarray(z)
        a = (z >= 0).astype(int)
        return a*z + (1 - a)*0.01*z

    def leaky_relu_derivative(self, z):
        z = np.asarray(z)
        a = (z >= 0).astype(int)
        return a + (1 - a)*0.01       


    def elu_activation(self, z):
        z = np.asarray(z)
        a = (z > 0).astype(int)
        return a*z + (1 - a)*(np.exp(z) - 1)

    def elu_derivative(self, z):
        z = np.asarray(z)
        a = (z > 0).astype(int)
        return a + (1 - a)*np.exp(z)


    def arctan_activation(self, z):
        z = np.asarray(z)
        return np.arctan(z)

    def arctan_derivative(self, z):
        z = np.asarray(z)
        return 1/(1+z*z)

    def tanh_activation(self, z):
        z = np.asarray(z)
        return np.tanh(z)

    def tanh_derivative(self, z):
        z = np.cosh(np.asarray(z))
        return 1/(z*z)


    def dfdx_cost_function(self, val, target):
    #The derivative of the cost function
        if(len(val.shape>1)):
            return (np.array(val)-np.array(target))
        return (val-target)

    def construct_network(self, dimensions, initialization):
    #dimensions contains all the information about the design of the network. 
        num_layers = len(dimensions) 
        T = [0] * (num_layers-1)
        if(initialization == "xavier"):
            for i in range(num_layers-1):
                m = dimensions[i+1]# number of nodes in next layer
                n = dimensions[i]   # number of nodes in current layer 
                # L = (np.random.random((m*n))*2-1)*0.1
                L = np.random.normal(0, np.sqrt(2/(m*n)), m*n)
                L = np.reshape(L, (m, n))
                T[i] = L  
        else:        
            for i in range(num_layers-1):
                m = dimensions[i+1]# number of nodes in next layer
                n = dimensions[i]   # number of nodes in current layer 
                # L = (np.random.random((m*n))*2-1)*0.1
                L = np.random.randn(m*n)*0.05
                L = np.reshape(L, (m, n))
                T[i] = L  
        self.network = T 

    def feedForward(self, input):
    #A method that takes an input vector and feeds it through the neural network. Returns activations from each layer. 
        output = np.asarray(input)
        #To store all the actications from the forward pass.
        outputs = [0]*(self.num_layers-1)
        for i in range(len(self.bias)-1):
            output = np.array(self.network[i]@output) +self.bias[i]
            output = self.activation_function(output)
            outputs[i] = output
        output = np.array(self.network[-1]@output) +self.bias[-1]
        output = self.output_activation(output)
        outputs[-1] = output
        return output, outputs
    
    def predict(self, input):
    #Makes prediction from input vector. Returns prediction.
        return(self.feedForward(input)[0])

    def backpropagation_multiple_layers_batch(self, inp, target, batch_size):
        #A method to train the MLP using backpropagation.     
        #Calculate the predicted outputs and activations by feeding the inputs forward
        gradient = []
        bias = []
        batch_size = len(inp)
        for j,k in zip(inp,target):
            pred, activations = self.feedForward(j)
            act = [0]*(len(self.network)+1)
            act[0] = j
            act[1: ] = activations
            errors = self.calculate_errors(activations, pred, k)
            for i in range(len(act)):
                act[i] = np.reshape(act[i], (1, act[i].shape[0]))
            gradient_tmp = []
            bias_tmp = []
            for i in range(1, len(errors)+1):
                gradient_tmp.append(errors[-i]@act[-i-1])
                bias_tmp.append(np.sum(errors[-i]))
            gradient.append(gradient_tmp)
            bias.append(bias_tmp)
        gradient_mean = np.copy(gradient[0])
        for bs in range(1, batch_size):
            for nw in range(len(self.network)):
                gradient_mean[nw] += gradient[bs][nw]

        for i in range(len(gradient_mean)):
            for j in range(len(gradient_mean[i])):
                gradient_mean[i][j] = gradient_mean[i][j]/batch_size
        bias_mean = np.mean(bias,axis=0)/batch_size
        for i in range(1, len(errors)+1):
            self.bias[-i] -= bias_mean[i-1]*self.learning_rate
            self.network[-i] -= gradient_mean[i-1]*self.learning_rate
    def backpropagation_multiple_layers(self, inp, target):
    #A method to train the MLP using backpropagation.     
        #Calculate the predicted outputs and activations by feeding the inputs forward
        pred, activations = self.feedForward(inp)
        act = [0]*(len(self.network)+1)
        act[0] = inp
        act[1: ] = activations
        if(self.model_type == "classifier"):
            errors = self.calculate_errors_cross_entropy(activations, pred, target)
        else:    
            errors = self.calculate_errors(activations, pred, target)
        for i in range(len(act)):
            act[i] = np.reshape(act[i], (1, act[i].shape[0]))
        for i in range(1, len(errors)+1):
            gradient = errors[-i]@act[-i-1]
            self.bias[-i] -= np.sum(errors[-i])*self.learning_rate
            self.network[-i] -= gradient*self.learning_rate

    def calculate_errors(self, activations, pred, target):
        errors = [0] * (len(self.network))
        errors[-1] = (pred-target)*self.outut_derivative(pred)
        errors[-1] = np.reshape( errors[-1], (errors[-1].shape[0], 1) )
        for i in range(2, len(self.network)+1):
            errors[-i] = (self.network[-i+1].T)@np.reshape(errors[-i+1], (errors[-i+1].shape[0], 1))*self.dfdx([-i])
        return errors

    def calculate_errors_cross_entropy(self, activations, pred, target):
        errors = [0] * (len(self.network))
        errors[-1] = (pred-target)
        errors[-1] = np.reshape( errors[-1], (errors[-1].shape[0], 1) )
        for i in range(2, len(self.network)+1):
            errors[-i] = (self.network[-i+1].T)@np.reshape(errors[-i+1], (errors[-i+1].shape[0], 1))*self.dfdx([-i])
        return errors

    def export_network(self, filename):
    #To save a network for future use.     
        pickle.dump( self, open( filename, "wb" ) )
        print("Network stored, filename: ", filename, ". To load, use pickle.load(open( filename, 'rb'))" )

    def train_method(self, train, train_targets):
        #Core training method   
        for i in range(len(train)):
            self.backpropagation_multiple_layers(train[i], train_targets[i])

    def train(self, train, train_targets, test, test_targets, partition_sizes= [0.8, 0.1, 0.1], epochs = 150, plot_training_results = False, verbose=False):
    # A method that takes a dataset, processes it, and trains until convergence criterium is met.
        training_accuracy = np.zeros(epochs)
        test_accuracy = np.zeros(epochs)
        for k in range(epochs):
            self.train_method(train, train_targets)
            training_accuracy[k], diff_training = self.loss_function(train, train_targets)
            test_accuracy[k], diff_test = self.loss_function(test, test_targets)

            if(test_accuracy[k] > self.best_accuracy):
                #Keep a copy of the best performing network
                self.best_accuracy = test_accuracy[k]
                self.best_network = copy.deepcopy(self.network)
                self.best_epoch = k
            if(verbose):
                print("Iteration %i, loss = %f" % (k, test_accuracy[k]))
                if(self.model_type == "classifier"):
                    fp = len(np.ravel(np.where(diff_test == 1)))
                    fn = len(np.ravel(np.where(diff_test == -1)))
                    print("Number of false positives: ", fp, "\nNumber of false negatives: ", fn)
        if(plot_training_results):
            self.plot_training_progress(training_accuracy, test_accuracy)     
        return(training_accuracy, test_accuracy)
        

    def plot_training_progress(self, train, test):
    #To plot a graph of the accuracy as a function of training epochs.
        if(self.loss_function.__name__ == "mean_squared_error"):
            label = "MSE"
        elif(self.loss_function.__name__ == "accuracy_score"):
            label = "Accuracy score"  
        else:
            label = "Model accuracy"
        fig1 = plt.figure()
        plt.plot(train)
        plt.plot(test)
        plt.legend(["Train", "Test", "Validation"])
        plt.xlabel("Epochs")
        plt.ylabel(label)
        plt.show()

    def accuracy_score(self, inputs, targets, cutoff = 0.5):
    #Calculates the accuracy score by counting the number of correct predicitons devided by the number of predictions.     
        N = len(inputs)
        pred = [0]*N
        # pred = self.predict(inputs)
        for j in range(N):
            pred[j] = self.predict(inputs[j])
        pred = np.ravel((np.asarray(pred) >= cutoff).astype(int))
        # pred = np.ravel(np.round(pred))
        s = len(np.ravel(np.where(pred == targets)))
        diff = pred-targets
        return(s/N, diff) 


    def validate(self):
        try:
            return self.loss_function(self.validation_data, self.validation_targets)
        except AttributeError:
            print("Please use load_data(X, y) first")
   
    def mean_squared_error(self, inputs, targets):
    #An alternative loss functino, intended for regression use.     
        predictions = np.zeros(len(inputs))
        for i in range(len(predictions)):
            predictions[i] = self.predict(inputs[i])
            # print(predictions)
        diff = predictions - targets    
        return mean_squared_error(targets, predictions), diff    

    def batch_train(self, train, train_targets, test, test_targets, batch_size, partition_sizes = [0.7, 0.2, 0.1], epochs = 150, plot_training_results = True, shuffle=True, verbose = False):
    #Batch training method. Picks a random sized batch. Calculates the average value of n vectors with average targets.
    # Trains the MLP on the batch.
        #We begin to randomize the vectors.
        training_accuracy = np.zeros(epochs)
        test_accuracy = np.zeros(epochs)
        num = len(train)//batch_size
        rest = len(train)%batch_size
        flag = (rest != 0)
        for k in range(epochs):
            if(shuffle):
                p = np.random.permutation(len(train))
                train = train[p]
                train_targets = train_targets[p]
            for i in range(num):
                inp = train[batch_size*i:batch_size*(i+1)]
                target = train_targets[batch_size*i:batch_size*(i+1)]            
                # batch_inputs = np.mean(inp, axis= 0)
                # batch_targets = np.mean(target, axis= 0)
                self.backpropagation_multiple_layers_batch(inp, target, batch_size)           
            #Finally the last batch.    
            if(flag):
                inp = train[batch_size*(i+1):batch_size*(i+2)]
                target = train_targets[batch_size*(i+1):batch_size*(i+2)]       
                # batch_inputs = np.average(inp, axis= 0)
                # batch_targets = np.average(target, axis= 0)    
                self.backpropagation_multiple_layers_batch(inp, target, batch_size)  
            training_accuracy[k] = self.loss_function(train, train_targets)
            test_accuracy[k] = self.loss_function(test, test_targets)
            if(test_accuracy[k] > self.best_accuracy):
                #Keep a copy of the best performing network
                self.best_accuracy = test_accuracy[k]
                self.best_network = copy.deep_copy(self.network)
                self.best_epoch = k
            if(verbose):
                print("Iteration %i, loss = %f" % (k, test_accuracy[k]))    
        if(plot_training_results):
            self.plot_training_progress(training_accuracy, test_accuracy)     
        return(training_accuracy, test_accuracy)

    def scaleData(self, scaling_data):
    #A method to scale data by a pre-constructed scaling transformation.    
        return self.scaler.transform(scaling_data)

    def createScaler(self, scaling_data):
    #Creates a scaler (preferably by the training data set.)    
        self.scaler = StandardScaler()
        scaling_data = self.scaler.fit_transform(scaling_data)  
        return  scaling_data

    def split_data(self, inputs, targets, pratition_sizes):
    #Takes a set of inputs and outputs, shuffles them and splits them into training, testing and validation sets. 
        test, train, test_targets, train_targets = model_selection.train_test_split(inputs, targets, test_size =pratition_sizes[0], random_state = 1)
        #we want partition_sizes[1] percent testint set. (1-partition_sizes[0])*x = partition_sizes[1]
        size = pratition_sizes[1]/(1-pratition_sizes[0])
        validation, test, validation_targets, test_targets = model_selection.train_test_split(test, test_targets, test_size =size, random_state = 1)
        return train, train_targets, test, test_targets, validation, validation_targets

    def cost_function(self, target, predicted):
    #The cost function    
        error = target - predicted 
        return 0.5*(error@error.T)





    # def relu(self, z):
    #     """
    #     Returns the ReLU-function.
    #     """
    #     z = np.ravel(z)
    #     ret  = np.zeros(len(z))
    #     for i in range(len(z)):
    #         if(6> z[i]  and z[i] >= 0 ):
    #             ret[i]=z[i]
    #         elif(6>= z[i]):
    #             ret[i]=6 
    #     return ret

    # def relu_derivative(self, z):

    #     """
    #     Returns the derivative of the ReLu-function. As this function is not
    #     differentiable in z=0, there are two nice options to deal with this.
    #     Returning an arbitrary value when z=0 (often used are 0, 0.5 and 1).
    #     Alternatively, approximate the ReLu-function with a function which is 
    #     differentiable everywhere. One such approximation is called 'softplus'
    #     and gives us the function y = ln(1 + e^z), where the derivative equals
    #     the Sigmoid-function.
    #     Here, it is used the 'arbitrary' return-value 0.5.
    #     """
    #     z = np.ravel(z)
    #     ret = np.zeros(len(z))
    #     for i in range(len(z)):
    #         if (z[i] >= 0 ):
    #             ret[i] = z[i]
    #         elif (z[i] < 0 or z[i] > 6):
    #             ret[i] = 0
    #         else:
    #             ret[i]=0.5
    #     return ret 


    # def exponential_activation(self, z):
    #     """
    #     The exponential sigmoid.
    #     """
    #     val = np.ravel(val)
    #     ret = np.zeros(len(val))
    #     for i in range(len(val)):
    #         if(val[i] >= 40 ):
    #             ret[i]=1
    #         elif (val[i]<-40):
    #             ret[i] = 0
    #         else:
    #             ret[i]= 1/(1+np.exp(-val[i]))
    #     return ret