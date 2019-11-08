#Classification - cancer data 
#File to run calculations. 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import numpy as np
from MLP2 import neuralNetwork #Import neural network class
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
import seaborn as sns
#import relevant datasets
from sklearn.datasets import load_breast_cancer
from  sklearn.preprocessing import scale
cancer = load_breast_cancer()
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Load data
X, y = load_breast_cancer(return_X_y=True)
#Split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 69)
#Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Produces plots for section "Regression with MLP"
def a():
    models = [
    neuralNetwork([30, 60,  1], activation = "relu" , model="classifier", learning_rate = 0.1),
    neuralNetwork([30, 60, 1], activation = "leakyrelu", learning_rate= 0.1, model="classifier"),
    neuralNetwork([30, 60, 1], activation = "tanh", learning_rate= 0.1, model="classifier"),
    neuralNetwork([30, 60,  1], activation = "elu", model="classifier"),
    neuralNetwork([30, 60,  1], activation = "arctan", model="classifier"),
    neuralNetwork([30, 60,  1], activation = "tanh", model="classifier"),
    neuralNetwork([30, 60,  1], activation = "exponential", learning_rate= 0.1, model="classifier")
    ]
    for model in models:
        print("------------------")
        print("Activation function: ", model._name)
        model.train(X_train, y_train, X_test, y_test, verbose=False,  plot_training_results = True , epochs = 60 )

        model.network = model.best_network
        accuracy, diff = (model.accuracy_score(X_test, y_test))
        fp = len(np.ravel(np.where(diff == 1)))
        fn = len(np.ravel(np.where(diff == -1)))
        print("Accuracy: ", accuracy, "\nNumber of false positives: ", fp, "\nNumber of false negatives: ", fn, "\nBest accuracy: ", model.best_accuracy )
def tune_network_simple():
    treshold = 0.8
    accuracies = np.reshape(np.zeros(9), (3, 3))
    for i in range(1, 4):

        models = [
        neuralNetwork([30, 10*i, 1], activation = "relu" , model="classifier", learning_rate = 0.1),
        neuralNetwork([30, 10*i, 1], activation = "tanh", learning_rate= 0.1, model="classifier"), 
        neuralNetwork([30, 10*i, 1], activation = "exponential", learning_rate= 0.1, model="classifier")
        ]
        for j in range(3):
            model = models[j]
            print("------------------")
            print(10*i, "hidden neurons")
            print("Activation function: ", model._name)
            model.train(X_train, y_train, X_test, y_test, verbose=False,  plot_training_results = False , epochs = 40 )
            accuracy, diff = (model.accuracy_score(X_test, y_test, cutoff = 0.5))
            diff2  = (model.accuracy_score(X_test, y_test, cutoff = treshold))[1]
            accuracies[j][i-1] = model.best_accuracy
            fp = len(np.ravel(np.where(diff == 1)))
            fn = len(np.ravel(np.where(diff == -1)))
            fp2 = len(np.ravel(np.where(diff2 == 1)))
            fn2 = len(np.ravel(np.where(diff2 == -1)))
            model.network = model.best_network
            print("Accuracy: ", accuracy, "\nNumber of false positives: ", fp, "\nNumber of false negatives: ", fn, "\nBest accuracy: ", model.best_accuracy )
            print("Confusion where y< %i is rounded down to 0:" %treshold)
            print("Number of false positives: ", fp2, "\nNumber of false negatives: ", fn2, "\nBest accuracy: ")
    ax = sns.heatmap(accuracies, 
        xticklabels = ['10', '20', '30'], 
        yticklabels = ["ReLu", "tanh", "logistic"], 
        annot = True, fmt = ".3f"
    )
    bottom, top = ax.get_ylim()
    ax.set_ylim(top-0.5, bottom + 0.5)
    plt.xlabel("Number of neurons in the hidden layer")
    plt.show()


def tune_network_complex():
    print("COMPLEX MODELS:")
    treshold = 0.8
    accuracies = np.reshape(np.zeros(9), (3, 3))
    for i in range(1, 4):
        models = [
        neuralNetwork([30, 100*i, 1], activation = "relu" , model="classifier", learning_rate = 0.1),
        neuralNetwork([30, 100*i, 1], activation = "tanh", learning_rate= 0.1, model="classifier"), 
        neuralNetwork([30, 100*i, 1], activation = "exponential", learning_rate= 0.1, model="classifier")
        ]
        for j in range(3):
            model = models[j]
            print("------------------")
            print(100*i, "hidden neurons")
            print("Activation function: ", model._name)
            model.train(X_train, y_train, X_test, y_test, verbose=False,  plot_training_results = True , epochs = 40 )
            model.network = model.best_network
            accuracy, diff = (model.accuracy_score(X_test, y_test, cutoff = 0.5))
            accuracies[j][i-1] = model.best_accuracy
            diff2  = (model.accuracy_score(X_test, y_test, cutoff = treshold))[1]
            fp = len(np.ravel(np.where(diff == 1)))
            fn = len(np.ravel(np.where(diff == -1)))
            fp2 = len(np.ravel(np.where(diff2 == 1)))
            fn2 = len(np.ravel(np.where(diff2 == -1)))
            print("Accuracy: ", accuracy, "\nNumber of false positives: ", fp, "\nNumber of false negatives: ", fn, "\nBest accuracy: ", model.best_accuracy )
            print("Confusion where y< %i is rounded down to 0:" %treshold)
            print("Number of false positives: ", fp2, "\nNumber of false negatives: ", fn2, "\nBest accuracy: ")
    ax = sns.heatmap(accuracies, 
        xticklabels = ['100', '200', '300'], 
        yticklabels = ["ReLu", "tanh", "logistic"], 
        annot = True, fmt = ".3f"
    )
    bottom, top = ax.get_ylim()
    ax.set_ylim(top-0.5, bottom + 0.5)
    plt.xlabel("Number of neurons in the hidden layer")
    plt.show()

def tune_network_complex_lr():
    #To tune the learning rate for a complex model
    print("COMPLEX MODELS: Tuning learning rate")
    treshold = 0.8
    accuracies = np.reshape(np.zeros(9), (3, 3))
    for i in range(1, 4):
        models = [
        neuralNetwork([30, 200, 1], activation = "relu" , model="classifier", learning_rate = 0.05*i),
        neuralNetwork([30, 200, 1], activation = "tanh", learning_rate= 0.1, model="classifier"), 
        neuralNetwork([30, 200, 1], activation = "exponential", learning_rate= 0.1, model="classifier")
        ]
        for j in range(3):
            model = models[j]

            print("------------------")
            print("Learning rate: " ,i*0.05)
            print("Activation function: ", model._name)
            model.train(X_train, y_train, X_test, y_test, verbose=False,  plot_training_results = True , epochs = 40 )
            accuracy, diff = (model.accuracy_score(X_test, y_test, cutoff = 0.5))
            accuracies[j][i-1] = model.best_accuracy
            model.network  = model.best_network
            diff2  = (model.accuracy_score(X_test, y_test, cutoff = treshold))[1]
            fp = len(np.ravel(np.where(diff == 1)))
            fn = len(np.ravel(np.where(diff == -1)))
            fp2 = len(np.ravel(np.where(diff2 == 1)))
            fn2 = len(np.ravel(np.where(diff2 == -1)))
            print("Accuracy: ", accuracy, "\nNumber of false positives: ", fp, "\nNumber of false negatives: ", fn, "\nBest accuracy: ", model.best_accuracy )
            print("Confusion where y< %i is rounded down to 0:" %treshold)
            print("Number of false positives: ", fp2, "\nNumber of false negatives: ", fn2, "\nBest accuracy: ")
    ax = sns.heatmap(accuracies, 
        xticklabels = ['0.05', '0.1', '0.15'], 
        yticklabels = ["ReLu", "tanh", "logistic"], 
        annot = True, fmt = ".3f"
    )
    bottom, top = ax.get_ylim()
    ax.set_ylim(top-0.5, bottom + 0.5)
    plt.xlabel("Number of neurons in the hidden layer")
    plt.show()

def best_network_with_cutoff():
    #Cutoff for complex relu
    sfp = 0
    sfn = 0
    print("Relu")
    N = 10
    for i in range(0, N):
        treshold =0.5
        model = neuralNetwork([30, 200, 1], activation = "relu", learning_rate= 0.1, model="classifier")
        model.train(X_train, y_train, X_test, y_test, verbose=False,  plot_training_results = False , epochs = 20 )
        accuracy, diff = (model.accuracy_score(X_test, y_test, cutoff = 0.5))
        model.network  = model.best_network
        diff2  = (model.accuracy_score(X_test, y_test, cutoff = treshold))[1]
        fp2 = len(np.ravel(np.where(diff2 == 1)))
        fn2 = len(np.ravel(np.where(diff2 == -1)))
        sfp += fp2
        sfn += fn2
        print("Cutoff: " , treshold)
        print("Confusion where y< %i is rounded down to 0:" %treshold)
        print("Number of false positives: ", fp2, "\nNumber of false negatives: ", fn2, "\nBest accuracy: ")
    #Cutoff for simple
    print("Average false positives: ", sfp/float(N), "Average false negatives: ", sfn/float(N)  )
    sfp = 0
    sfn = 0
    print("\n\n\ntanh")
    for i in range(0, N):
        treshold = 0.5
        model = neuralNetwork([30, 20, 1], activation = "tanh", learning_rate= 0.1, model="classifier")
        model.train(X_train, y_train, X_test, y_test, verbose=False,  plot_training_results = False , epochs = 20 )
        accuracy, diff = (model.accuracy_score(X_test, y_test, cutoff = 0.5))
        model.network  = model.best_network
        diff2  = (model.accuracy_score(X_test, y_test, cutoff = treshold))[1]
        sfp += fp2
        sfn += fn2
        fp2 = len(np.ravel(np.where(diff2 == 1)))
        fn2 = len(np.ravel(np.where(diff2 == -1)))
        print("Cutoff: " , treshold)
        print("Confusion where y< %i is rounded down to 0:" %treshold)
        print("Number of false positives: ", fp2, "\nNumber of false negatives: ", fn2, "\nBest accuracy: ")
    print("Average false positives: ", sfp/float(N), "Average false negatives: ", sfn/float(N)  )
# a()

# tune_network_simple()
# plt.show()
# tune_network_complex()
# tune_network_complex_lr()
best_network_with_cutoff()
