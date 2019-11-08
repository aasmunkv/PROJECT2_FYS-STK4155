# part a

import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score as sklearn_accuracy
import logreg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# setting up data
cancer = load_breast_cancer()
X = cancer.data # design matrix
y = cancer.target # target values

# seeding that gives the results in the report, numpy seed 20 in logreg.py and
# random state 0 in train_test_split

# splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# scaling data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# epochs and etas to test
epochs = [i for i in range(1, 102, 5)]
etas = [10**-i for i in range(0,5)]

# matrices to store all the accuracy scores
accuracy_train = np.zeros((len(epochs), len(etas)))
accuracy_test = np.zeros((len(epochs), len(etas)))


# testing own logreg code for all combinations
clf = logreg.LogReg()

for i in range(len(epochs)):
    for j in range(len(etas)):
        beta, beta_hist, cost_hist = clf.fit(X_train, y_train, num_epochs=epochs[i], eta=etas[j])
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)

        accuracy_test[i][j] = logreg.accuracy_score(y_test, y_pred_test)
        accuracy_train[i][j] = logreg.accuracy_score(y_train, y_pred_train)


# plotting train vs test accuracy for eta=1,0.1,0.001 as function of epochs
plt.figure(1)
plt.plot(epochs, accuracy_train[:,0], "c", label="train, eta=1")
plt.plot(epochs, accuracy_test[:,0], "c--", label="test, eta=1")
plt.plot(epochs, accuracy_train[:,1], "m", label="train, eta=0.1")
plt.plot(epochs, accuracy_test[:,1], "m--", label="test, eta=0.1")
plt.plot(epochs, accuracy_train[:,2], "C1", label="train, eta=0.01")
plt.plot(epochs, accuracy_test[:,2], "C1--", label="test, eta=0.01", )
plt.ylabel('Accuracy')
plt.xlabel('Number of epochs')
plt.xticks(np.arange(min(epochs), max(epochs)+1, 5)) # ticks for every epoch in epochs
plt.legend()

# plotting test accuracies as function of etas and epochs
plt.figure(2)
ax = sns.heatmap(
    accuracy_test,
    xticklabels=np.log10(etas),
    yticklabels=epochs,
    annot=True,
    fmt=".3f")
ax.invert_yaxis()
bottom, top = ax.get_ylim()
ax.set_ylim(bottom - 0.5, top + 0.5)
plt.xlabel('log(learning rate)')
plt.ylabel('Number of epochs')
plt.show()

# chosen model for benchmarking, epochs=31 and eta=0.1
beta, beta_hist, cost_hist = clf.fit(X_train, y_train, num_epochs=31, eta=0.1)
y_pred = clf.predict(X_test)
acc = logreg.accuracy_score(y_test, y_pred)

# counting true predictions
fp=fn=pos=neg=0

for i in range(len(y_pred)):
    # false negatives
    if y_test[i]==1 and y_test[i]!=y_pred[i]:
        fn += 1
    # false positives
    if y_test[i]==0 and y_test[i]!=y_pred[i]:
        fp += 1
    # number of positives and negatives in test set
    if y_test[i]==1:
        pos +=1
    else:
        neg +=1

# model stats to be put into table
model_stats = [31, 0.1, acc, fp, fn]
#model_stats = np.zeros((2,5))
#model_stats[0,0] = 31
#model_stats[0,1] = 0.1
#model_stats[0,2] = acc
#model_stats[0,3] = fp
#model_stats[0,4] = fn

# making LaTex table for accuracy scores
df = pd.DataFrame.from_records([model_stats],
                   columns=['Number of epochs','Learning rate', 'Accuracy',
                                               'False positives', 'False negatives'])

tab = df.to_latex(index=False, float_format="%.2f")
print(f"\n\n{tab}\n\n")


# comparison to scikit-learn using LogisticRegression class
logreg2 = LogReg()
logreg2.fit(X_train, y_train)
y_pred = logreg2.predict(X_test)
print("scikit-learn accuracy: {:.5f}".format(sklearn_accuracy(y_test, y_pred)))
