#!/usr/bin/env python
# coding: utf-8
print('NAME:nikita agarwal USN:1JT18CS037 \n');

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv("Admission_Predict_Ver1.1.csv")

print(df.head())
#print(df.info())

x = df.iloc[:, 1:8] 
y = df.iloc[:, 8]     

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y)

#print(x_train)

from sklearn import svm
clf = svm.SVR(gamma='auto')
clf.fit(x_train, y_train)
label=[]
accuracy=[]
label.append('SVR')
accuracy.append(clf.score(x_test, y_test))
print(clf.score(x_test, y_test))

from sklearn import linear_model
clf = linear_model.Ridge(alpha=.5)
clf.fit(x_train, y_train)
label.append('Ridge')
accuracy.append(clf.score(x_test, y_test))
print(clf.score(x_test, y_test))


clf = linear_model.BayesianRidge()
clf.fit(x_train, y_train)
label.append('BayesianRidge')
accuracy.append(clf.score(x_test, y_test))
print(clf.score(x_test, y_test))


clf.predict(x_test[10:20])
y_test[15:25]

import matplotlib.pyplot as plt
import numpy as np
index = np.arange(len(label))
def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.bar(index, accuracy)
    plt.xlabel('Model', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.xticks(index, label, fontsize=10, rotation=90)
    plt.title('Accuracy of different models')
    plt.savefig("model_accuracy.png")
    plt.show()

plot_bar_x()




