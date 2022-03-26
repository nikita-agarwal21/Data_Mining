# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 19:46:06 2021

@author: hp
"""
#import numpy as np
import pandas as pd

dataset = pd.read_csv('F:/dm projects/ai-ml/iris.csv')

print(dataset.head())

feature_columns = ['sepal_length', 'sepal_width', 'petal_length','petal_width']
X = dataset[feature_columns].values
y = dataset['species'].values

print('input ...');
#print(X);

print('species ...');
#print(y);

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print('y after label encoding ...');
print(y);

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("y_pred    y_test")

for i in range(len(y_pred)):
    print(y_pred[i], "   ", y_test[i])

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
