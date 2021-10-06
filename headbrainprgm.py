# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 18:07:19 2021

@author: hp
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as mp

dataset=pd.read_csv('headbrain.csv')
x=dataset.iloc[:,:-1].values#x dataset is always everything except one column 
y=dataset.iloc[:,3].values



from sklearn.model_selection import train_test_split#to train test data

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
#print(x_test)
#print(y_test)

original =pd.DataFrame(x_test,y_test)#array representation 1st column-y_test
#print (original)

from sklearn.linear_model import LinearRegression#to draw a logic between x,y 

regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)#gives d predicted data
#print(y_pred)

compare =pd.DataFrame(y_test,y_pred)#array representation 1st column-y_pred
print (compare)


print(regressor.score(x_test,y_test))#accuracy

