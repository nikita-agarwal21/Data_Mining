# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 15:42:23 2021

@author: hp
"""

"""multiple regression"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as mp

dataset=pd.read_csv('startups.csv')
x=dataset.iloc[:,:-1].values#x dataset is always everything except one column 
y=dataset.iloc[:,4].values#last column

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

#make_column_tranformer is used to encode a entiire column specifies in [] 
#categories-auto mean generat new column nd the name will be automatically selected 
#passthrough-keep d remaining column as it is
A=make_column_transformer((OneHotEncoder(categories='auto'),[3]),remainder='passthrough')#use 3rd column-states nd convert it into numerical form
value=A.fit_transform(x)
#print(value) #the 3rd column as numerical,r&d spend,adminstration,marketing spend
#print(dataset)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression#to draw a logic between x,y 

x_train,x_test,y_train,y_test=train_test_split(value,y,test_size=0.20,random_state=0)#random state to keep the test data same
#print(x_test)
#print(y_test)

original =pd.DataFrame(x_test,y_test)#array representation 1st column-y_test
#print (original)

regressor=LinearRegression()#function call to predict data
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(y_pred)
print(y_test)
print(pd.DataFrame(y_test,y_pred))

y_pred1=regressor.predict([[1.0,0.0,0.0,2000,1000,3000]])
print(y_pred1)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))
print(regressor.score(x_test,y_test))







