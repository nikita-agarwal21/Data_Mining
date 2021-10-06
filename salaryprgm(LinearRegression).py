# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 14:20:58 2021

@author: hp
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as mp

dataset=pd.read_csv('salary.csv')
x=dataset.iloc[:,:-1].values#x dataset is always everything except one column 
y=dataset.iloc[:,1].values
"""
plt.scatter(x,y)
plt.xlabel('yr of exp')
plt.ylabel('salary')
plt.title('datagraph')
plt.show()

"""

from sklearn.model_selection import train_test_split#to train test data

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
#print(x_test)
#print(y_test)

original =pd.DataFrame(x_test,y_test)#array representation 1st column-y_test
#print (original)

from sklearn.linear_model import LinearRegression#to draw a logic between x,y 

regressor=LinearRegression()#function call to predict data
regressor.fit(x_train,y_train)#to find relation between x,y data passed to algorithm

y_pred=regressor.predict(x_test)#gives d predicted data
#print(y_pred)

compare =pd.DataFrame(y_test,y_pred)#array representation 1st column-y_pred
print (compare)


#print(regressor.score(x_test,y_test))#accuracy


#y_pred1=regressor.predict([[19]])#to predict data x_test value-yr of exp
#print(y_pred1)

#print(regressor.intercept_)
#print(regressor.coef_)

plt.scatter(x_train,y_train)
plt.plot(x_train,regressor.predict(x_train),c='red')#plots y_pred data against our x_train
plt.show()

