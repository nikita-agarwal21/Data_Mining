# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 12:31:34 2021

@author: hp
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data=pd.read_csv('advertisement.csv')
cols=['TV','radio','newspaper']
x=data[cols]
y=data.sales



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression#to draw a logic between x,y 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)#random state to keep the test data same
#print(x_test)
#print(y_test)

original =pd.DataFrame(x_test,y_test)#array representation 1st column-y_test
#print (original)

regressor=LinearRegression()#function call to predict data
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

print(pd.DataFrame(y_pred,y_test))


from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_test, y_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))





