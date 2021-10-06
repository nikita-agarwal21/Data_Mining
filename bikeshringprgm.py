# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 12:32:22 2021

@author: hp
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression

data=pd.read_csv('train.csv')

x=data.iloc[:,:-1].values
y=data.iloc[:,11]
A=make_column_transformer((OneHotEncoder(categories='auto'),[0]),remainder='passthrough')
x=A.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)

print("MSE: ",mean_squared_error(y_test,y_pred))
print(reg.score(x_test,y_test))

plt.plot(y_test,y_pred)
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.show()

#comparison between registered and count
xw=data['registered'].values
yc=data['count'].values
xw=xw.reshape(-1,1)
yc=yc.reshape(-1,1)
xw_train,xw_test,yc_train,yc_test=train_test_split(xw,yc,test_size=0.20,random_state=0)

reg=LinearRegression()
reg.fit(xw_train,yc_train)
yc_pred=reg.predict(xw_test)
print("MSE registered and count: ",mean_squared_error(yc_test,yc_pred))
#print(yc_pred)

plt.scatter(xw_train,yc_train,color='red')
plt.plot(xw_train,reg.predict(xw_train),color='black')
plt.xlabel('registered',fontsize = 18)
plt.ylabel('count',fontsize = 18)
plt.title('registered vs count',fontsize = 18)

plt.show()


#comparison between temperature and count
xw=data['temp'].values
yc=data['count'].values
xw=xw.reshape(-1,1)
yc=yc.reshape(-1,1)
xw_train,xw_test,yc_train,yc_test=train_test_split(xw,yc,test_size=0.20,random_state=0)

reg=LinearRegression()
reg.fit(xw_train,yc_train)
yc_pred=reg.predict(xw_test)
print("MSE temperature and count: ",mean_squared_error(yc_test,yc_pred))
#print(yc_pred)

plt.scatter(xw_train,yc_train,color='yellow',s=5)
plt.plot(xw_train,reg.predict(xw_train),color='black')
plt.xlabel('temperature',fontsize = 18)
plt.ylabel('count',fontsize = 18)
plt.title('temperature vs count',fontsize = 18)
plt.show()


#comparison between humidity and count
xw=data['humidity'].values
yc=data['count'].values
xw=xw.reshape(-1,1)
yc=yc.reshape(-1,1)
xw_train,xw_test,yc_train,yc_test=train_test_split(xw,yc,test_size=0.20,random_state=0)

reg=LinearRegression()
reg.fit(xw_train,yc_train)
yc_pred=reg.predict(xw_test)
print("MSE humididty and count: ",mean_squared_error(yc_test,yc_pred))
#print(yc_pred)

plt.scatter(xw_train,yc_train,color='blue')
plt.plot(xw_train,reg.predict(xw_train),color='black')
plt.xlabel('humidity',fontsize = 18)
plt.ylabel('count',fontsize = 18)
plt.title('humidity vs count',fontsize = 18)
plt.show()

