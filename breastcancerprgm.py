# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 12:32:32 2021

@author: hp
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as mp

#from sklearn.datasets import load_breast_cancer
#df=load_breast_cancer()
#print(df.feature_names)

data=pd.read_csv('data.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

gender=make_column_transformer((OneHotEncoder(categories='auto'),[1]),remainder='passthrough')
x=gender.fit_transform(x)
print(x)



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)

"""
from sklearn.preprocessing import MinMaxScaler
norm=MinMaxScaler().fit(x_train)
x_train_norm=norm.transform(x_train)
x_test_norm=norm.transform(x_test)
print(x_train_norm)
"""

#applying scaling on input param
from sklearn.preprocessing import StandardScaler#subtracting the mean nd scaling to unit variance(divide all values by standard deviation)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
logModel=LogisticRegression(random_state=0)
logModel.fit(x_train,y_train)
print(logModel.score(x_test,y_test))

y_pred=logModel.predict(x_test)
#print(pd.DataFrame(y_test,y_pred))

from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix)


import seaborn as snb
snb.heatmap(pd.DataFrame(cnf_matrix),annot=True,fmt='g')
plt.title('confusion matrix',fontsize=12)
plt.xlabel('actual label')
plt.ylabel('predicted label')
plt.show()