# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 17:06:19 2021

@author: hp
"""
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as nm
data=pd.read_csv("breastcancer.csv")
x=data.iloc[:,0:4]
y=data.iloc[:,5]

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

gender=make_column_transformer((OneHotEncoder(categories='auto'),[1]),remainder='passthrough')
x=gender.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
"""
"""from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(x_train,y_train)
y_pred=log.predict(x_test)
"""
"""
from sklearn.linear_model import LogisticRegression
logModel=LogisticRegression(random_state=0)
logModel.fit(x_train,y_train)
print(logModel.score(x_test,y_test))

y_pred=logModel.predict(x_test)
#print(pd.DataFrame(y_test,y_pred))

from sklearn import metrics
cnf=metrics.confusion_matrix(y_test,y_pred)
import seaborn as sns
sns.heatmap(pd.DataFrame(cnf),annot=True)
plt.title("confusion matrix")
plt.xlabel("actual label")
plt.ylabel("predicted label")
plt.show()
print(metrics.accuracy_score(y_test,y_pred))
print(metrics.precision_score(y_test,y_pred))
print(metrics.recall_score(y_test,y_pred))

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train1=scaler.transform(x_train)
x_test1=scaler.transform(x_test)
print(x_train1)
"""
