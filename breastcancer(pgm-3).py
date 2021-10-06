# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 17:31:52 2021

@author: hp
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("BreastCancerPrediction.csv")
cols=['Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses']
x=data[cols]
y=data.Class

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import MinMaxScaler
norm=MinMaxScaler().fit(x_train)
x_train_norm=norm.transform(x_train)
x_test_norm=norm.transform(x_test)
#print(x_test_norm[:5])


print("Before scaling")
print(x_test[:5])
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
print("after scaling")
print(x_test[:5])


from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(x_train_norm,y_train)
y_pred=log.predict(x_test_norm)
#print(y_pred)
#print("actual vs predicted")
#print(pd.DataFrame(y_pred,y_test))
from sklearn import metrics
cnf=metrics.confusion_matrix(y_test,y_pred)
print("cnf_mx",cnf)
sns.heatmap(pd.DataFrame(cnf),annot=True)
plt.title("Confusion matrix")
plt.xlabel("actual value")
plt.ylabel("predicted value")
plt.show()
print("Accuracy score:",metrics.accuracy_score(y_test,y_pred))