# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 00:15:30 2021

@author: hp
"""
#Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome

import pandas as pd
import matplotlib.pyplot as plt
import numpy as mp
data=pd.read_csv('diabetes.csv')
feature_cols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
x=data[feature_cols]
y=data.Outcome


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression() 
logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)
#print(y_pred)
#print(pd.DataFrame(y_pred,y_test))

from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)