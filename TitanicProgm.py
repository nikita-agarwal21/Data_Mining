# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 01:11:06 2021

@author: hp
"""

import pandas as pd
import numpy as mp
import matplotlib.pyplot as plt
import seaborn as snb

data=pd.read_csv('titanic.csv')
#snb.heatmap(data.isnull())
#plt.show()

#putting dummy value
def impute_age(cols):#col1-age;col2-passenger class
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age

data['Age']=data[['Age','Pclass']].apply(impute_age,axis=1)#column,,arg passed,fn_name,column wise operation
#snb.heatmap(data.isnull())
#plt.show()

#remove column
data.drop('Cabin',axis=1,inplace=True)#after removing d col store it in the data only
#snb.heatmap(data.isnull())
#plt.show()

#remove rows
data.dropna(inplace=True)#drop rows which is empty within a cell
#snb.heatmap(data.isnull())
#plt.show()

#alphabets into numbers 
sex=pd.get_dummies(data['Sex'],drop_first=True)
#based on d no of categories columns are made then one of the column is removed for  simplicity
embark=pd.get_dummies(data['Embarked'],drop_first=True)
#print(sex)#m/f
#print(embark)#p,q,s

data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
data=pd.concat([data,sex,embark],axis=1)#add editted sex,embark col to data


x=data.drop('Survived',axis=1)
y=data['Survived']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
                                                  
                                                  
#x_train,x_test,y_train,y_test=train_test_split(data.drop('Survived',axis=1),data['Survived'],test_size=0.20,random_state=3)
                                                   #xdata-except survived,ydata=survived
                                                  
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression() 
logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)
#print(y_pred)
#print(pd.DataFrame(y_pred,y_test))

from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

snb.heatmap(pd.DataFrame(cnf_matrix),annot=True,fmt='g')
plt.show()
