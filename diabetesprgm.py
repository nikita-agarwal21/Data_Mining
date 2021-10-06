# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 23:08:46 2021

@author: hp
"""
import pandas as pd
data=pd.read_csv('diabetes.csv')
#print(type(data))
#print(data['Insulin'])
count=0
for i in data['BMI']:
    if(i>25):
        count+=1
print(str(count) + "people are obbesed")
for i in data['Outcome']:
    if(i==1):
        count+=1
print(str(count) + "people are diabetic")

print(" ")

mylist=['Glucose','Insulin']
data2=data[mylist]
print(data2)

print(" ")

data1=data.iloc[:,[0,1,2]]#first,2nd,3rd  row
print(data1)
data3=data.iloc[::,[-1]]#last row
print(data3)


from matplotlib import pyplot as plt
from matplotlib import style
import pandas as pd

data= pd.read_csv("diabetes.csv")
x=data['Glucose']
y=data['Outcome']
plt.scatter(x,y)
plt.xlabel('glucose')
plt.ylabel('outcome')
plt.show()