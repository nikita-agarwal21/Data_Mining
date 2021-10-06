# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 20:10:24 2021

@author: hp
"""

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

data=pd.read_csv("kdrama.csv")

"""""
x=data['Rating']
y=data['Name']
plt.plot(x,y)
plt.xlabel('rating')
plt.ylabel('name')
plt.show()
"""""

name=data['Name']
year=data['Rating']

x=name[0:10]
y=year[0:10]
#color=['r','y','b','g']
plt.pie(y,labels=x,shadow=True,autopct='%1.1f%%')
sns.set(style='dark')
#sns.lineplot(x,y)
plt.title('kdrama')
plt.show()

name=data['Name']
year=data['Year of release']
x1=name[0:10]
y1=year[0:10]
plt.bar(y1,x1)
plt.xlabel('name')
plt.ylabel('year')
plt.title('kdrama')
plt.show()
