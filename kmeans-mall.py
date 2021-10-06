# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 16:41:59 2021

@author: hp
"""
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("mallCustomerpredict.csv")
'''
'''
#kmeans clustering
#to find k using elbow method
x=data.iloc[:,[3,4]].values
from sklearn.cluster import KMeans
wcs=[]#variance list
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=100,n_init=10,random_state=0)#rrandom points choosed
    kmeans.fit(x)
    wcs.append(kmeans.inertia_)#variance data is added   
'''
'''
plt.plot(range(1,11),wcs,'-o')
plt.xlabel('k value')
plt.ylabel('variance')
plt.title('elbow method to find k')
plt.show()
'''
'''
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=100,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)#which cluster x belonged to is predicted
print(y_kmeans)

plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=10,c='green',label='cluster 1')#x,y cluster is mentioned
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=10,c='blue',label='cluster 2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=10,c='red',label='cluster 3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=10,c='yellow',label='cluster 4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=10,c='black',label='cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=30,c='cyan',label='centroid')
plt.legend()
plt.xlabel('annual income')
plt.ylabel('spending')
plt.show()

'''

#k-means clustering

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

data=pd.read_csv("mallCustomer.csv")

x=data.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
variance=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    variance.append(kmeans.inertia_)

'''plt.plot(range(1,11),variance,'-o')
plt.title("Elbow method")
plt.xlabel("K Value")
plt.ylabel("Variance")
plt.show()'''

kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
pred=kmeans.fit_predict(x)
plt.style.use('dark_background')
plt.scatter(x[pred==0,0],x[pred==0,1],s=70,c='r',label='cluster1')
plt.scatter(x[pred==1,0],x[pred==1,1],s=70,c='g',label='cluster2')
plt.scatter(x[pred==2,0],x[pred==2,1],s=70,c='b',label='cluster3')
plt.scatter(x[pred==3,0],x[pred==3,1],s=70,c='white',label='cluster4')
plt.scatter(x[pred==4,0],x[pred==4,1],s=70,c='cyan',label='cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,c='yellow',label='centroid')
plt.legend()
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

target=pd.DataFrame({'Target':kmeans.labels_})
new_data=pd.concat([data,target],axis=1,sort=False)
#print(new_data.head())

x_new=new_data.drop(['Target'],axis=1)
y_new=new_data['Target']

gen=pd.get_dummies(x_new['Gender'])
x_new=x_new.drop(['Gender'],axis=1)
x_new=pd.concat([x_new,gen],axis=1,sort=False)

x_train,x_test,y_train,y_test=train_test_split(x_new,y_new,test_size=0.20,random_state=2)


dt=DecisionTreeClassifier()
rf=RandomForestClassifier()

model_dt=dt.fit(x_train,y_train)
model_rf=rf.fit(x_train,y_train)

y_pred=model_dt.predict(x_test)
df=pd.DataFrame(y_pred,y_test)
print('decison tree predicted values')
print(df[:6])
print()
y_pred1=model_rf.predict(x_test)
rf1=pd.DataFrame(y_pred1,y_test)
print('random forest  predicted values')
print(rf1[:6])

from sklearn import metrics
print("Decision Tree Scores")
print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))
print("MAE (test): ",metrics.mean_absolute_error(y_test, y_pred))
print("MSE (test): ",metrics.mean_squared_error(y_test, y_pred))
print()
print("Random Forest Scores")
print("Accuracy: ",metrics.accuracy_score(y_test,y_pred1))
print("MAE (test): ",metrics.mean_absolute_error(y_test, y_pred1))
print("MSE (test): ",metrics.mean_squared_error(y_test, y_pred1))

