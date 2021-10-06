# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 21:57:22 2021

@author: hp
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("mallCustomer.csv")
'''
sns.heatmap(data.isnull())
plt.show()

'''

'''
#piechart for gender
gen=['Male','Female']
size=data['Gender'].value_counts()
plt.pie(size,labels=gen,autopct="%1.1f%%",explode=(0,0.1),shadow=True)
plt.title('Gender Count')
plt.show()



#histogram for age
age=data['Age']
range1=[18,20,25,30,35,45,50,60]
plt.hist(age,range1,histtype='bar',rwidth=0.8,color='b')
plt.title('Visualization of Age')
plt.xlabel('Age')
plt.ylabel('count')
plt.show()
'''

'''
#histogram fro annual income
ann=data['Annual_Income']
range1=[15,25,35,45,55,65,75,85,95,105,120,130,140]
plt.hist(ann,range1,histtype='bar',rwidth=0.8,color='r')
plt.title('Visualization of Annual income')
plt.xlabel('Annual income')
plt.ylabel('count')
plt.show()



#histogram for spending 
ss=data['Spending_Score ']
range1=[0,10,20,30,40,50,60,70,80,90,100]
plt.hist(ss,range1,histtype='bar',rwidth=0.8,color='y')
plt.title('Visualization of Spending score')
plt.xlabel('Spending score')
plt.ylabel('count')
plt.show()




#subplots between income nd spending and age
income=data['Annual_Income']
spending=data['Spending_Score ']
x1=income[0:10]
y1=spending[0:10]
plt.subplot(331)
plt.scatter(y1,x1,s=10)
plt.xlabel('income')
plt.ylabel('spending')
plt.title('comparing spending vs income and age')

age=data['Age']
spending=data['Spending_Score ']
x=age[0:10]
y=spending[0:10]
plt.subplot(336)
plt.scatter(y,x,s=10)
plt.xlabel('age')
plt.ylabel('spending')

plt.show()
'''

'''
#countplot for age
age=data['Age']
#range1=[18,20,25,30,35,45,50,60]
sns.countplot(data['Age'],palette='hsv',order=[18,22,25,28,30,32,35,38,42,46,50,52,56,60,65,70])
plt.title("Visualization of Age")
plt.show()
'''

'''
#comparison between all d column
plt.figure(figsize=(10,8), dpi= 80)
sns.pairplot(data, kind="scatter", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
plt.show()
'''

'''
#countplot for spending score
sns.countplot(data['Spending_Score '],palette='copper',order=[3,6,9,12,15,18,24,27,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,87,90,93,99])
plt.title("Visualization of Spendingscore")
plt.show()
'''

'''
#stripplot for comparing gender vs ss
sns.stripplot(data['Gender'],data['Spending_Score '],palette='Blues',size=5)
plt.title('gender vs spending score')
plt.show()
'''

'''
x=data['Age']
y=data['Spending_Score ']
y1=data['Annual_Income']
plt.scatter(x,y,color='g',label='age vs spending')
plt.scatter(x,y1,color='r',label='age vs income')
plt.legend()
plt.show()

sns.lmplot(x='Annual_Income',y='Spending_Score ',data=data)
plt.show()
'''

'''
#data cleaning -gender col dataa to numeric
gender=pd.get_dummies(data['Gender'],drop_first=True)
data.drop(['Gender'],axis=1,inplace=True)
data=pd.concat([data,gender],axis=1)
#print(data[:5])
gen=['Male','Female']
size=data['Male'].value_counts()
plt.pie(size,labels=gen,autopct="%1.1f%%",explode=(0,0.1),shadow=True)
plt.title('Gender Count')
plt.show()
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
plt.plot(range(1,11),wcs,'-o')
plt.xlabel('k value')
plt.ylabel('variance')
plt.title('elbow method to find k')
plt.show()


'''
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=100,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)#which cluster x belonged to is predicted
#print(y_kmeans)

#y_pred1=regressor.predict([[1.0,0.0,0.0,2000,1000,3000]])
#print(y_pred1)



plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=10,c='green',label='cluster 1')#x,y cluster is mentioned
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=10,c='blue',label='cluster 2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=10,c='red',label='cluster 3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=10,c='yellow',label='cluster 4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=10,c='black',label='cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=30,c='cyan',label='centroid')

plt.legend()
plt.title('clusters of customers')
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.show()


'''
#linear regression part

from matplotlib import style

col=['Annual_Income']
col1=['Spending_Score ']
     
#x=data[col]           
 
#y=data[col1]   
x=data.iloc[:,:-1]        
y=data.iloc[:,4]

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
A=make_column_transformer((OneHotEncoder(categories="auto"),[1]),remainder="passthrough")
x=A.fit_transform(x)

#print(x),using all cols of x

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=3)


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
#print(y_pred)

df=pd.DataFrame(y_pred,y_test)                 #for all cols
print('Actual vs predicted using linear Regression')
#print(df[:10])

plt.style.use('dark_background')
plt.scatter(x_train,y_train,c='yellow')
plt.plot(x_train,reg.predict(x_train),c='r')
plt.title("Actual vs Predicted using linear Regression")
plt.xlabel("Annual_Income")
plt.ylabel("Spending Score")

plt.show()
'''
