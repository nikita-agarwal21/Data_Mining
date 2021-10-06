import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("titanic.csv")
#sns.heatmap(data.isnull())
#plt.show()
def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 35
        elif Pclass==2:
            return 25
        else:
            return 20
    else:
        return Age

data['Age']=data[['Age','Pclass']].apply(impute_age,axis=1)
#sns.heatmap(data.isnull())
#plt.show()
data.drop('Cabin',axis=1,inplace=True)
data.dropna(inplace=True)
#sns.heatmap(data.isnull())
#plt.show()
sex=pd.get_dummies(data['Sex'],drop_first=True)
embarked=pd.get_dummies(data['Embarked'],drop_first=True)
data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
data=pd.concat([data,sex,embarked],axis=1)
#print(data[:5])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data.drop('Survived',axis=1),data['Survived'],
                                               test_size=0.20,random_state=3)
#print(y_test[:5])

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
pred=model.predict(x_test)
from sklearn import metrics
cnf_mx=metrics.confusion_matrix(y_test,pred)
print(cnf_mx)
sns.heatmap(pd.DataFrame(cnf_mx))
plt.show()