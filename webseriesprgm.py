# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 21:23:50 2021

@author: hp
"""


from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

data=pd.read_csv("webseries.csv")

#pie chart for title and rating
title=data['Series Title']
rating=data['IMDB Rating']
x=title[0:10]
y=rating[0:10]
plt.pie(y,labels=x,shadow=True,autopct='%1.1f%%')
sns.set(style='dark')
#sns.lineplot(x,y)
plt.title('web series')
plt.show()

#bar graph for genre and the rating
genre=data['Genre']
rating=data['IMDB Rating']
x=rating[0:10]
y=genre[0:10]
plt.bar(x,y)
plt.xlabel('rating')
plt.ylabel('Genre')
plt.title('web series')
plt.show()

#subplots for comparison between graph of plots basedon series released year,rating given for a particular series  
year=data['Year Released']
rating=data['IMDB Rating']
x1=year[0:10]
y1=rating[0:10]
plt.subplot(221)
plt.plot(y1,x1)
plt.xlabel('rating')
plt.ylabel('year')
plt.title('web series')

title=data['Series Title']
rating=data['IMDB Rating']
x=title[0:10]
y=rating[0:10]
plt.subplot(223)
plt.plot(y,x)
plt.xlabel('rating')
plt.ylabel('title')

plt.show()


