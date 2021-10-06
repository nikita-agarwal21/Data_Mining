# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 01:02:02 2021

@author: hp
"""

"""
from matplotlib import pyplot as plt
#from matplotlib import style

x=[1,2,3,4,5]
y=[5,10,15,20,25]#same size dimensions length
#plt.plot(x,y)   #can mention only  color
plt.scatter(x,y,c='r',s=50,marker='*') #color,sixe,markers

plt.xlabel('age')
plt.ylabel('years')

plt.show()
"""

"""
#style.use('dark_background')#also ('ggplot)
x1=[1,2,3,4,5]
y1=[5,10,20,14,50]
x2=[1,7,3,9,5]
y2=[10,20,30,24,50]
plt.plot(x1,y1,c='b',label='x1 vs y1')
plt.plot(x2,y2,c='r',label='x2 vs y2')
plt.title('multi graph')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend()#to display line name like scale the lbel mentioned in plot
plt.grid('True',color='white')#grid view
plt.show()
"""

#histogram
from matplotlib import pyplot as plt
population=[20,27,50,12,13,54,65,67,84,10,15,54,59,68,84,41,42,98]
age_range=[0,10,20,30,40,50,100]
#plt.hist(population,age_range,histtype='bar',rwidth=0.8)
plt.hist(population,age_range,histtype='stepfilled',rwidth=0.8)
plt.xlabel('age group')
plt.ylabel('no of people')
plt.title('histogram')
plt.show()


"""
#piechart
from matplotlib import pyplot as plt
activities=['sleeping','eating','working','drawing']
duration=[7,2,21,3]
color=['r','y','b','g']
plt.pie(duration,labels=activities,colors=color,shadow=True,autopct='%1.1f%%',explode=(0.2,0.2,0.2,0.2))
plt.title('my activity')
plt.show()
"""

"""
#multigraph
from matplotlib import pyplot as plt
x1=[1,2,3,4,5]
x2=[2,4,6,8,10]
y1=[10,20,30,40,60]
y2=[5,10,15,20,25]
x3=[10,20,30,40,60]
y3=[5,10,15,20,25]
plt.subplot(331)#no of rows,column,at which position should it be plotted
plt.plot(x1,y1)
plt.title('multi graph')
plt.subplot(332)
plt.plot(x3,y3)
plt.subplot(339)
plt.plot(x2,y2)
plt.show()
"""
