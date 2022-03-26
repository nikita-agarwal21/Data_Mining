# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 19:53:39 2021

@author: hp
"""

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([5,9,8,4,7,2,3,1,6,4,6])
x2 = np.array([5,1,2,6,3,8,7,9,4,6,4])

# another set of data
#x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
#x2 = np.array([5, 4, 6, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])

print('x1 : ')
print(x1)

print('x2 : ')
print(x2)

plt.plot()
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Dataset')
plt.scatter(x1, x2)
plt.show()
# create new plot and data
#plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
print('X')

print(X)

colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']
# KMeans algorithm
K = 3
kmeans_model = KMeans(n_clusters=K).fit(X)
plt.plot()
for i, l in enumerate(kmeans_model.labels_):
  plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l],ls='None')
  plt.xlim([0, 10])
  plt.ylim([0, 10])
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

