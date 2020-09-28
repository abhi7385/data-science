# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#
dataset = pd.read_csv('G://Mall_Customers.csv')


dataset.head()

X = dataset.iloc[:,[3, 4]].values
#EDA
np.isnan(X).sum()

# Standarize features

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_std = sc.fit_transform(X)

# K=3 than we can build alg, but 
# lets search  vbest k , i.e. elobow tech

from sklearn.cluster import KMeans

addk = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    addk.append(kmeans.inertia_)
plt.plot(range(1, 11), addk)
plt.title('Elbow Method for best K in K means')
plt.xlabel('Number of k')
plt.ylabel('SSD')
plt.show()

# Fitting K-Means to the data set

kmeans4 = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
kmeans5 = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
kmeans6 = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)


kmeans4.fit_transform(X)
kmeans5.fit_transform(X)
kmeans6.fit_transform(X)

y_kpred = kmeans5.fit_transform(X)

kmeans4.labels_
kmeans5.labels_
kmeans6.labels_

#####################################3

# Creation of new opbservation in run time

new_observation = [[43, 67]]

kmeans4.predict(new_observation)
kmeans5.predict(new_observation)
kmeans6.predict(new_observation)

################################
print("*************************")
kmeans4.cluster_centers_
print("*************************")
kmeans5.cluster_centers_
print("*************************")
kmeans6.cluster_centers_
print("*************************")


plt.scatter(X[:, 0], X[:, 1], s = 50, c='b')
plt.scatter(55.2962963 , 49.51851852, s = 200, c ='g', marker='s',label = 'Cluster 1')
plt.scatter(88.2       , 17.11428571, s = 200, c= 'r', marker='d',label = 'Cluster 2')
plt.scatter(26.30434783, 20.91304348, s= 200, c= 'y', marker='+',label = 'Cluster 3')
plt.scatter(25.72727273, 79.36363636, s = 200, c= 'g', marker='o' , label = 'Cluster 4')
plt.scatter(86.53846154, 82.12820513, s = 200 , c = 'y', marker='>', label = 'Cluster 5')

plt.show()


#plt.scatter(X[y_kpred == 0, 0], 
#X[y_kpred == 0, 1], s =100, c='red',label = 'Cluster 1')
#plt.show()











































    


























