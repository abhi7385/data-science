# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


dfw = pd.read_csv('G:\Traning\AlgoRithm\PCA/wine.data',header=None)

dfw.head()

dfw.columns = ['Class label','Alcohal','Malic acid','Ash','Alcalinity of ash','Magnesium',
 'Total Phenols','Flavanoids','Nonflavanoid phenols',
 'Proant','Color Intensity', 'Hue','OD280 of diluted','Proline']


X, y = dfw.iloc[:,1:].values, dfw.iloc[:,0].values

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)


covariant_matrix = np.cov(X_train_std.T)
#13 *13


#covariant_matrix[0::5]

eigen_values, eigen_vactors = np.linalg.eig(covariant_matrix)



eigen_values, eigen_vactors[::5]


import matplotlib.pyplot as plt

tot = sum(eigen_values)

var_exp = [(i/tot) for i in sorted(eigen_values, reverse= True)]

cum_var_exp = np.cumsum(var_exp)


plt.bar(range(1,14),var_exp, alpha=0.5, align='center',label='Individual variance' )
plt.step(range(1,14),cum_var_exp, where= 'mid',label='cumulative variance')

plt.ylabel('var ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()


eigen_pairs = [(np.abs(eigen_values[i]),eigen_vactors[:,i]) for i in range(len(eigen_values))]

eigen_pairs.sort(reverse=True)

eigen_pairs[:5]


w = np.hstack((eigen_pairs[0][1][:,np.newaxis],eigen_pairs[1][1][:,np.newaxis]))


w.shape

X_train_std[0]
 X_train_std[0].dot(w)

X_train_pca = X_train_std.dot(w)


colors = ['r','b','g']
markers = ['s','x','o']

for l,c, m in zip(np.unique(y_train),colors ,markers):
    plt.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l,1], c=c, label=l, marker=m)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(loc='best')
plt.show()



from sklearn.decomposition import PCA

pca = PCA(n_components=2)

from sklearn.linear_model import LogisticRegression


ltmodel = LogisticRegression()

X_train_std_pca = pca.fit_transform(X_train_std)
X_test_std_pca = pca.transform(X_test_std)

#y_pred =ltmodel.fit(X_train_std_pca,y_train)

ltmodel.fit(X_train_std_pca,y_train)

decision_regions(X_train_std_pca, y_train, classifier = ltmodel )

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(loc='best')
plt.show()



from matplotlib.colors import ListedColormap

def decision_regions(X, y, classifier, res = 0.02):
    
    markers = ('s','x','o','^','v')
    colors = ('red','blue','green','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    
    x1_min, x1_max = X[:,0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:, 1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, res), np.arange(x2_min, x2_max, res))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl ,1] , alpha =0.8, c = cmap(idx),
                    marker=markers[idx], label=cl)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
































































































































