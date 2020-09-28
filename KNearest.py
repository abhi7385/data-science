import pandas as pd 
dataset=pd.read_csv(r'Desktop/Abhi/Purchased.csv')
##########
#EDA,preposesing
###Bivariable,univariable,multivariable,analysis
##center tendency,skew,kurt
#
X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)

dataset['Purchased'].value_counts()
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifierK3=KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=2)
classifierk5=KNeighborsClassifier(n_neighbors=5,metric='euclidean',p=2)
classifierK3.fit(X_train,y_train)
classifierk5.fit(X_train,y_train)

y_predk3=classifierK3.predict(X_test)
y_predk5=classifierk5.predict(X_test)

classifierK3.fit(X_train,y_train).score(X_train,y_train)
classifierk5.fit(X_train,y_train).score(X_train,y_train)
import numpy as np
X_testNew=np.array([[19,7000]])

y_predk3_X_testNew=classifierK3.predict(X_testNew)
y_predk5_X_testNew=classifierk5.predict(X_testNew)


import seaborn as sns

sns.lmplot('Age','EstimatedSalary',data=dataset,
           fit_reg=False,hue='Purchased',
           scatter_kws={'s':100}) 

#### search possible value of k 
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import GridSearchCV
#flow Creation
X_std=sc.fit_transform(X)
pipe=Pipeline([('sc',sc),
               ('knn',
                classifierk5.fit(X_train,y_train))])
######
search_space=[{'knn__n_neighbors':[1,2,3,4,5,6,7,8,9,10]}]
gscf=GridSearchCV(pipe,search_space,
                  cv=5).fit(X_std,y)
gscf.best_estimator_.get_params()['knn__n_neighbors']

####best K plot
















