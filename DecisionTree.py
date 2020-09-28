import pandas as pd
dataset3=pd.read_csv(r'Desktop/Abhi/Purchased.csv')
######EDA
dataset3.info()
dataset3.describe()
###
pd.isnull(dataset3['EstimatedSalary']).sum()
dataset3.dropna(subset)
#############
X=dataset3.iloc[:,[2,3]].values
y=dataset3.iloc[:,4].values
####
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
dataset3.iloc[:,4].value_counts()
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
########
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier=classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
#3
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
#####
#recall or sensitivity or TPR
####TPR=(TP)/(TP+FN)
from sklearn.metrics import accuracy_score,precision_score,recall_score
sample1=classifier.predict(X_train)
sample2=classifier.predict(X_test)

accuracy_score(y_train,sample1)
accuracy_score(y_test,sample2)

precision_score(y_train,sample1)
precision_score(y_test,sample2)

recall_score(y_train,sample1)
recall_score(y_test,sample2)

##############
#######
from sklearn.model_selection import KFold
n=X.shape[0]
####from sklearn.Cross_validation import kFold
#cv=KFold(n=X.shape[0],n_folds=10,shuffle=True,random_state=1)
cvf = KFold(n_splits=10,shuffle=True,random_state=1)
cvf.get_n_splits(n)
import numpy as np
from sklearn.model_selection import cross_val_scoreS
scores = np.mean(cross_val_score(classifier, X, y,cv=cvf,
                         scoring='accuracy',n_jobs=1))


