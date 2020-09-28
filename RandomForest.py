import pandas as pd 
dataset=pd.read_excel(r'Desktop/Abhi/Jobs.xlsx')
#######
dataset.describe()
dataset.info()
#####
dataset.cov()
dataset.corr()
dataset.skew()
dataset.kurt()

##with target check independant variable
####after that check induval
######

X=dataset.iloc[:,:-1].values
#T
y=dataset.iloc[:,6].values
####by using labelencoding convert catogorical into  numerical
from  sklearn.preprocessing import LabelEncoder,OneHotEncoder 
labelencoder=LabelEncoder()
X[:,1]=labelencoder.fit_transform(X[:,1])
X[:,3]=labelencoder.fit_transform(X[:,3])
X[:,4]=labelencoder.fit_transform(X[:,4])
X[:,5]=labelencoder.fit_transform(X[:,5])
y=labelencoder.fit_transform(y)
###########
oneH=OneHotEncoder(categorical_features=[1,3,4,5])
oneH.fit_transform(X).toarray()
#################################same catogorical value tranforme into
###numeriacal by different method
d={'Y':1,'N':0}
dataset['Hired']=dataset['Hired'].map(d)
dataset['Employed?']=dataset['Employed?'].map(d)
dataset['Interned']=dataset['Interned'].map(d)
dataset['Top-tier school']=dataset['Top-tier school'].map(d)
d1={'BS':0,'MS':1,'PhD':2}
dataset['Level of Education']=dataset['Level of Education'].map(d1)

feacture=list(dataset.columns[:6])
X=dataset[features]
y=dataset['Hired']
#######

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
########
from sklearn.ensemble import RandomForestClassifier
ClassiRF=RandomForestClassifier(n_estimators=10,
                               criterion='entropy',
                               random_state=0)
ClassiRF.fit(X_train,y_train)
y_pred=ClassiRF.predict(X_test)
#####
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)


