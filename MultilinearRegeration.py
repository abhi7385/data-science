import pandas as pd
#load file
import os

 
os.chdir('E:\Abhi\multiple-linear-regeration')
dataset2=pd.read_csv('multiple-linear-regeration\Startups.csv')
dataset2=pd.read_csv('Desktop/Abhi/50_Startups.csv')
#iv
x=dataset2.iloc[:,:-1].values
#T
y=dataset2.iloc[:,4].values

###################################
#Encoding(convert catogorical data in number)
##
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncode=LabelEncoder()
x[:,3]=labelEncode.fit_transform(x[:,3])
####### to equel the variane the used onehotencoder(balance the weightage)
oneH=OneHotEncoder(categorical_features=[3])
import numpy as np
x=oneH.fit_transform(x).toarray()
###
#########split the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=1)
#######model train
from sklearn.linear_model import LinearRegression
regressorml=LinearRegression()
#
regressorml.fit(X_train,y_train)
y_pred=regressorml.predict(X_test)
####
from sklearn.metrics import mean_squared_error,r2_score
mean_squared_error(y_test,y_pred)
r2_score(y_test,y_pred)
##
import matplotlib.pyplot as plt
import seaborn as sns
############
import pandas as pd
csv_data='''A,B,C,D
1.0,2.0,3.0,4.0
6.0,7.0,8.0
0.0,11.0,12.0'''

df=pd.read_csv(StringIO(csv_data))
#####

from io import StringIO
df=pd.read_csv(StringIO(csv_data))
df.isnull() ######it give the position where value is null or present in for of 
#####true and false
df.isnull().sum() #######how much null value present in coloumn
tmp2=df.dropna(how='all')
tmp3=df.dropna(axis=0)
tmp4=df.dropna(axis=1)
###########
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy='mean',axis=1)
imputer=imputer.fit(df)
imputer=imputer.transform(df.values)
######if fit an transform used at same time
imputer=imputer.fit_transform(df.values)

##########
import numpy as npS
df=pd.DataFrame([['green','A',10.1,'c1'],
                ['red','AA',13.5,'c2'],
                ['blue','AAA',13.4,'c1']])

df.columns=['color','Q','price','class']
q_mapping={'A':1,'AA':2,'AAA':3}
df['Q']=df['Q'].map(q_mapping)
np.unique(df['class'])

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
x=df[['color','Q','price']].values
color_label=LabelEncoder()
x[:,0]=color_label.fit_transform(x[:,0])
###########
oneH=OneHotEncoder(categorical_features=[0])
oneH.fit_transform(x).toarray()
###############













