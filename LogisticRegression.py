import pandas as pd
dataln=pd.read_csv(r'Desktop/Abhi/Purchased.csv')
####
#Multivariable analysis 
dataln.info()
dataln.isna()
dataln.isna().sum()
dataln.dropna(subset=['Age'],inplace=True,axis=0)
dataln.describe(include='all')
dataln.cov()
######practice for insert NaN value and after of that delet using row and 
####coloumn
##dataln=dataln.values convert  dataframe into array
##from numpy import nan ###if want insert nan value at coloumt have to 
##import nan
###dataln.iloc[2,2]=nan
###dataln.dropna(subset=['Age'],inplace=True,axis=0)###is don't exicuted because
###array does not have attrribute dropna for this you have to convert
##array into data frame
##dataln=pd.DataFrame(dataln)
##dataln.dropna(axis=0)

dataln.corr()
#output 
#                  User ID       Age  EstimatedSalary  Purchased
#User ID          1.000000 -0.000721         0.071097   0.007120
#Age             -0.000721  1.000000         0.155238   0.622454
#EstimatedSalary  0.071097  0.155238         1.000000   0.362083
#Purchased        0.007120  0.622454         0.362083   1.000000
#by corr table see that id and age having less value mean when take age
#variable does not take id which does not contribute in output
dataln.skew()##it give center tendency of data
dataln.kurt()
dtype(dataln)
sns.pairplot(dataln,hue='Purchased',size=3)
##########
#######univariable analysis under EDA
data=[dataln['Gender']]
plt.boxplot(data)
####
type(dataln['Age'])
dataln['Age'].mean()
dataln['Purchased'].value_counts()#how much no of time give purchased value
dataln['Gender'].value_counts()
dataln['Age'].quantile()
####Bivariant analysis
x=dataln['EstimatedSalary']
y=dataln['Purchased']
plt.figure(figsize=(7,9))
plt.scatter(x,y)
sns.boxplot(x='Purchased',y='Age',data=dataln)
ab=dataln[:,]
#######multivariable analysis
import matplotlib.pyplot as plt
xx=dataln.corr()
plt.figure(figsize=(8,9))
plt.imshow(dataln.corr(),cmap='hot')
plt.colorbar()
plt.xticks(range(len(xx)),xx.columns,rotation=20)
plt.yticks(range(len(xx)),xx.columns)
plt.show()
###########
import seaborn as sns
f,ax=plt.subplots(figsize=(10,8))
corr=dataln.corr()
sns.heatmap(corr,xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
#####it will be give tendancy of two columns
sns.FacetGrid(dataln,hue='Purchased',size=5).map(sns.distplot,'Age').add_legend()
sns.FacetGrid(dataln,hue='Purchased',size=5).map(sns.distplot,'EstimatedSalary').add_legend()
sns.boxplot(y='Age',data=dataln)
#######
import numpy as np
x=dataln.iloc[:,[2,3]].values
#T
y=dataln.iloc[:,4].values
#########
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.25)
#####################
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
##########
from sklearn.linear_model import LogisticRegression
Lregression=LogisticRegression(random_state=0)
Lregression.fit(X_train,y_train)
y_pred=Lregression.predict(X_test)
#########
X_new=np.array([dataln['Age'],dataln['EstimatedSalary']])
y_pred_new=Lregression.predict(X_new)

#######
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
###########
s_probs=Lregression.predict_proba(X_test)
lr_prob=s_probs[:,1]

##########
##Roc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score 
######
ns_prob=[0 for _ in range(len(y_test))]
ns_auc=roc_auc_score(y_test,ns_prob)
lr_auc=roc_auc_score(y_test,lr_prob)

print('No prob without auc=%.3f'%(ns_auc))
print('No prob without auc=%.3f'%(lr_auc))

ns_fpr,ns_tpr,_=roc_curve(y_test,ns_prob)
lr_fpr,lr_tpr,_=roc_curve(y_test,lr_prob)


plt.plot(ns_fpr,ns_tpr,linestyle='_ _',label='Prob')
plt.plot(lr_fpr,lr_tpr,marker='.',label='logistic')
plt.xlabel('FPR/(1-Specificity)/(1-TN/TN+FP)')
plt.ylabel('TPR/Recall/Sensitivity/(TP/TP+FN)')
plt.legend()
plt.show
###########
#####f1 score
###F1=2*(presision*recall)/(precision+recall)

from sklearn.metrics import f1_score
F1=f1_score(y_test,y_pred)
####k-fold cross_validation with logistice regeration
from sklearn.model_selection import cross_val_score 
Lreg=LogisticRegression()
print(cross_val_score(Lreg,x,y,cv=10,
                      scoring='accuracy').mean())

######
#cross validation on variable selection
dataset.head()
feature_cols=['User ID','Age','EstimatedSalary']
X1=dataln[feature_cols]
y1=dataln.Purchased
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
score=cross_val_score(lm,X1,y1,cv=10,scoring='neg_mean_squared_error')
print(score)
mse_score=-score
import numpy as np
rmse=np.sqrt(mse_score)
print(rmse.mean())
############################
feature_cols1=['Age','EstimatedSalary']
X1=dataln[feature_cols1]
y1=dataln.Purchased
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
score=cross_val_score(lm,X1,y1,cv=10,scoring='neg_mean_squared_error')
print(score)
mse_score=-score
import numpy as np
rmse=np.sqrt(mse_score)
print(rmse.mean())
#######by this we understand that which variable is contribute or not
import seaborn as sns
Lregression.score(X_test,y_test)
from sklearn import metrics
cm1=metrics.confusion_matrix(y_test,y_pred)
print(cm1)
plt.figure(figsize=(9,9))
sns.heatmap(cm1,annot=True,fmt='.3f',linewidths=.5)
plt.ylabel('Actual');
plt.xlabel('Pred');
all_sample_title='accuracy score:{0}',formate(score)
plt.show()
##########
##################################














