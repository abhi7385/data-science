import pandas as pd
#load file 
dataset=pd.read_csv(r"E:\Abhi\Salary.csv")
#EDA
dataset.head()
dataset.describe()
dataset.info()
dataset.shape
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(dataset.Salary.dropna(),kde=False,bins=39)
plt.plot(dataset)
plt.scatter(dataset['YearsExperience'],dataset['Salary'])
plt.plot(dataset['YearsExperience'],dataset['Salary'],'-p',
         color='grey',markersize=15,linewidth=4,
         markerfacecolor='white',markeredgecolor='grey',
         markeredgewidth=2)
#####values get dataset in array
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1:].values
###########
####data spliting into training And test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=1/3)
#model selection
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
#train model
regressor.fit(X_train,y_train)
####predicting test set result
y_pred=regressor.predict(X_test)

regressor.score(X_train,y_train)
# cross validation
regressor.score(X_test,y_pred)

regressor.coef_
regressor.intercept_

from sklearn.metrics import mean_squared_error,r2_score
mean_squared_error(y_test,y_pred)
r2_score(y_test,y_pred)
## to see r like data 
import statsmodels.api as sm

from statsmodels.sandbox.regression.predstd import wls_prediction_std
#######
regressorST=sm.OLS(y_train,X_train)
ols_output=regressorST.fit()
ols_output.summary()
#######
plt.matshow(dataset.corr())
########### plot plsn correlation
fig=plt.subplots(figsize=(10,10))
sns.get(font_scale=1.5)
sns.heatmap(dataset.corr(),square=True,cbar=True,annot_kws={'size':10})
plt.show()
sns.get_dataset_names()
######################
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train))
plt.title('Salary vs Exp')
plt.xlabel('year of Exp')
plt.ylabel('Salary')
plt.show()















