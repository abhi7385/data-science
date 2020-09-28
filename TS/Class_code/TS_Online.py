# -*- coding: utf-8 -*-


from random import gauss,seed

import matplotlib.pyplot as plt
#from pandas.plotting import 

seed(1)

series = [gauss(0.0 , 1.0) for i in range(1000)]

#from pandas import Series

import pandas as pd

series = pd.Series([gauss(0.0 , 1.0) for i in range(1000)])

plt.plot(series)
plt.title("White Noise")
plt.show()


plt.hist(series)
plt.title("White Noise")
plt.show()

print(series.describe())
###########################################


import numpy as np

steps = np.random.normal(loc=0, scale=1, size=5000)

steps[0] = 0

P = 100 + np.cumsum(steps)

plt.plot(P)
plt.title("Random Walk")
plt.show()
####################################

import numpy as np

steps1 = np.random.normal(loc=0, scale=1, size=5000)

steps1[0] = 0

P1 = 100 * np.cumsum(steps1)

plt.plot(P1)
plt.title("Random Walk")
plt.show()

################################################

# Augmented Dickey-Fuller Test = ADF test

from statsmodels.tsa.stattools import adfuller


AMZN = pd.read_csv(r'G:\Traning\AlgoRithm\TimeSerise/AMZN.csv')

plt.plot(AMZN['Adj Close'])
plt.title("data plot to see seasonalty/tr/cy")
plt.show()


result = adfuller(AMZN['Adj Close'])

print(result)

print("p value" + str(result[1]))

#######################################################
#
AMZN_net = AMZN['Adj Close'].pct_change()

AMZN_net = AMZN_net.dropna()

result_change = adfuller(AMZN_net)
print("p-value" + str(result_change[1]))
plt.plot(AMZN_net)
plt.title("random walk with Drift")
plt.show()

######################################
#ACF
#PACF

#X = 3,5,6,6,7,4,5,6,7,2,3,4.....

#y1,y2.y3y4......yn


 Sea = AMZN_net.diff(4)
 
Sea.head(10)

Sea = Sea.dropna()


plt.plot(Sea)
plt.title("random walk with Drift")
plt.show()

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(Sea)
plt.show()

#######################################

from statsmodels.tsa.arima_process import ArmaProcess

# AR parameter = +0.9
plt.subplot(2,1,1)

ar1 = np.array([1, -0.9])

ma1 = np.array([1])

AR_obj1 = ArmaProcess(ar1,ma1)

simulated_data_1 = AR_obj1.generate_sample(nsample=1000)

plt.plot(simulated_data_1)


plt.subplot(2,1,2)

ar2 = np.array([1, 0.9])

ma2 = np.array([1])

AR_obj2 = ArmaProcess(ar2,ma2)

simulated_data_2 = AR_obj2.generate_sample(nsample=1000)

plt.plot(simulated_data_2)




































































