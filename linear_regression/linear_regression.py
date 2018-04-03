#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:40:28 2018

@author: whitestallion
"""

#y = mx + b find m and b.

import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing 
from sklearn import svm, model_selection as cross_validation, neighbors
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import pickle
style.use('ggplot')

#df = quandl.get('WIKI/GOOGL')
#df.to_csv("/Users/whitestallion/Desktop/machine learning basics/google.csv")
#save quandl query for other information with local save
df = pd.DataFrame.from_csv("/Users/whitestallion/Desktop/machine learning basics/google.csv")

df = df[['Adj. Open','Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HC_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HC_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.007 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, 
                                                                     y, 
                                                          test_size=0.2)

clf = LinearRegression(n_jobs = -1)
#Train classifier
clf.fit(X_train, y_train) 
#Serialize
with open('/Users/whitestallion/Desktop/machine learning basics/linear_regression/linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
    
pickle_in = open('/Users/whitestallion/Desktop/machine learning basics/linear_regression/linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)

#forecasting X_lately set
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)
print('Done, days ahead: ' + str(forecast_out))
print('Accuracy: ' + str(accuracy))
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = one_day + last_unix 

#add forecast dates
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


print(df.head())
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

#df.to_csv("/Users/whitestallion/Desktop/machine learning basics/linear_regression/google_forecast.csv")