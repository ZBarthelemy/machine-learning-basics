#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:08:34 2018

"""

#Classification of space with Euclidean distance
#Cons: computational heavy for large N does not scale well
#      no good way to train the set
#Pros: can be threaded for clustering measures

import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing 
from sklearn import svm, model_selection as cross_validation, neighbors 

accuracies = []
for count in range(25):
    df = pd.read_csv('/Users/whitestallion/Desktop/machine learning basics/k_nearNeighbors/breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    
    X = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
                                                                         y,
                                                                         test_size=0.3)
    clf = neighbors.KNeighborsClassifier(n_jobs=1)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print ('run ' + str(count) + ' accuracy :' + str(accuracy))
    accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies))
    #print('accuracy: ' + str(accuracy))
    #example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
    #example_measures = example_measures.reshape(len(example_measures),-1)
    #prediction = clf.predict(example_measures)
    #print('example measure: ' + str(example_measures))
    #print('prediction: ' + str(prediction))
