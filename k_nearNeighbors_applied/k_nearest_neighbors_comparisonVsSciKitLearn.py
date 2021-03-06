#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:58:44 2018

@author: whitestallion
"""
from math import sqrt
import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random


def k_nearest_neighbors(data, predict, k = 3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total groups!')
    distances = []
    for group in data:
        for features in data[group]:
            #euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    #print( Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    #print(vote_result, confidence)
    return vote_result, confidence

accuracies = []
for count in range(25):
    df = pd.read_csv("/Users/whitestallion/Desktop/machine learning basics/k_nearNeighbors/breast-cancer-wisconsin.data.txt")
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()
    
    #print(full_data[:5])
    random.shuffle(full_data)
    #print(5*'#')
    #print(full_data[:5])
    
    test_size = 0.3
    train_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size*len(full_data))]
    
    test_set = {2:[], 4:[]}
    test_data = full_data[-int(test_size*len(full_data)):]
    
    for i in train_data:
        # get classes
        train_set[i[-1]].append(i[:-1])
    for i in test_data:
        # get classes
        test_set[i[-1]].append(i[:-1])
        
    correct = 0
    total = 0
    
    #training
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k = 5)
            if group == vote:
                correct += 1
            #else:
                #print('Confidence score of wrong guess :' + str(confidence))
            total += 1
    accuracies.append(correct/total)
    print('Accuracy for run ' + str(count) ,correct/total)
print(sum(accuracies)/len(accuracies))