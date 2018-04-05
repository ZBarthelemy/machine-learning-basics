#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:08:34 2018

@author: whitestallion
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

df =  pd.read_csv('breast-cancer-wisconsin.data.txt')