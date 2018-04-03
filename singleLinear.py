#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 14:47:47 2018

@author: whitestallion
"""

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
import random

style.use('fivethirtyeight')
#y = mx + b
# m = [avg(x)*avg(y) - avg(x*y)] / [avg(x)^2-avg(x^2)]
# b = [avg(y) - m * avg(x)]
# r_sq = 1 - SE(y_estimate)/SE(mean(y))

#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_dataset(n, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(n):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope_and_intercept(xs, ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs*ys))/
         (mean(xs) * mean(xs) - mean(xs*xs)) )
    b = mean(ys) - m * mean(xs)
    return m, b

def squared_error(ys_original, ys_line):
    return sum((ys_line - ys_original)**2)
    
def coefficient_of_determination(ys_original, ys_line):
    y_mean_line = [mean(ys_original) for y in ys_original]
    squared_error_regr = squared_error(ys_original, ys_line)
    squared_error_y_mean = squared_error(ys_original, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

xs, ys = create_dataset(40, 10, 2, correlation ='pos')
m,b = best_fit_slope_and_intercept(xs, ys)
print ('m: ' +str(m) , 'b: ' + str(b))

regression_line = [(m*x)+b for x in xs]
predict_x = 8
predict_y = (m*predict_x)+b

r_squared = coefficient_of_determination(ys, regression_line)
print('R^2: ' + str(r_squared))

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color = 'g')
plt.plot(xs, regression_line)
plt.show()

#m = best_fit_slope(xs, ys)