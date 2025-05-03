import os, fnmatch
import re
import numpy as np
import pandas as pd
from itertools import combinations
from math import comb

basin3Flag= False
strategy = "mergeAsRef"

# match a pattern:
# find('*.txt', '/path/to/dir')
def find_pattern(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def get_file_name(path_string):
    pattern = re.compile(r'([^<>/\\\|:""\*\?]+)\.\w+$')
    data = pattern.findall(path_string)
    if data:
        return data[0]
    
# Function to compute statistics
def compute_stats(true_values, corrected_values):
    if corrected_values.isnull().all():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    data = pd.concat([true_values, corrected_values], axis=1).dropna()
    true_values = data.iloc[:, 0]
    corrected_values = data.iloc[:, 1]

    pbias = 100 * ((true_values - corrected_values).sum()) / true_values.sum()
    cc = true_values.corr(corrected_values)
    rmse = ((true_values - corrected_values) ** 2).mean() ** .5
    me = (true_values - corrected_values).mean()
    me1 = (true_values - corrected_values.mean()).abs().mean()
    mae = (true_values - corrected_values).abs().mean()
    mape = 100 * ((true_values - corrected_values) / true_values).abs().mean()
    return pbias, cc, rmse, me, me1, mae, mape

# # compute merged true values
# def getMergedTrue(arr):
#     m = arr.mean()
#     sig = arr - m
#     t = np.power(sig, -2, dtype=float)
#     s = t.sum()
#     w = t / s
#     return (arr * w).sum()
# compute merged true values
def getMergedTrue(arr):
    m = arr.mean()
    return m

# generate coefficient matrix for MCL
# 1/(2*(n-1)C2) * theta_jk
def generateAInverse(paramNum):
    denominator = 2*comb(paramNum-1,2)
    
    myrows = np.arange(1,paramNum+1)
    A = []    
    # iterate N rows
    for N in myrows:           
        B = []
        # iterate combinations for columns
        combin = combinations(myrows,2)
        for c in combin:
            j = c[0]
            k = c[1]

            if N == j:
                B += [paramNum-2]
            elif N == k:
                B += [paramNum-2]
            elif (N != j) & (N != k):
                B += [-1]
        
        A += [[x / denominator for x in B]]

    return A

# compute squared distance between two series
# sumn(xi-xj)^2*1/n (eq 2 in Pan. RSE 2015)
def getSquaredDistance(x,y):
    # # Compute the square of the difference for each pair of corresponding elements in x and y
    # diff_squared = np.square(x - y)
    
    # # Sum the squares and divide by the number of elements
    # return np.sum(diff_squared) / len(x)

    # here we are computing every thing
    # step by step
    p1 = np.sum([(a * a) for a in x])
    p2 = np.sum([(b * b) for b in y])
    
    # using zip() function to create an
    # iterator which aggregates elements 
    # from two or more iterables
    p3 = -1 * np.sum([(2 * a*b) for (a, b) in zip(x, y)])
    size = x.size

    return np.sum(p1 + p2 + p3)/size

# compute uncertainty of components
def getUncert(true,a):
    return abs(true -  a) / true

# compute uncertainty coefficient of runoff
def getUncertCoefR(x,index):
    # R 0.023(highest value with lowest uncertainty percent)-0.288(lowest value with highest percent)
    return 0.023 + (0.288 - 0.023) * (x.max() - x[index]) / (x.max() - x.min())
