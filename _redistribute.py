import os, fnmatch
import pandas as pd
import re
import numpy as np
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

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

# compute uncertainty coefficient of runoff
def getUncertCoefR(x,index):
    # R 0.023(highest value with lowest uncertainty percent)-0.288(lowest value with highest percent)
    return 0.023 + (0.288 - 0.023) * (x.max() - x[index]) / (x.max() - x.min())

# The function checks the values of a and b and returns different values based on the conditions
def compute_newR(row, l, col): # P and PR_####_P
    a = row[l] 
    b = row[col]   
    
    # Check if b is null
    if pd.isnull(b) or b == -9999.0:    
        row[col+'_r1'] = np.nan
    # Check if a is positive and b is greater than twice of a # Check if a is negative and b is less than twice of a
    elif (a > 0 and b > 2 * a) | (a < 0 and b < 2 * a):
        row[col+'_r1'] = b - 2 * a    
        # if it is precipitation, reverse its sign
        if col.endswith('P'):
            row[col+'_r1'] = -row[col+'_r1']
        row[col+'_1'] = row[col] - row[col+'_r1']
    # Check if a is negative and b is positive
    elif (a < 0 and b > 0) | ( a > 0 and b < 0):
        row[col+'_r1'] = b
        # if it is precipitation, reverse its sign
        if col.endswith('P'):
            row[col+'_r1'] = -row[col+'_r1']
        row[col+'_1'] = 0
    # If none of the above conditions are met
    else:
        row[col+'_r1'] = 0
    
    return row

# the function redistributes PR_####_P_r1 to PR_####_P, and get new PR_####_P_1
# as long as all values of r1 is not null, it will be redistributed according to weights columns (PR_####_P_w)
def redistributeR1(row,filtered):
    # get all columns with r1
    r1_columns = [col for col in filtered if col.endswith('_r1')]

    # check if all r1 values are 0
    all_r1_null = all(np.isnan(row[col]) for col in r1_columns)

    # if all r1 values are null, do nothing
    if all_r1_null:
        return row
    # as long as one r1 is not null, update columns (e.g. PR_####_P_1) where r1 is null (e.g. PR_####_P_r1) according to weights columns (PR_####_P_w)
    else:
        sum_r1 = sum(row[col] for col in r1_columns)
        cols_to_update = []
        weights = []

        # check which r1 column is 0 then distribute sum_r1 to them
        # before that, the weights of these columns should be summed up and recomputed
        for col in r1_columns:
            if row[col] == 0:
                cols_to_update.append(col[:-3])
                weights.append(row[col[:-3] + '_w'])
        # redistribute to cols_to_update
        sumWeights = sum(weights)
        if sumWeights != 0:
            for col in cols_to_update:
                if col.endswith('P'):
                    row[col+'_1'] = row[col]-sum_r1 * row[col + '_w'] / sumWeights
                else:
                    row[col+'_1'] = row[col]+sum_r1 * row[col + '_w'] / sumWeights

    return row

# input file path
path = os.path.join(os.path.dirname(__file__), '', '')
filePath = path+"3BasinsComparison/"
print(filePath)
outPath = path + "redistribution_outliers/"
overAdjustPerc = 0.2
test = False

# traverse input files
pattern = "*.csv"
if test:
    # pattern = "1147010_bccTest.csv"
    pattern = "4127800_bcc.csv"
fileList = find_pattern(pattern, filePath)
print(fileList)

# water budget components
lab = ['P', 'E', 'R', 'S']
# budget closure correction methods (BCCs) 
met = ['PR', 'CKF', 'MCL', 'MSD']
# traverse each basin files
for fl in fileList:
    fileName = get_file_name(fl)
    fn = fileName[:-4]
    print("----------------------------------",fn)
    data = pd.read_csv(fl)#.head(6)
    columns =data.columns
    # print(data)

    # match combinations and prelocate new columns
    exhaustCompnents = [] # for P,E,S [['1', '2', '3', '4', '5'], ['1', '2', '3', '4'], ['1', '2', '3', '4']]
    for m in ['P','E','S']: 
        r = re.compile(m+"\d$")
        _colFiltered = list(filter(r.match, columns))
        # print(colFiltered)
        exhaustCompnents.append([s[1:] for s in _colFiltered])  # For P: ['1', '2', '3', '4', '5']
    # print('exhaustCompnents',exhaustCompnents)
    Combinations = [a+b+'1'+c for a in exhaustCompnents[0] for b in exhaustCompnents[1] for c in exhaustCompnents[2]]
    # add r' columns and for each outlier component
    for combin in Combinations:
        for l in lab:
            for m in met:
                data[m + '_' + combin + '_' + l + '_r1'] = -9999.0 
                data[m + '_' + combin + '_' + l + '_1'] = data[m + '_' + combin + '_' + l] 
    columns =data.columns
    # print(data.columns)

    # Compute r1 (outliers)
    for l in lab:        #['P', 'E', 'R', 'S']
        for m in met:    #['PR', 'CKF', 'MCL', 'MSD']    
            r = re.compile(m+"_\d{4}_"+l+"$") # PR_####_P
            filtered = list(filter(r.match, columns))

            for col in filtered:
                # compare each row with merged true value
                data = data.apply(lambda row: compute_newR(row, l, col), axis=1) 
            
    # redistribute r1 for each BCC and composition
    # for combin in ['5414']:#Combinations: 
    #     for m in ['MSD']:#met: # PR_####
    for combin in Combinations: 
        for m in met: # PR_####
            r = re.compile(m+"_"+combin+"_[PERS]") # PR_1111_P/E/R/S + null/_w/_r/_r1/_1 (20 columns in total)
            filtered = list(filter(r.match, columns))
            # print(filtered)

            data = data.apply(lambda row: redistributeR1(row,filtered), axis=1)

    if test:
        data.to_csv(outPath+fn+"_test.csv",index=False)
    else:
        data.to_csv(outPath+fn+".csv",index=False)


