import os
import pandas as pd
import re
import numpy as np
from globVar import basin3Flag, find_pattern, get_file_name 

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# compute uncertainty coefficient of runoff
def getUncertCoefR(x,index):
    # R 0.023(highest value with lowest uncertainty percent)-0.288(lowest value with highest percent)
    return 0.023 + (0.288 - 0.023) * (x.max() - x[index]) / (x.max() - x.min())

def get_first_single_digit_number(s):
    match = re.search(r'\d{4}', s)
    if match:
        return match.group()
    else:
        return None

def digit_by_position(digit_string, position):
    # Getting the digit at the given position (1-based)
    if position <= len(digit_string):
        return digit_string[position - 1]
    return None

def redistribute_ET4Ref(row):
    tempE = row['P'] - row['R'] - row['S']

    if tempE < 0:
        row['EE'] = row['E']        
    else:
        row['EE'] = tempE

    return row

def sign(x):
  """
  Returns the sign of a number.

  Args:
    x: The number to get the sign of.

  Returns:
    1 if x is positive, -1 if x is negative, and 0 if x is zero.
  """
  if x > 0:
    return 1
  elif x < 0:
    return -1
  else:
    return 0
  
# The function checks the values of a and b and returns different values based on the conditions
def compute_newR1(row, l, col): # P and PR_####_P  
    try:
        a = row[l+'1'] # the reference to be compared with P
    except:
        a = row[l+'2']
    b = row[col]   # the BCC result
    
    # Check if b is null
    if pd.isnull(b) or b == -9999.0:    
        row[col+'_r1'] = np.nan
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

# input file path
path = os.path.join(os.path.dirname(__file__), '', '')
# filePath = path+"extracted_withoutCDR/"
# outPath = path + "extracted_withoutCDR_output/"
# filePath = path+"3BasinsComparison/"
# outPath = path + "3BasinsComparison - output/"
filePath = path+"28BasinsComparison/"
outPath = path + "28BasinsComparison - output/"

test = False
# traverse input files
pattern = "*.csv"
if test:
    pattern = "Amazon.csv"
fileList = find_pattern(pattern, filePath)
# print(fileList)

# water budget components
lab = ['P', 'E', 'R', 'S']
# budget closure correction methods (BCCs) 
met = ['PR', 'CKF', 'MCL', 'MSD']
# traverse each basin files
for fl in fileList:
    fileName = get_file_name(fl)
    print("----------------------------------",fileName)
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
    # add r'' columns and for each extreme component
    for combin in Combinations:
        for l in lab:
            for m in met:
                data[m + '_' + combin + '_' + l + '_r1'] = -9999.0 
                data[m + '_' + combin + '_' + l + '_1'] = data[m + '_' + combin + '_' + l] 
    columns =data.columns
    # print(data.columns.to_list())

    # Compute r1 (outliers)
    for l in lab:        #['P', 'E', 'R', 'S']['P']:#
        for m in met:    #['PR', 'CKF', 'MCL', 'MSD'] ['MSD']:#   
            r = re.compile(m+"_\d{4}_"+l+"$") # PR_####_P
            filtered = list(filter(r.match, columns))

            for col in filtered:#['MSD_5414_P']:#
                # compare each row with merged true value
                data = data.apply(lambda row: compute_newR1(row, l, col), axis=1)  # P and PR_####_P  
    # print(data.columns.to_list())


    if test:
        data.to_csv(outPath+fileName+"_test.csv",index=False)
    else:
        data.to_csv(outPath+fileName+".csv",index=False)


