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
    a = row[l] # the reference to be compared with P
    b = row[col]   # the BCC result
    
    # Check if b is null
    if pd.isnull(b) or b == -9999.0:    
        row[col+'_r1'] = np.nan
    # # Check if a is positive and b is greater than twice of a # Check if a is negative and b is less than twice of a
    # elif (a > 0 and b > 2 * a) | (a < 0 and b < 2 * a):
    #     row[col+'_r1'] = b - 2 * a    
    #     # if it is precipitation, reverse its sign
    #     if col.endswith('P'):
    #         row[col+'_r1'] = -row[col+'_r1']
    #     row[col+'_1'] = row[col] - row[col+'_r1']
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

overAdjustPerc = 0.5
def compute_newR2(row, l, col): # E_P# and PR_####_P_r
    a = row[l] 
    b = row[col]    

    # Check if b is null
    if pd.isnull(b) or b == -9999.0:    
        row[col+'2'] = np.nan
    elif abs(b) > (1+overAdjustPerc)*a:
        row[col+'2'] = b - sign(b)*(1+overAdjustPerc)*a # PR_####_P_r2 = PR_####_P_r - 1.2*Ep
        # if it is precipitation, reverse its sign
        if col[:-2].endswith('P'):
            row[col+'2'] = -row[col+'2']
            row[col[:-2]+'_2'] = row[col[:-2]] + row[col+'2'] # PR_####_P_2 = PR_####_P + PR_####_P_r2
        else:
            row[col[:-2]+'_2'] = row[col[:-2]] - row[col+'2'] # PR_####_E_2 = PR_####_E - PR_####_E_r2

        # if row['date'] == '200910':
        # print(row['date'],b,sign(b),a,b - sign(b)*(1+overAdjustPerc)*a,row[col+'2'],row[col[:-2]+'_2'])
    else:
        row[col+'2'] = 0

    return row

# the function redistributes PR_####_P_r1 to PR_####_P, and get new PR_####_P_1
# as long as all values of r1 is not null, it will be redistributed according to weights columns (PR_####_P_w)
# update: include both r1 and r2
def redistributeR(row,filtered, index):
    # get all columns with r1： PR_####_P_r1
    r1_columns = [col for col in filtered if col.endswith('_r'+index)]

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
            if row[col] == 0 and col[-4:-3]!='R': # exclude Runoff
                cols_to_update.append(col[:-3])
                weights.append(row[col[:-3] + '_w'])
        # redistribute to cols_to_update
        sumWeights = sum(weights)
        if sumWeights != 0:
            for col in cols_to_update:
                if col.endswith('P'):
                    row[col+'_'+index] = row[col]-sum_r1 * row[col + '_w'] / sumWeights
                else:
                    row[col+'_'+index] = row[col]+sum_r1 * row[col + '_w'] / sumWeights

    return row

# input file path
path = os.path.join(os.path.dirname(__file__), '', '')
if basin3Flag:
    filePath = path+"3BasinsComparison_mergeClosed_partTrue/"
    outPath = path + "3redistribution_outliers_mergeClosed_partTrue/"
else:
    filePath = path+"28BasinsComparison_mergeClosed_partTrue/"
    outPath = path + "28redistribution_outliers_mergeClosed_partTrue/"

test = False
# traverse input files
pattern = "*.csv"
if test:
    # pattern = "1147010_bccTest.csv"
    pattern = "4127800_bcc.csv"
fileList = find_pattern(pattern, filePath)
# print(fileList)

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

    ####################################
    # recompute merged components P/E/R/S to make it close  
    # ##################################
    data['RR'] = data['R'] 
    data['PP'] = data['P']
    data['SS'] = data['S']
    data = data.apply(redistribute_ET4Ref, axis=1)

    ####################################
    # recompute columns E_P/E/R/S# with P/E/R/S  
    # ##################################
    # For P: true value merged values
    p_columns = [col for col in data.columns if re.match(r'^P\d+$', col) ]
    for col in p_columns:
        data[f'E_{col}'] = abs(data[col] - data['PP'])
    # data['E_P1'] = abs(data['P1']-data['PP'])
    # data['E_P2'] = abs(data['P2']-data['PP'])
    # data['E_P3'] = abs(data['P3']-data['PP'])
    # data['E_P4'] = abs(data['P4']-data['PP'])
    # data['E_P5'] = abs(data['P5']-data['PP'])
    # For E: E#*20%
    e_columns = [col for col in data.columns if re.match(r'^E\d+$', col) ]
    for col in e_columns:
        data[f'E_{col}'] = abs(data[col] - data['EE'])
    # data['E_E1'] = abs(data['E1']-data['EE'])
    # data['E_E2'] = abs(data['E1']-data['EE'])
    # data['E_E3'] = abs(data['E1']-data['EE'])
    # data['E_E4'] = abs(data['E1']-data['EE'])
    # For R: R*7%
    data['E_R1'] = data['R1']*0.07
    # For S: true value TWSC_GRACE_Mascon_JPL_calculate
    s_columns = [col for col in data.columns if re.match(r'^S\d+$', col) ]
    for col in s_columns:
        data[f'E_{col}'] = abs(data[col] - data['SS'])
    # data['E_S1'] = abs(data['S1']-data['SS'])
    # data['E_S2'] = abs(data['S2']-data['SS'])
    # data['E_S3'] = abs(data['S3']-data['SS'])
    # data['E_S4'] = abs(data['S4']-data['SS'])

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
                data[m + '_' + combin + '_' + l + '_r2'] = -9999.0 
                data[m + '_' + combin + '_' + l + '_1'] = data[m + '_' + combin + '_' + l] 
                data[m + '_' + combin + '_' + l + '_2'] = data[m + '_' + combin + '_' + l] 
    columns =data.columns
    # print(data.columns.to_list())

    # Compute r1 (outliers)
    # Compute r2 (extremes)
    for l in lab:        #['P', 'E', 'R', 'S']['P']:#
        for m in met:    #['PR', 'CKF', 'MCL', 'MSD'] ['MSD']:#   
            r = re.compile(m+"_\d{4}_"+l+"$") # PR_####_P
            filtered = list(filter(r.match, columns))

            for col in filtered:#['MSD_5414_P']:#
                # compare each row with merged true value
                data = data.apply(lambda row: compute_newR1(row, l+l, col), axis=1)  # P and PR_####_P  

                # Extract the 4-digit number from the column
                col_number = get_first_single_digit_number(col)
                if col_number:
                    # Determine the digit based on the letter
                    if l == 'P':
                        digit = digit_by_position(col_number, 1)
                    elif l == 'E':
                        digit = digit_by_position(col_number, 2)
                    elif l == 'R':
                        digit = digit_by_position(col_number, 3)
                    elif l == 'S':
                        digit = digit_by_position(col_number, 4)
                    else:
                        digit = None

                    if digit is not None:
                        col_e = f'E_{l}{digit}'
                        data = data.apply(lambda row: compute_newR2(row, col_e, col + '_r'), axis=1)  # E_P#, and PR_####_P_r
    # print(data.columns.to_list())

    # redistribute r1 for each BCC and composition
    # for combin in ['5414']:#Combinations: 
    #     for m in ['MSD']:#met: # PR_####
    for combin in Combinations: 
        for m in met: # PR_####
            r = re.compile(m+"_"+combin+"_[PERS]") # PR_1111_P/E/R/S + null/_w/_r/_r1/r2/_1 (24 columns in total)
            filtered = list(filter(r.match, columns))
            # print(filtered)

            # redistribute r1
            data = data.apply(lambda row: redistributeR(row,filtered,'1'), axis=1)
            # redistribute r2
            data = data.apply(lambda row: redistributeR(row,filtered,'2'), axis=1)

    if test:
        data.to_csv(outPath+fn+"_test3.csv",index=False)
    else:
        data.to_csv(outPath+fn+".csv",index=False)


