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

def redistribute_ET(row):
    tempE = row['P_closed'] - row['R_closed'] - row['S_closed']

    if tempE > 0:
        row['E_closed'] = tempE

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

# the function redistributes PR_####_P_r1 to PR_####_P, and get new PR_####_P_1
# as long as all values of r1 is not null, it will be redistributed according to weights columns (PR_####_P_w)
def redistributeR(row,filtered,index):
    # get all columns with r1
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
            if row[col] == 0 and col[-4:-3]!='R': #: exclude Runoff
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

# whether only compute outliers
# Change this flags to 制作原_bcc文件的副本到同一文件夹下(三种方法)
introduceObs = False
redistribute = True
# input file path
path = os.path.join(os.path.dirname(__file__), '', '')
if basin3Flag:
    xlsx_file = path+"3BasinsComparison/stationsPrecipitation.xlsx"
    if introduceObs and redistribute:
        filePath = path+"3BasinsComparison_obsIntroduced/"
        outPath = path + "3redistribution_obsIn_outliersRedistributed/"
    elif introduceObs and not redistribute:
        filePath = path+"3BasinsComparison_obsIntroduced/"
        outPath = path + "3BasinsComparison_obsIntroduced1/"
    else:
        filePath = path+"3BasinsComparison/"
        outPath = path + "3BasinsComparison1/"

    # Read Excel file
    excel_data = pd.read_excel(xlsx_file, dtype=float)
    excel_data = excel_data.rename(columns={2181900: str(2181900),4127800: str(4127800),6742900: str(6742900)})
else:
    xlsx_file = path+"28BasinsComparison/stationsPrecipitation.xlsx"
    if introduceObs and redistribute:
        filePath = path+"28BasinsComparison_obsIntroduced/"
        outPath = path + "28redistribution_obsIn_outliersRedistributed/"
    elif introduceObs and not redistribute:
        filePath = path+"28BasinsComparison_obsIntroduced/"
        outPath = path + "28BasinsComparison_obsIntroduced1/"
    else:
        filePath = path+"28BasinsComparison/"
        outPath = path + "28BasinsComparison1/"

    # Read Excel file
    excel_data = pd.read_excel(xlsx_file, dtype=float)
    excel_data = excel_data.rename(columns={1159100: str(1159100), 1234150: str(1234150), 2180800: str(2180800), 2181900: str(2181900), 2909150: str(2909150), 2912600: str(2912600), 3265601: str(3265601), 3629001: str(3629001), 4103200: str(4103200), 4115201: str(4115201), 4127800: str(4127800), 4146281: str(4146281), 4146360: str(4146360), 4147703: str(4147703), 4150450: str(4150450), 4150500: str(4150500), 4152050: str(4152050), 4207900: str(4207900), 4208025: str(4208025), 4213711: str(4213711), 4214270: str(4214270), 4243151: str(4243151), 5404270: str(5404270), 6226800: str(6226800), 6340110: str(6340110), 6435060: str(6435060), 6457010: str(6457010), 6590700: str(6590700)})

test = False
# traverse input files
pattern = "*.csv"
if test:
    # pattern = "1147010_bccTest.csv"
    pattern = "4127800_bcc.csv"
    # pattern = "data_bcc.csv"
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
    # compute true values
    # Get corresponding columns from Excel data
    if fn in excel_data.columns:
        data['P_closed'] = excel_data[fn]
    else:      
        data['P_closed'] = data['P']
    data['R_closed'] = data['R'] 
    data['E_closed'] = data['E']
    data['S_closed'] = data['S']
    data = data.apply(redistribute_ET, axis=1)

    # ####################################
    # # recompute merged components P/E/R/S to make it close  
    # # ##################################
    # data['RR'] = data['R'] 
    # data['PP'] = data['P']
    # data['SS'] = data['S']
    # data = data.apply(redistribute_ET4Ref, axis=1)

    # ####################################
    # # recompute columns E_P/E/R/S# with P/E/R/S  
    # # ##################################
    # # For P: true value merged values
    # p_columns = [col for col in data.columns if re.match(r'^P\d+$', col) ]
    # for col in p_columns:
    #     data[f'E_{col}'] = abs(data[col] - data['PP'])
    # # For E: E#*20%
    # e_columns = [col for col in data.columns if re.match(r'^E\d+$', col) ]
    # for col in e_columns:
    #     data[f'E_{col}'] = abs(data[col] - data['EE'])
    # # For R: R*7%
    # data['E_R1'] = data['R1']*0.07
    # # For S: true value TWSC_GRACE_Mascon_JPL_calculate
    # s_columns = [col for col in data.columns if re.match(r'^S\d+$', col) ]
    # for col in s_columns:
    #     data[f'E_{col}'] = abs(data[col] - data['SS'])

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
                if redistribute:
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
                data = data.apply(lambda row: compute_newR1(row, l, col), axis=1)  # P and PR_####_P +l
    # print(data.columns.to_list())

    # redistribute r1 for each BCC and composition
    # for combin in ['5414']:#Combinations: 
    #     for m in ['MSD']:#met: # PR_####
    if redistribute:
        for combin in Combinations: 
            for m in met: # PR_####
                r = re.compile(m+"_"+combin+"_[PERS]") # PR_1111_P/E/R/S + null/_w/_r/_r1/r2/_1 (24 columns in total)
                filtered = list(filter(r.match, columns))
                # print(filtered)

                # redistribute r1
                data = data.apply(lambda row: redistributeR(row,filtered,'1'), axis=1)

    data.to_csv(outPath+fn+".csv",index=False)


