import os, fnmatch
import pandas as pd
import re
import numpy as np

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

def redistribute_ET(row,min):
    if row['E']+row['residual'] < 0:
        row['E_closed'] = min
        row['S_closed'] = row['S'] + row['residual'] + row['E'] - row['E_closed']
    else:
        row['E_closed'] = row['E']+row['residual']
        row['S_closed'] = row['S']
    return row
    
# input file path
path = os.path.join(os.path.dirname(__file__), '', '/')
filePath = os.path.join(os.path.dirname(__file__), '', '3BasinsComparison/')
outPath = os.path.join(os.path.dirname(__file__), '', '3BasinsComparison_mergeClosed/')
test = False

# traverse input files
pattern = "*.csv"
if test:
    pattern = "4127800_bcc.csv"
fileList = find_pattern(pattern, filePath)
print(fileList)

df_All = pd.DataFrame()
# water budget components
lab = ['P', 'E', 'R', 'S']
# budget closure correction methods (BCCs) 
met = ['PR', 'CKF', 'MCL', 'MSD']
# traverse each basin files
for fl in fileList:
    fileName = get_file_name(fl)
    fn = fileName[:-4]
    print("----------------------------------",fn)
    data = pd.read_csv(fl)  
    columns =data.columns

    # compute residual of merged components P/E/R/S
    data['residual'] = data['P'] - data['E'] - data['R'] - data['S']
    minET = min(data['E'])
    print('minET',minET)

    # close P,E,R,S by redistribute the residual to ET
    data['P_closed'] = data['P']     
    data['R_closed'] = data['R'] 
    data = data.apply(redistribute_ET, axis=1, min = minET)

    # close P,E,R,S by considering physical processes
    # the mean value of residual series is close to 0, so we can adjust the P/E/R/S according to the whole time series
    # The uncertainty ranks: P,R,S,E
    # The time series can be adjusted for each component to minimize the sum of squired residual close to 0     
    # data['P_closed_Ts'] = 
    # data['E_closed_Ts'] = 
    # data['R_closed_Ts'] = 
    # data['S_closed_Ts'] = 

    data.to_csv(outPath+fileName+fl[-4:],index=False) 