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
    
# input file path
path = os.path.join(os.path.dirname(__file__), '', 'WaterBudget_dataset/')
outPath = path + "output_bccClassify/"
# # the percentage to adjust Ep
# EpPerc = 0.1
overAdjustPerc = 0.2
test = False

# traverse input files
if test:
    filePath = path+"output/"
    # pattern = "6742900_bcc.csv"
    pattern = "1147010_bccTest.csv"
else:
    filePath = path+"output/"
    pattern = "*.csv"
fileList = find_pattern(pattern, filePath)
# print(fileList)

df_All = pd.DataFrame()
# water budget components
lab = ['P', 'E', 'R', 'S']
# budget closure correction methods (BCCs) 
met = ['PR', 'CKF', 'MCL', 'MSD']
trueValues = ['P','E','R','S']
# traverse each basin files
for fl in fileList:
    fileName = get_file_name(fl)
    fn = fileName[:-4]
    print("----------------------------------",fn)
    data = pd.read_csv(fl) 

    col = data.columns
    # match remote sensing observations
    r = re.compile("[PERS]\d")
    colFiltered = list(filter(r.match, col))  # ['P1', 'E1', 'R1'...]        

    # match combinations
    exhaustCompnents = [] # for P,E,S [['1', '2', '3', '4', '5'], ['1', '2', '3', '4'], ['1', '2', '3', '4']]
    for m in ['P','E','S']: 
        r = re.compile(m+"\d")
        _colFiltered = list(filter(r.match, col))
        # print(colFiltered)
        exhaustCompnents.append([s[1:] for s in _colFiltered])  # For P: ['1', '2', '3', '4', '5']
    # print('exhaustCompnents',exhaustCompnents)
    Combinations = [a+b+'1'+c for a in exhaustCompnents[0] for b in exhaustCompnents[1] for c in exhaustCompnents[2]]  # ['1111', '1112'...] 
    
    def check_over(row):
        for combin in Combinations:  # ['1111', '1112'...]
            for mInd, m in enumerate(lab):  # P/E/R/S
                for k in met:  # PR/MSD/MCL
                    if abs(row[k + '_' + combin + '_' + m + '_r']) > ((1+overAdjustPerc)*row['E_'+m+combin[mInd]]):
                        row[k + '_' + combin + '_' + m + '_over'] = 1
                    else:
                        row[k + '_' + combin + '_' + m + '_over'] = 0
        return row

    data = data.apply(check_over, axis=1)
    # print(data.columns)
    columns =data.columns

    individualStat = pd.Series()
    individualStat['basin'] = fn
    for l in lab:        #lab: ['P', 'E', 'R', 'S']
        r = re.compile("\w+_\d{4}_"+l+"_over$") # PR_####_P_over
        filtered = list(filter(r.match, columns))
        # print("overly adjusted columns",filtered)
          
        arr = data[filtered]              
        total = arr.count().sum()
        total_overlyAdj = arr.eq(1).sum().sum()
        if total>0:
            # print("{:<10}{:<10}{:<10}{:<10}".format(l,total,total_overlyAdj,total_overlyAdj/total))
            individualStat[l] = 100*total_overlyAdj/total

    df_All = pd.concat([df_All, pd.DataFrame([individualStat])], ignore_index=True)

# print(df_All)
if test:
    print(df_All)
else:
    df_All.to_csv(outPath+"overlyAdjusted.csv",index=False)