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

# The function checks the values of a and b and returns different values based on the conditions
def compare_values(row, l, col):
    a = row[l]
    b = row[col]   
    
    # Check if b is null
    if pd.isnull(b):    
        return b
    # Check if a is positive and b is greater than twice of a
    elif a > 0 and b > 2 * a:
        return 1
    # Check if a is negative and b is less than twice of a
    elif a < 0 and b < 2 * a:
        return 2
    # Check if a is negative and b is positive
    elif a < 0 and b > 0:
        return 3
    # Check if a is positive and b is negative
    elif a > 0 and b < 0:
        return 4
    # If none of the above conditions are met
    else:
        return 0
    
# input file path
path = os.path.join(os.path.dirname(__file__), '', 'WaterBudget_online/')
filePath = path+"3BasinsComparison/"
outPath = path + "3BasinsComparison_output/"
test = False

# traverse input files
pattern = "*.csv"
if test:
    pattern = "1147010_bccTest.csv"
fileList = find_pattern(pattern, filePath)
# print(fileList)

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

    individualStat = pd.Series()
    individualStat['basin'] = fn
    for l in lab:        #['P', 'E', 'R', 'S']
        for m in met:    #['PR', 'CKF', 'MCL', 'MSD']    
            r = re.compile(m+"_\d{4}_"+l+"$")
            filtered = list(filter(r.match, columns))
            # print('filtered columns',filtered)

            for col in filtered:
                data[col] = data.apply(lambda row: compare_values(row, l, col), axis=1)
            
            arr = data[filtered]              
            total = arr.count().sum()
            zero_count = arr.eq(0).sum().sum()
            one_count = arr.eq(1).sum().sum()
            two_count = arr.eq(2).sum().sum()
            three_count = arr.eq(3).sum().sum()
            four_count = arr.eq(4).sum().sum()            
            # print("{:<3}{:<4}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}".format(l,m,zero_count,one_count,two_count,three_count,four_count,total))
            total_extreme = one_count+two_count+three_count+four_count
            if total>0:
                # print("{:<3}{:<4}{:<5}".format(l,m,total_extreme/total))
                individualStat[l+'_'+m] = 100*total_extreme/total
                individualStat[l+'_'+m+'_1'] = 100*one_count/total
                individualStat[l+'_'+m+'_2'] = 100*two_count/total
                individualStat[l+'_'+m+'_3'] = 100*three_count/total
                individualStat[l+'_'+m+'_4'] = 100*four_count/total

    # print(individualStat)
    # df_All.loc[len(df_All)] = individualStat
    df_All = pd.concat([df_All, pd.DataFrame([individualStat])], ignore_index=True)

# print(df_All)
if test:
    df_All.to_csv(outPath+"extremesAll_test.csv",index=False)
else:
    df_All.to_csv(outPath+"extremesAll.csv",index=False)

    # data.to_csv(outPath+fileName[:-4]+'_extreme'+fl[-4:],index=False)

    # # r = re.compile("\w+_\d{4}_[PE]$")
    # # r = re.compile("\w+_\d{4}_[P]$")
    # # r = re.compile("\w+_\d{4}_[E]$")
    # r = re.compile("\w+_\d{4}_[R]$")
    # filtered = list(filter(r.match, columns))
    # # print('filtered columns',filtered)

    # arr = data[filtered]  
    # neg = (arr<0).sum().sum()
    # total = arr.size
    # # print('total: ',total)
    # # print('negative: ',neg)
    # # print(round(100*neg/total,2))
    # print("{:<30}{:<10}".format(fileName, round(100*neg/total,2)))
