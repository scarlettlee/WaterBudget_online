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

def redistribute_ET(row):
    tempE = row['P_closed'] - row['R_closed'] - row['S_closed']

    if tempE > 0:
        row['E_closed'] = tempE

    return row

# input file path
path = os.path.join(os.path.dirname(__file__), '', '/')
csv_folder = os.path.join(os.path.dirname(__file__), '', '')
xlsx_file = csv_folder+"3BasinsComparison/stationsPrecipitation.xlsx"
filePath = os.path.join(os.path.dirname(__file__), '', '3BasinsComparison/')
outPath = os.path.join(os.path.dirname(__file__), '', '3BasinsComparison_mergeClosed_partTrue/')
test = False

# Read Excel file
excel_data = pd.read_excel(xlsx_file, dtype=float)
excel_data = excel_data.rename(columns={2181900: str(2181900),4127800: str(4127800),6742900: str(6742900)})

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

    # Get corresponding columns from Excel data
    if fn in excel_data.columns:
        data['P_closed'] = excel_data[fn]
    else:      
        data['P_closed'] = data['P']

    data['R_closed'] = data['R'] 
    data['E_closed'] = data['E']
    data['S_closed'] = data['S']

    data = data.apply(redistribute_ET, axis=1)
    data.to_csv(outPath+fileName+fl[-4:],index=False) 