import os, fnmatch
import pandas as pd
import re
import numpy as np
from globVar import basin3Flag,  obsIntroduced, find_pattern, get_file_name

def redistribute_ET(row):
    tempE = row['P_closed'] - row['R_closed'] - row['S_closed']

    if tempE > 0:
        row['E_closed'] = tempE

    return row

# input file path
path = os.path.join(os.path.dirname(__file__), '', '/')
csv_folder = os.path.join(os.path.dirname(__file__), '', '')
if basin3Flag:
    xlsx_file = csv_folder+"3BasinsComparison - old/stationsPrecipitation.xlsx"
    if obsIntroduced:
        filePath = os.path.join(os.path.dirname(__file__), '', '3BasinsComparison/')
    else:
        filePath = os.path.join(os.path.dirname(__file__), '', '3BasinsComparison_obsIntroduced/')
    outPath = os.path.join(os.path.dirname(__file__), '', '3BasinsComparison_mergeClosed_partTrue/')

    # Read Excel file
    excel_data = pd.read_excel(xlsx_file, dtype=float)
    excel_data = excel_data.rename(columns={2181900: str(2181900),4127800: str(4127800),6742900: str(6742900)})
else:
    xlsx_file = csv_folder+"3BasinsComparison/stationsPrecipitation.xlsx"
    filePath = os.path.join(os.path.dirname(__file__), '', '28BasinsComparison_obsIntroduced/')
    outPath = os.path.join(os.path.dirname(__file__), '', '28BasinsComparison_mergeClosed_partTrue/')

    # Read Excel file
    excel_data = pd.read_excel(xlsx_file, dtype=float)
    excel_data = excel_data.rename(columns={1159100: str(1159100), 1234150: str(1234150), 2180800: str(2180800), 2181900: str(2181900), 2909150: str(2909150), 2912600: str(2912600), 3265601: str(3265601), 3629001: str(3629001), 4103200: str(4103200), 4115201: str(4115201), 4127800: str(4127800), 4146281: str(4146281), 4146360: str(4146360), 4147703: str(4147703), 4150450: str(4150450), 4150500: str(4150500), 4152050: str(4152050), 4207900: str(4207900), 4208025: str(4208025), 4213711: str(4213711), 4214270: str(4214270), 4243151: str(4243151), 5404270: str(5404270), 6226800: str(6226800), 6340110: str(6340110), 6435060: str(6435060), 6457010: str(6457010), 6590700: str(6590700)})


test = False
# traverse input files
pattern = "*.csv"
if test:
    pattern = "4127800_bcc.csv"
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