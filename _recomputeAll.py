# from curses import A_CHARTEXT
import os, fnmatch
import pandas as pd
import re
import numpy as np
from itertools import combinations
from math import comb
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None

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

def weighted_mean(x):
    return 0.25 * x.iloc[0] + 0.5 * x.iloc[1] + 0.25 * x.iloc[2]

def centered_mean(x):
    return 0.5 * x.iloc[2] - 0.5 * x.iloc[0]

test = False
pth = os.path.join(os.path.dirname(__file__), '', 'data/')
pthTwsc = os.path.join(os.path.dirname(__file__), '', 'twsc/')
output_dir = os.path.join(os.path.dirname(__file__), '', 'dataRecompute/')

if test:
    pattern = '1147010.csv'
    # pattern = '6742900.csv'
else:
    pattern = "*.csv"
fileList = find_pattern(pattern, pth)
# print(fileList)

# 2002-04 to 2016-12
if test:
    TWSC = ['CSR']#,'GFZ','JPL','mascon_JPL'
else:
    TWSC = ['CSR','GFZ','JPL','mascon_JPL']
for fl in fileList:
    fileName = get_file_name(fl)
    print("/////////////////////////////////////////////////////////////////",fileName)

    # df is existing data for validating if BCC functions are right
    # data is used for computing BCC corrected water budget components
    if test:
        data = pd.read_csv(fl)#.head(20)
        # print('data\n',data)
    else:
        data = pd.read_csv(fl) 

    data = data.drop(['SM_GLDAS','SWE_GLDAS'], axis=1)
    data.rename(columns={'Pre_GPCC':'P1','Pre_GPCP':'P2','Pre_Gsmap':'P3','Pre_IMERG':'P4','Pre_PERSIANN_CDR':'P5',
                    'ET_FLUXCOM':'E1','ET_GLDAS':'E2','ET_GLEAM':'E3','ET_PT-JPL':'E4',
                    'TWSC_GRACE_CSR_calculate':'S1','TWSC_GRACE_GFZ_calculate':'S2','TWSC_GRACE_JPL_calculate':'S3','TWSC_GRACE_Mascon_JPL_calculate':'S4',
                    'GRDC':'R1','Unnamed: 0':'date'},inplace=True)
    col = data.columns

    # first, reassign s
    for i,twsc in enumerate(TWSC):
        deltaS = pd.read_csv(pthTwsc+'TWSC_GRACE_'+twsc+'.csv')
        deltaS[fileName+'_'] = deltaS[fileName]
        data['S'+str(i+1)] = deltaS[fileName+'_'].rolling(3, center=True).apply(centered_mean)
        # print(data)

    # second, recompute P and E
    for component in ['P','E']:
        r = re.compile(component + "[12345](?!\d)$")  # P1
        filtered = list(filter(r.match, col))

        for column in filtered:
            data[column] = data[column].rolling(3, center=True).apply(weighted_mean)

    # save data
    if test:
        data.to_csv(output_dir+fileName+'_Test.csv',index=False)
    else:
        data.to_csv(output_dir+get_file_name(fl)+fl[-4:],index=False)    