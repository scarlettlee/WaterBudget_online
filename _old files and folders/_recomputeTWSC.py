# from curses import A_CHARTEXT
import os, fnmatch
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

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

test = True
middleDay = False # take the middle day for daily interpolation
pth = os.path.join(os.path.dirname(__file__), '', 'data/')
pthTwsc = os.path.join(os.path.dirname(__file__), '', 'twsc/')
output_dir = os.path.join(os.path.dirname(__file__), '', 'dataTWSC1/')
# fl = pth +'3629001.csv'

if test:
    # pattern = '1147010.csv'
    pattern = '6742900.csv'
else:
    pattern = "*.csv"
fileList = find_pattern(pattern, pth)
# print(fileList)

# 2002-04 to 2016-12
TWSC = ['CSR']#,'GFZ','JPL','mascon_JPL'
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

    for twsc in TWSC:
        oriDeltaS = pd.read_csv(pthTwsc+'TWSC_GRACE_'+twsc+'.csv')
        deltaS = oriDeltaS[['Unnamed: 0',fileName]]
        # deltaS = oriDeltaS.iloc[3:181]
        # Assuming df is your DataFrame and 'date' is your date column
        deltaS['date'] = pd.to_datetime(deltaS['Unnamed: 0'], format='%Y%m')
        if middleDay:
            deltaS['date'] = pd.to_datetime(deltaS['Unnamed: 0'], format='%Y%m') + pd.Timedelta(days=14)
        # deltaS[fileName] = deltaS[fileName].interpolate(method='linear')
        # print(deltaS[['date',fileName]])

        # Set 'date' as the index of the DataFrame
        deltaS.set_index('date', inplace=True)
        deltaS['Unnamed: 0'] = deltaS['Unnamed: 0'].astype(int)
        print('deltaS',deltaS)

        # Resample the DataFrame to the daily level
        df_daily = deltaS.resample('D').asfreq()

        # Apply linear interpolation to fill in the missing daily values
        df_daily[fileName] = df_daily[fileName].interpolate(method='linear')
        print("df_daily",df_daily)

        # Group the DataFrame by month and apply the scaling function to each month
        df_monthly = df_daily.resample('M').mean() #.apply(scale_monthly_values)
        df_monthly['Unnamed: 0'] = df_monthly['Unnamed: 0'].astype(int)
        # print(df_daily)

        x = df_monthly[fileName][1:-1]
        y = deltaS[fileName][1:-1]
        # Fit a linear regression model to the data
        slope, _ = np.polyfit(x, y, 1)
        print('scale factor',slope)
        # # Create a range of x values that cover the range of your data
        # x_values = np.linspace(x.min(), x.max(), 100)
        # # Compute the corresponding y values for the regression line
        # y_values = slope * x_values + intercept
        # # Plot the data and the regression line
        # plt.scatter(x, y)
        # plt.plot(x_values, y_values, color='red')
        # # Add the equation to the plot
        # plt.text(0.05, 0.95, f'y = {slope:.2f}x + {intercept:.2f}', transform=plt.gca().transAxes)

        # df_daily = df_daily.droplevel(0).reset_index().iloc[0:-1]
        # df_daily[fileName] = df_daily[fileName].rolling(7, center=True).median()

        df_daily[fileName+'_'] = df_daily[fileName]*slope
        print('df_daily',df_daily)
        # Resample to a monthly frequency, taking the first value of each month
        df_monthly = df_daily.resample('M').first()
        # Compute the difference between each pair of consecutive months
        df_monthly_diff = df_monthly.diff()
        print('df_monthly_diff',df_monthly_diff)
        # Shift the index to the first month
        df_monthly_diff.index = df_monthly_diff.index.shift(-1, freq='M')
        print('df_monthly_diff',df_monthly_diff)

        deltaS = deltaS.reset_index()
        deltaS[fileName+'_'] = df_monthly_diff.dropna().reset_index()[fileName+'_']
        print(deltaS)

        # save data
        if test:            
            if middleDay:
                df_daily.reset_index().to_csv(output_dir+fileName+'_TestDaily.csv',index=False)
                deltaS.to_csv(output_dir+fileName+'_Test.csv',index=False)
            else:
                df_daily.reset_index().to_csv(output_dir+fileName+'_TestDaily_firstDay.csv',index=False)
                deltaS.to_csv(output_dir+fileName+'_Test_firstDay.csv',index=False)
        else:
            df_daily.to_csv(output_dir+get_file_name(fl)+'_bcc'+fl[-4:],index=False)    

plt.show()