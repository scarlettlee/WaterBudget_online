import os, fnmatch
import re
import numpy as np
import pandas as pd

basin3Flag= False

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
    
# Function to compute statistics
def compute_stats(true_values, corrected_values):
    if corrected_values.isnull().all():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    data = pd.concat([true_values, corrected_values], axis=1).dropna()
    true_values = data.iloc[:, 0]
    corrected_values = data.iloc[:, 1]

    pbias = 100 * ((true_values - corrected_values).sum()) / true_values.sum()
    cc = true_values.corr(corrected_values)
    rmse = ((true_values - corrected_values) ** 2).mean() ** .5
    me = (true_values - corrected_values).mean()
    me1 = (true_values - corrected_values.mean()).abs().mean()
    mae = (true_values - corrected_values).abs().mean()
    mape = 100 * ((true_values - corrected_values) / true_values).abs().mean()
    return pbias, cc, rmse, me, me1, mae, mape