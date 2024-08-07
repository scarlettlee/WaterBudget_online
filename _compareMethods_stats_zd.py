import os
import fnmatch
import pandas as pd
import numpy as np
import re

# Function to find files matching a pattern in a directory
def find_pattern(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


# Function to extract file name from path
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

# Define paths and folders
# csv_folder = os.path.join(os.path.dirname(__file__), '', '3BasinsComparison/')
# csv_folder = os.path.join(os.path.dirname(__file__), '', '3BasinsComparison_mergeClosed/')
csv_folder = os.path.join(os.path.dirname(__file__), '', '')
xlsx_file = csv_folder+"3BasinsComparison/stationsPrecipitation.xlsx"
# output_folder = os.path.join(os.path.dirname(__file__), '', '3BasinsComparison_output/')
# output_folder = os.path.join(os.path.dirname(__file__), '', 'stats_mergedClosed/')
output_folder = os.path.join(os.path.dirname(__file__), '', 'stats_mergedClosed_partTrue/')


# Find CSV files
# csv_files = find_pattern("*.csv", csv_folder+'redistribution_outliers_mergeClosed/')
csv_files = find_pattern("*.csv", csv_folder+'redistribution_outliers_mergeClosed_partTrue/')

# Read Excel file
excel_data = pd.read_excel(xlsx_file, dtype=float)
excel_data = excel_data.rename(columns={2181900: str(2181900),4127800: str(4127800),6742900: str(6742900)})
# print(excel_data.columns)

df_statsAll = pd.DataFrame(columns=['Basin', 'index', 'method', 'PBIAS', 'CC', 'RMSE', 'ME', 'ME1', 'MAE', 'MAPE'])
# Iterate through CSV files
for csv_file in csv_files:
    file_name = get_file_name(csv_file).split('_')[0]
    print("-------------------------------------Processing file:", file_name)

    # Read CSV data
    csv_data = pd.read_csv(csv_file)
    csv_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ***************************** P ***************************
    csv_data.rename(
        columns={'P1': 'Pre_GPCC', 'P2': 'Pre_GPCP', 'P3': 'Pre_Gsmap', 'P4': 'Pre_IMERG', 'P5': 'Pre_PERSIANN_CDR',
                 'P': 'P_Merge'}, inplace=True)
    # Extract required columns from CSV data
    required_columns = ['P_Merge', 'Pre_GPCC', 'Pre_GPCP', 'Pre_Gsmap', 'Pre_IMERG', 'Pre_PERSIANN_CDR']
    required_columns += [col for col in csv_data.columns if col.endswith('_P')]
    
    # Get corresponding columns from Excel data
    if file_name in excel_data.columns:
        excel_columns = excel_data[file_name]
        # DataFrame to store statistics
        df_stats = pd.DataFrame(columns=['Basin', 'Combination', 'PBIAS', 'CC', 'RMSE', 'ME', 'ME1', 'MAE', 'MAPE'])
    
        # Compute statistics for each column pair
        for csv_column in required_columns:
            pbias, cc, rmse, me, me1, mae, mape = compute_stats(excel_columns, csv_data[csv_column])
    
            # Append statistics to DataFrame
            new_row = pd.DataFrame([{'Basin': file_name, 'Combination': csv_column, 'PBIAS': pbias, 'CC': cc, 
                         'RMSE': rmse, 'ME': me, 'ME1': me1, 'MAE': mae, 'MAPE': mape}])
            df_stats = pd.concat([df_stats,new_row],ignore_index=True)
    
        # # Save DataFrame to CSV
        # output_csv = os.path.join(output_folder, file_name + '_comparison_P.csv')
        # df_stats.to_csv(output_csv, index=False)

        # output integrated results        
        unique_start_chars = set(col.split('_')[0] for col in required_columns)
        unique_start_chars = list(unique_start_chars)
        for method in unique_start_chars:
            target_rows = [row for row in df_stats.Combination if row.startswith(method+'_')]
            pbias, cc, rmse, me, me1, mae, mape = df_stats[df_stats.Combination.isin(target_rows)][['PBIAS', 'CC', 'RMSE', 'ME', 'ME1', 'MAE', 'MAPE']].mean()

            new_row = pd.DataFrame([{'Basin': file_name, 'index':'P', 'method': method, 'PBIAS': pbias, 'CC': cc, 
                            'RMSE': rmse, 'ME': me, 'ME1': me1, 'MAE': mae, 'MAPE': mape}])
            df_statsAll = pd.concat([df_statsAll,new_row],ignore_index=True)
    else:
        print("No corresponding column found in Excel file for:", file_name)    
    # ***************************** P ***************************

    # ***************************** ET ***************************
    csv_data.rename(
        columns={'E1': 'ET_FLUXCOM', 'E2': 'ET_GLDAS', 'E3': 'ET_GLEAM', 'E4': 'ET_PT-JPL'}, inplace=True)
    # Extract required columns from CSV data
    required_columns = ['ET_FLUXCOM', 'ET_GLDAS', 'ET_GLEAM', 'ET_PT-JPL']
    required_columns += [col for col in csv_data.columns if col.endswith('_E')]
    
    df_stats = pd.DataFrame(columns=['Basin', 'Combination', 'PBIAS', 'CC', 'RMSE', 'ME', 'ME1', 'MAE', 'MAPE'])
    
    for csv_column in required_columns:
        pbias, cc, rmse, me, me1, mae, mape = compute_stats(csv_data['E_closed'], csv_data[csv_column])

        # Append statistics to DataFrame
        new_row = pd.DataFrame([{'Basin': file_name, 'Combination': csv_column, 'PBIAS': pbias, 'CC': cc, 
                        'RMSE': rmse, 'ME': me, 'ME1': me1, 'MAE': mae, 'MAPE': mape}])
        df_stats = pd.concat([df_stats,new_row],ignore_index=True)
    
    # output_csv = os.path.join(output_folder, file_name + '_comparison_E.csv')
    # df_stats.to_csv(output_csv, index=False)

    # output integrated results        
    unique_start_chars = set(col.split('_')[0] for col in required_columns)
    unique_start_chars = list(unique_start_chars)
    for method in unique_start_chars:
        target_rows = [row for row in df_stats.Combination if row.startswith(method+'_')]
        pbias, cc, rmse, me, me1, mae, mape = df_stats[df_stats.Combination.isin(target_rows)][['PBIAS', 'CC', 'RMSE', 'ME', 'ME1', 'MAE', 'MAPE']].mean()

        new_row = pd.DataFrame([{'Basin': file_name, 'index':'E', 'method': method, 'PBIAS': pbias, 'CC': cc, 
                        'RMSE': rmse, 'ME': me, 'ME1': me1, 'MAE': mae, 'MAPE': mape}])
        df_statsAll = pd.concat([df_statsAll,new_row],ignore_index=True)
    # ***************************** ET ***************************

    # ***************************** TWSC ***************************
    csv_data.rename(
        columns={'S1': 'TWSC_GRACE_CSR', 'S2': 'TWSC_GRACE_GFZ', 'S3': 'TWSC_GRACE_JPL',
                 'S4': 'TWSC_GRACE_Mascon_JPL'}, inplace=True)
    # Extract required columns from CSV data
    required_columns = ['TWSC_GRACE_CSR', 'TWSC_GRACE_GFZ', 'TWSC_GRACE_JPL', 'TWSC_GRACE_Mascon_JPL']
    required_columns += [col for col in csv_data.columns if col.endswith('_S')]

    df_stats = pd.DataFrame(columns=['Basin', 'Combination', 'PBIAS', 'CC', 'RMSE', 'ME', 'ME1', 'MAE', 'MAPE'])

    for csv_column in required_columns:
        pbias, cc, rmse, me, me1, mae, mape = compute_stats(csv_data['S_closed'], csv_data[csv_column])

        # Append statistics to DataFrame
        new_row = pd.DataFrame([{'Basin': file_name, 'Combination': csv_column, 'PBIAS': pbias, 'CC': cc, 
                        'RMSE': rmse, 'ME': me, 'ME1': me1, 'MAE': mae, 'MAPE': mape}])
        df_stats = pd.concat([df_stats,new_row],ignore_index=True)

    # output_csv = os.path.join(output_folder, file_name + '_comparison_S.csv')
    # df_stats.to_csv(output_csv, index=False)
    # output integrated results        
    unique_start_chars = set(col.split('_')[0] for col in required_columns)
    unique_start_chars = list(unique_start_chars)
    for method in unique_start_chars:
        target_rows = [row for row in df_stats.Combination if row.startswith(method+'_')]
        pbias, cc, rmse, me, me1, mae, mape = df_stats[df_stats.Combination.isin(target_rows)][['PBIAS', 'CC', 'RMSE', 'ME', 'ME1', 'MAE', 'MAPE']].mean()

        new_row = pd.DataFrame([{'Basin': file_name, 'index':'S', 'method': method, 'PBIAS': pbias, 'CC': cc, 
                        'RMSE': rmse, 'ME': me, 'ME1': me1, 'MAE': mae, 'MAPE': mape}])
        df_statsAll = pd.concat([df_statsAll,new_row],ignore_index=True)
    # ***************************** TWSC ***************************    

# print(df_statsAll)
output_csv = os.path.join(output_folder, 'comparison_allBasins.csv')
df_statsAll.to_csv(output_csv, index=False)