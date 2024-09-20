import os
import pandas as pd
import numpy as np
from globVar import basin3Flag, find_pattern, get_file_name, compute_stats

# Define paths and folders
csv_folder = os.path.join(os.path.dirname(__file__), '', '')
if basin3Flag:  
    xlsx_file = csv_folder+"3BasinsComparison - old/stationsPrecipitation.xlsx"
    output_folder = os.path.join(os.path.dirname(__file__), '', '3stats_mergedClosed_partTrue/')
    csv_files = find_pattern("*.csv", csv_folder+'3redistribution_outliers_mergeClosed_partTrue/')

    # Read Excel file
    excel_data = pd.read_excel(xlsx_file, dtype=float)
    excel_data = excel_data.rename(columns={2181900: str(2181900),4127800: str(4127800),6742900: str(6742900)})
else:
    xlsx_file = csv_folder+"3BasinsComparison/stationsPrecipitation.xlsx"
    output_folder = os.path.join(os.path.dirname(__file__), '', '28stats_mergedClosed_partTrue/')
    csv_files = find_pattern("*.csv", csv_folder+'28redistribution_outliers_mergeClosed_partTrue/')

    # Read Excel file
    excel_data = pd.read_excel(xlsx_file, dtype=float)
    excel_data = excel_data.rename(columns={1159100: str(1159100), 1234150: str(1234150), 2180800: str(2180800), 2181900: str(2181900), 2909150: str(2909150), 2912600: str(2912600), 3265601: str(3265601), 3629001: str(3629001), 4103200: str(4103200), 4115201: str(4115201), 4127800: str(4127800), 4146281: str(4146281), 4146360: str(4146360), 4147703: str(4147703), 4150450: str(4150450), 4150500: str(4150500), 4152050: str(4152050), 4207900: str(4207900), 4208025: str(4208025), 4213711: str(4213711), 4214270: str(4214270), 4243151: str(4243151), 5404270: str(5404270), 6226800: str(6226800), 6340110: str(6340110), 6435060: str(6435060), 6457010: str(6457010), 6590700: str(6590700)})

# new 28 basins or not
basins28 = not basin3Flag

# ind = ''
# ind = '_1'
# ind = '_2'
out = []
for ind in ['','_1','_2']:    #
    df_statsAll = pd.DataFrame(columns=['Basin', 'index', 'method', 'PBIAS', 'CC', 'RMSE', 'ME', 'ME1', 'MAE', 'MAPE'])
    # Iterate through CSV files
    for csv_file in csv_files:
    # for csv_file in [r'E:\Huan\OneDrive - pku.edu.cn\LIHUAN\coding_desktop\python\ArcPy\FileProcess\WaterBudget_online\28redistribution_outliers_mergeClosed_partTrue\1159100.csv']:
        file_name = get_file_name(csv_file).split('_')[0]
        print("-------------------------------------Processing file:", file_name)

        # Read CSV data
        csv_data = pd.read_csv(csv_file)
        csv_data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # ***************************** P ***************************
        if basins28:
            csv_data.rename(
                columns={'P1': 'Pre_GPCC', 'P2': 'Pre_GPM', 'P3': 'Pre_MSWEP', 'P4': 'Pre_PERSIANN',
                        'P': 'P_Merge'}, inplace=True)
            # Extract required columns from CSV data
            required_columns = ['P_Merge', 'Pre_GPCC', 'Pre_GPM', 'Pre_MSWEP', 'Pre_PERSIANN']
        else:
            csv_data.rename(
                columns={'P1': 'Pre_GPCC', 'P2': 'Pre_GPCP', 'P3': 'Pre_Gsmap', 'P4': 'Pre_IMERG', 'P5': 'Pre_PERSIANN_CDR',
                        'P': 'P_Merge'}, inplace=True)
            # Extract required columns from CSV data
            required_columns = ['P_Merge', 'Pre_GPCC', 'Pre_GPCP', 'Pre_Gsmap', 'Pre_IMERG', 'Pre_PERSIANN_CDR']
        required_columns += [col for col in csv_data.columns if col.endswith('_P'+ind)]
        # print(required_columns)
        
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
            # print(unique_start_chars)
            unique_start_chars = list(unique_start_chars)
            # print(unique_start_chars)
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
        if basins28:
            csv_data.rename(
                columns={'E1': 'ET_ERA5', 'E2': 'ET_GLEAM', 'E3': 'ET_MERRA'}, inplace=True)
            # Extract required columns from CSV data
            required_columns = ['ET_ERA5', 'ET_GLEAM', 'ET_MERRA']
        else:
            csv_data.rename(
                columns={'E1': 'ET_FLUXCOM', 'E2': 'ET_GLDAS', 'E3': 'ET_GLEAM', 'E4': 'ET_PT-JPL'}, inplace=True)
            # Extract required columns from CSV data
            required_columns = ['ET_FLUXCOM', 'ET_GLDAS', 'ET_GLEAM', 'ET_PT-JPL']
        required_columns += [col for col in csv_data.columns if col.endswith('_E'+ind)]
        
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

        # ***************************** R ***************************
        csv_data.rename(
            columns={'R1': 'GRDC'}, inplace=True)
        # Extract required columns from CSV data
        required_columns = ['GRDC']
        required_columns += [col for col in csv_data.columns if col.endswith('_R'+ind)]
        
        df_stats = pd.DataFrame(columns=['Basin', 'Combination', 'PBIAS', 'CC', 'RMSE', 'ME', 'ME1', 'MAE', 'MAPE'])
        
        for csv_column in required_columns:
            pbias, cc, rmse, me, me1, mae, mape = compute_stats(csv_data['R_closed'], csv_data[csv_column])

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

            new_row = pd.DataFrame([{'Basin': file_name, 'index':'R', 'method': method, 'PBIAS': pbias, 'CC': cc, 
                            'RMSE': rmse, 'ME': me, 'ME1': me1, 'MAE': mae, 'MAPE': mape}])
            df_statsAll = pd.concat([df_statsAll,new_row],ignore_index=True)
        # ***************************** R ***************************

        # ***************************** TWSC ***************************
        if basins28:
            csv_data.rename(
                columns={'S1': 'GRACE_CSR', 'S2': 'GRACE_GFZ', 'S3': 'GRACE_JPL'}, inplace=True)
            # Extract required columns from CSV data
            required_columns = ['GRACE_CSR', 'GRACE_GFZ', 'GRACE_JPL']
        else:
            csv_data.rename(
                columns={'S1': 'TWSC_GRACE_CSR', 'S2': 'TWSC_GRACE_GFZ', 'S3': 'TWSC_GRACE_JPL',
                        'S4': 'TWSC_GRACE_Mascon_JPL'}, inplace=True)
            # Extract required columns from CSV data
            required_columns = ['TWSC_GRACE_CSR', 'TWSC_GRACE_GFZ', 'TWSC_GRACE_JPL', 'TWSC_GRACE_Mascon_JPL']
        required_columns += [col for col in csv_data.columns if col.endswith('_S'+ind)]

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
    output_csv = os.path.join(output_folder, 'comparison_allBasins'+ind+'.csv')
    df_statsAll.to_csv(output_csv, index=False)

    out.append(df_statsAll[['PBIAS','CC','RMSE','ME','ME1','MAE','MAPE']].mean())
    # print('In '+ind+':',df_statsAll[['PBIAS','CC','RMSE','ME','ME1','MAE','MAPE']].mean())

print(out)