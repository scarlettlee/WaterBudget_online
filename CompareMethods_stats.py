import os
import pandas as pd
import numpy as np
from globVar import basin3Flag, find_pattern, get_file_name, compute_stats
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# For compare:  MSD_5414_S, MSD_5414_S, MSD_5414_S_1, MSD_5414_S_2 ]#
datasetFolders = ['BasinsComparison', 'BasinsComparison_obsIntroduced', 'redistribution_obsIn_outliersRedistributed', 'redistribution_outliers_mergeClosed_partTrue']
suffix = ['','','_1','_2']
test = False

# Define paths and folders
csv_folder = os.path.join(os.path.dirname(__file__), '', '')

# new 28 basins or not
basins28 = not basin3Flag
out = [] # store the results
for index, dataset in enumerate(datasetFolders):    
    df_statsAll = pd.DataFrame(columns=['Dataset','Basin', 'index', 'method', 'PBIAS', 'CC', 'RMSE', 'ME', 'ME1', 'MAE', 'MAPE'])

    if basin3Flag:  
        csv_files = find_pattern("*.csv", csv_folder+'3'+dataset+'/')
        if test:
            # csv_files = [csv_folder+'3'+dataset+'/2181900.csv']    
            csv_files = [csv_folder+'3'+dataset+'/4127800.csv']    
            # csv_files = [csv_folder+'3'+dataset+'/6742900.csv']    
        
        refDataFolder = '3BasinsComparison_mergeClosed_partTrue'
    else:
        csv_files = find_pattern("*.csv", csv_folder+'28'+dataset+'/')
        if test:
            csv_files = [csv_folder+'28'+dataset+'/4127800.csv']  

        refDataFolder = '28BasinsComparison_mergeClosed_partTrue'

    # Iterate through CSV files: each basin
    for csv_file in csv_files:
        if "_bcc" in csv_file:
            continue
        file_name = get_file_name(csv_file).split('_')[0]
        print("-------------------------------------Processing file:", file_name) 

        # Read reference data
        # P_closed	R_closed	E_closed	S_closed
        ref_data = pd.read_csv(os.path.join(csv_folder, refDataFolder, file_name+'_bcc.csv'))
        ref_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        # only keep columns with '_closed'
        ref_data = ref_data[[col for col in ref_data.columns if col.endswith('_closed')]]

        # Read CSV data
        csv_data = pd.read_csv(csv_file)
        csv_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        # print(len(csv_data))

        # if not the last foler
        # combine csv data and reference data and drop unclosed-ref rows
        if dataset != 'redistribution_outliers_mergeClosed_partTrue':
            csv_data = pd.concat([csv_data, ref_data], axis=1).dropna(subset=['P_closed','R_closed','E_closed','S_closed'])
        # print(len(csv_data))

        # iterate PERS
        for component in ['P', 'E', 'R', 'S']:
            required_columns = [col for col in csv_data.columns if col.endswith('_'+component+suffix[index])]

            # DataFrame to store statistics
            df_stats = pd.DataFrame(columns=['Dataset','Basin', 'Combination', 'PBIAS', 'CC', 'RMSE', 'ME', 'ME1', 'MAE', 'MAPE'])
        
            # Compute statistics for each column pair
            for csv_column in required_columns:
                pbias, cc, rmse, me, me1, mae, mape = compute_stats(csv_data[component+'_closed'][3:], csv_data[csv_column][3:])
        
                # Append statistics to DataFrame
                new_row = pd.DataFrame([{'Dataset': dataset,'Basin': file_name, 'Combination': csv_column, 'PBIAS': pbias, 'CC': cc, 
                            'RMSE': rmse, 'ME': me, 'ME1': me1, 'MAE': mae, 'MAPE': mape}])
                df_stats = pd.concat([df_stats,new_row],ignore_index=True) 

            # output integrated results        
            unique_start_chars = set(col.split('_')[0] for col in required_columns)
            # print(unique_start_chars)
            unique_start_chars = list(unique_start_chars)
            # print(unique_start_chars)
            for method in unique_start_chars:
                target_rows = [row for row in df_stats.Combination if row.startswith(method+'_')]
                pbias, cc, rmse, me, me1, mae, mape = df_stats[df_stats.Combination.isin(target_rows)][['PBIAS', 'CC', 'RMSE', 'ME', 'ME1', 'MAE', 'MAPE']].mean()

                new_row = pd.DataFrame([{'Dataset': dataset,'Basin': file_name, 'index':component, 'method': method, 'PBIAS': pbias, 'CC': cc, 
                                'RMSE': rmse, 'ME': me, 'ME1': me1, 'MAE': mae, 'MAPE': mape}])
                df_statsAll = pd.concat([df_statsAll,new_row],ignore_index=True)

    # df_statsAll = df_statsAll[(df_statsAll['method'] != 'MSD')]
    meanStats = df_statsAll[['PBIAS','CC','RMSE','ME','ME1','MAE','MAPE']].mean()
    out.append(meanStats)
    print('In '+dataset+':\n',meanStats)

    # # 按方法分组并计算每个分组的统计指标    
    # grouped_stats = df_statsAll.groupby('method')[['PBIAS','CC','RMSE','ME','ME1','MAE','MAPE']].mean()
    # # 遍历每个方法，输出统计指标
    # for method, stats in grouped_stats.iterrows():
    #     print(f'In {dataset}, Method: {method}:\n', stats)
    #     out.append(stats)

    # # 筛选出method为CKF和MSD的项
    # filtered_df = df_statsAll[(df_statsAll['method'] == 'CKF') | (df_statsAll['method'] == 'MSD')]
    # # 按index和method分组并计算每个分组的统计指标
    # grouped_stats = filtered_df.groupby(['index', 'method'])[['PBIAS','CC','RMSE','ME','ME1','MAE','MAPE']].mean()
    # # 遍历每个分组，输出统计指标
    # for (index, method), stats in grouped_stats.iterrows():
    #     print(f'In {dataset}, Index: {index}, Method: {method}:\n', stats)
    #     out.append(stats)


# print(out)    
