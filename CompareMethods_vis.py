import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from globVar import basin3Flag, find_pattern, get_file_name, compute_stats
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# For compare:  MSD_5414_S, MSD_5414_S, MSD_5414_S_1, MSD_5414_S_2 ]#
datasetFolders = ['BasinsComparison1', 'BasinsComparison_obsIntroduced1', 'redistribution_obsIn_outliersRedistributed', 'redistribution_outliers_mergeClosed_partTrue']
suffix = ['','','_1','_2']
test = True

# Define paths and folders
csv_folder = os.path.join(os.path.dirname(__file__), '', '')
outPath = os.path.join(os.path.dirname(__file__), '', 'results/')

targetMethod = "CKF"
out = [] # store the results
# Create scatter plots with subplots
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 8), sharex=True, sharey=True)
fig.suptitle(f'Scatter Plots of DataFolders vs Reference Data ('+ targetMethod + ')')
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
            csv_files = [csv_folder+'28'+dataset+'/2181900.csv']    
            # csv_files = [csv_folder+'28'+dataset+'/4127800.csv']  

        refDataFolder = '28BasinsComparison_mergeClosed_partTrue'

    # Iterate through CSV files: each basin
    for csv_file in csv_files:
        file_name = get_file_name(csv_file).split('_')[0]
        print("-------------------------------------Processing file:", dataset, file_name) 

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
        for row, component in enumerate(['P', 'E', 'R', 'S']):
            # PR_5414_P(/PR_5414_P_1/PR_5414_P_2), CKF_5414_P, MCL_5414_P, MSD_5414_P
            required_columns = [col for col in csv_data.columns if col.endswith('_'+component+suffix[index])]

            # DataFrame to store method data and reference data
            df_allPairs = pd.DataFrame(columns=['method','Value', 'Reference'])

            # DataFrame to store statistics
            df_stats = pd.DataFrame(columns=['Dataset','Basin', 'Combination', 'PBIAS', 'CC', 'RMSE', 'ME', 'ME1', 'MAE', 'MAPE'])
        
            # Compute statistics for each column pair
            for csv_column in required_columns:
                # Append method data to DataFrame
                m = csv_column.split('_')[0]
                new_frame = csv_data[[csv_column, component+'_closed']]\
                        .rename(columns={csv_column: 'Value', component+'_closed': 'Reference'})
                new_frame['method'] = m
                df_allPairs = pd.concat([df_allPairs,new_frame[3:]],ignore_index=True)

                # Compute statistics
                pbias, cc, rmse, me, me1, mae, mape = compute_stats(csv_data[component+'_closed'], csv_data[csv_column])
        
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

            # Plot scatter plot            
            ax = axes[row, index]            
            # Filter data for the current dataset and index
            filtered_data = df_allPairs[df_allPairs['method'] == targetMethod]#filtered_df[(filtered_df['Dataset'] == dataset) & (filtered_df['component'] == component)]           
            # filtered_data.to_csv(outPath+dataset+'_'+component+'.csv', index=False)

            # Scatter plot
            ax.scatter(filtered_data['Reference'], filtered_data['Value'], alpha=0.5)  
            # sns.histplot(
            #     data=filtered_data,
            #     x='Reference',  
            #     y='Value', 
            #     cmap = 'viridis',
            #     # cmap = 'rainbow',
            #     # hue='basin', 
            #     # hue='method', 
            #     ax=ax
            # )          
            ax.axline((0, 0), slope=1, color='black', linestyle='--')
            # draw a y=0 line
            ax.axhline(y=0, color='black', linestyle='--')
            # Set titles and labels
            if row == 0:
                ax.set_title(dataset)
            if index == 0:
                ax.set_ylabel(component)
            if row == len(['P', 'E', 'R', 'S']) - 1:
                ax.set_xlabel('Reference Data')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()