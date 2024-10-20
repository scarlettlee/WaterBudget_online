import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import mean_squared_error
from globVar import basin3Flag, find_pattern, get_file_name, compute_stats

# Define paths and folders
csv_folder = os.path.join(os.path.dirname(__file__), '', '')
if basin3Flag:  
    csv_files = find_pattern("*.csv", csv_folder+'3redistribution_outliers_mergeClosed_partTrue/')
else:
    csv_files = find_pattern("*.csv", csv_folder+'28redistribution_outliers_mergeClosed_partTrue/')

# new 28 basins or not
basins28 = not basin3Flag

# Merge all basin files into one dataframe for making plots
csv_data = pd.DataFrame()
for csv_file in csv_files:
    file_name = get_file_name(csv_file).split('_')[0]
    print("-------------------------------------Processing file:", file_name)

    # Read CSV data
    data = pd.read_csv(csv_file)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)    
    columns =data.columns
    # print(len(columns))

    # matchPR_####_P_r1
    r = re.compile(r".*_\d{4}_[P|E|R|S](?:_[12])?$") 
    filtered = list(filter(r.match, columns))
    # print("Filtered columns:", len(filtered))
    # # print all 3840 column name to overcome the limited output in the terminal
    # with open(csv_folder+'columns_list.txt', 'w') as file:
    #     for column in filtered:
    #         file.write(f"{column}\n")

    # match p/e/r/s_closed
    r = re.compile(".*_closed")
    filtered = filtered+list(filter(r.match, columns))+['date']
    # print("Filtered columns:", len(filtered))
    
    newData = data[filtered]
    newData['basin'] = file_name
    # print(len(newData.columns))

    ###############################
    # reframe the dataframe
    ###############################
    # Melt the DataFrame to long format
    melted_df = pd.melt(newData, id_vars=['basin','date','P_closed','E_closed','R_closed','S_closed'], var_name='variable', value_name='value')

    # Split the 'variable' column into 'A_type', 'B_type', 'C_type', 'D_type'
    melted_df[['method', 'combination', 'component', 'abnormal']] = melted_df['variable'].str.split('_', expand=True)
    # Drop the original 'variable' column
    melted_df.drop(columns=['variable'], inplace=True)
    # replace none in abnormal column with 0
    melted_df['abnormal'] = melted_df['abnormal'].fillna(0)
    # print(melted_df)

    csv_data = pd.concat([csv_data, melted_df], ignore_index=True)
# print(csv_data)
# save csv_data
# csv_data.to_csv(csv_folder+'vis/4.2-Basins.csv', index=False)
# csv_data[(csv_data['component']=='E')&(csv_data['value']<0)].to_csv(csv_folder+'vis/outliers.csv', index=False)

# Create a 4x3 grid of scatter plots
components = csv_data['component'].unique()
abnormal_values = csv_data['abnormal'].unique()
methods = csv_data['method'].unique()
basins = csv_data['basin'].unique()

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 20))

for i, component in enumerate(components):  # Iterate over each component
    for j, abnormal in enumerate(abnormal_values):  # Iterate over each abnormal value
        ax = axes[i, j]
        
        # Filter data for the specific abnormal value and component
        subset = csv_data[(csv_data['component'] == component) & (csv_data['abnormal'] == abnormal)]

        # # Create scatter plot
        # sns.scatterplot(
        #     data=subset,
        #     x='value',  # The "value" column represents the observed data
        #     y=f'{component}_closed', 
        #     # hue='basin', 
        #     hue='method', 
        #     ax=ax
        # )
        # Create heat scatter plot
        sns.histplot(
            data=subset,
            x='value',  # The "value" column represents the observed data
            y=f'{component}_closed', 
            hue='basin', 
            # hue='method', 
            ax=ax
        )
        # # Create heat scatter plot using kdeplot
        # sns.kdeplot(
        #     data=subset,
        #     x='value',  # The "value" column represents the observed data
        #     y=f'{component}_closed',
        #     hue='basin',  # Uncomment if you want to add distinction by 'basin'
        #     ax=ax,
        #     fill=True,   # Fills the plot
        #     cmap='viridis'  # Choose a colormap
        # )

        # add a y = x line not a diagonal line
        ax.axline((0, 0), slope=1, color='black', linestyle='--')
        
        # Calculate and display RMSE
        rmse_values = []
        for method in methods:
            method_subset = subset[subset['method'] == method]

            # Drop NaN values before RMSE calculation
            method_subset = method_subset.dropna(subset=[f'{component}_closed', 'value'])

            if not method_subset.empty:
                rmse = mean_squared_error(method_subset[f'{component}_closed'], method_subset['value'], squared=False)
                rmse_values.append(f"{method}: {rmse:.2f}")
        
        # # Add legend only to the last subplot in each row
        # if i != 0 or j != 0:
        #     ax.get_legend().remove()
        # else:
        #     ax.legend(loc='lower right', bbox_to_anchor=(1, 0))

        ax.set_title(f'{component} (Abnormal: {abnormal})')
        ax.text(0.05, 0.95, "\n".join(rmse_values), transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        ax.set_xlabel("")  # Label for x-axis indicating observed values
        ax.set_ylabel(f"{component}_closed")

# Adjust layout
plt.tight_layout()
plt.show()