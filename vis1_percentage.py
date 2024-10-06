import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from globVar import basin3Flag, find_pattern, get_file_name 

sns.set_style("whitegrid")

# input file path
path = os.path.join(os.path.dirname(__file__), '', '')
if basin3Flag:
    filePath = path+"3redistribution_outliers_mergeClosed_partTrue/"
else:
    filePath = path+"28redistribution_outliers_mergeClosed_partTrue/"

test = False
pattern = "*.csv"
if test:
    # pattern = "1147010_bccTest.csv"
    pattern = "4127800.csv"
fileList = find_pattern(pattern, filePath)

# abnormal situations: outliers (r1), overlyAdjusted (r2)
abnormal = ['r1', 'r2']
# budget closure correction methods (BCCs) 
method = ['PR', 'CKF', 'MCL', 'MSD']
# water budget components
component = ['P', 'E', 'R', 'S']

# global dataframe for visualization
# with columns: basinID, OT/OA_P/E/R/S_PR/CKF/MCL/MSD
globDF = pd.DataFrame(columns=['basinID', 'abnormal', 'method','component', 'percentage'])

# traverse each basin files to 
# compute percentage of outliers for each component
for fl in fileList:
    fileName = get_file_name(fl)
    print("----------------------------------",fileName)
    data = pd.read_csv(fl)#.head(6)
    columns =data.columns
    # print(data)    

    # compute percentage of outliers: example MSD_4313_S_r1
    for ab in abnormal:
        for l in component:        #['P', 'E', 'R', 'S']
            for m in method:    #['PR', 'CKF', 'MCL', 'MSD']
                r = re.compile(m+"_\d{4}_"+l+"_"+ab+"$") # PR_####_P_r1
                filtered = list(filter(r.match, columns))
                # print(filtered)
                
                # Concatenate the specified columns into a single series, ignoring nulls
                concatenated_values = pd.concat([data[column].dropna() for column in filtered])
                # Count the total number of non-null values
                total_count_non_null = len(concatenated_values)
                # Count the number of values greater than 0
                count_greater_than_zero = (concatenated_values > 0).sum()  
                # Calculate the percentage
                percentage = (count_greater_than_zero / total_count_non_null) * 100 if total_count_non_null > 0 else 0

                # Define the new row as a Series
                basin = pd.Series({
                    'basinID': fileName,
                    'abnormal': ab,
                    'method': m,
                    'component': l,
                    'percentage': percentage
                })
                globDF = pd.concat([globDF,basin.to_frame().T],ignore_index=True)
# # print(globDF)
# globDF.to_csv(path+'vis/globDF28.csv', index=False)

###################################
# visualize 
###################################
# Initialize a grid of plots with a specific size (4 rows * 2 columns)
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 12), sharey=True)
# Generate plots for each component and abnormal situation
for i, compon in enumerate(component):
    for j, abnorm in enumerate(abnormal):
        # Filter data for current component and abnormal situation
        subset = globDF[(globDF['component'] == compon) & (globDF['abnormal'] == abnorm)]
        # # save subset to csv
        # subset.to_csv(path+'vis/'+component+abnormal+'_subset.csv', index=False)
        
        # Create bar plot for the subset
        ax = axes[i, j]
        sns.barplot(
            data=subset, 
            x='basinID', 
            y='percentage', 
            hue='method', 
            ax=ax,
            ci=None  # Disable confidence interval
        )
        
        # Set plot title and labels
        ax.set_title(f'Component: {compon}, Abnormal: {abnorm}')

         # Set x-label only for the last row
        if i == len(component) - 1:
            ax.set_xlabel('BasinID')
        else:
            ax.set_xlabel('')
        
        # Set y-label only for the first column
        if j == 0:
            ax.set_ylabel('Percentage')
        else:
            ax.set_ylabel('')

        # Add legend only to the last subplot in each row
        if i != 0 or j != 0:
            ax.get_legend().remove()

# Automatically adjust subplot parameters to give specified padding
plt.tight_layout()
plt.show()
