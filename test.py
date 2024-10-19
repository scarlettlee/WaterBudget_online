import pandas as pd

# Example DataFrame
# Make sure 'date' is your datetime column and you have columns like 'AAA_BBBB_C_D'
data = {
    'date': pd.date_range(start='2021-01-01', periods=5, freq='D'),
    'basin': ['A', 'A', 'A', 'B', 'C'],
    'AAA_12_C_D': [1, 2, 3, 4, 5],
    'ZZZ_24_X_W_1': [6, 7, 8, 9, 10]
}

df = pd.DataFrame(data)
print(df)

# Melt the DataFrame to long format
melted_df = pd.melt(df, id_vars=['date','basin'], var_name='variable', value_name='value')

# Split the 'variable' column into 'A_type', 'B_type', 'C_type', 'D_type'
melted_df[['A_type', 'B_type', 'C_type', 'D_type','E_type']] = melted_df['variable'].str.split('_', expand=True)
print(melted_df)

# Drop the original 'variable' column
melted_df.drop(columns=['variable'], inplace=True)

# Reorder columns if needed
melted_df = melted_df[['date', 'basin','A_type', 'B_type', 'C_type', 'D_type', 'value']]

print(melted_df)
