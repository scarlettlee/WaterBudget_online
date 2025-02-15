# visualize the synthetic data
import numpy as np
import matplotlib.pyplot as plt

# Generate noise samples
noise = np.random.normal(0, 1, 100)

# Set your desired upper limit
upper_limit = 2  # example upper limit

# Clip the noise to the upper limit
noise_clipped = np.clip(noise, None, upper_limit)

# Plot the clipped noise
plt.plot(noise)
plt.plot(noise_clipped)
plt.show()

# import pandas as pd

# s1 = pd.Series(['a', 'b'])
# s2 = pd.Series(['c', 'd'])
# s = pd.DataFrame([s1])
# print(s)
# s = pd.DataFrame([s1.T])
# print(s)

# c = pd.concat([s1, s2])
# d = pd.concat([s1, s2], ignore_index=True)
# print(c)
# print(d)

# e = pd.concat([s, s1, s2])
# print(e)


# # Example DataFrame
# # Make sure 'date' is your datetime column and you have columns like 'AAA_BBBB_C_D'
# data = {
#     'date': pd.date_range(start='2021-01-01', periods=5, freq='D'),
#     'basin': ['A', 'A', 'A', 'B', 'C'],
#     'AAA_12_C_D': [1, 2, 3, 4, 5],
#     'ZZZ_24_X_W_1': [6, 7, 8, 9, 10]
# }

# df = pd.DataFrame(data)
# print(df)

# # Melt the DataFrame to long format
# melted_df = pd.melt(df, id_vars=['date','basin'], var_name='variable', value_name='value')

# # Split the 'variable' column into 'A_type', 'B_type', 'C_type', 'D_type'
# melted_df[['A_type', 'B_type', 'C_type', 'D_type','E_type']] = melted_df['variable'].str.split('_', expand=True)
# print(melted_df)

# # Drop the original 'variable' column
# melted_df.drop(columns=['variable'], inplace=True)

# # Reorder columns if needed
# melted_df = melted_df[['date', 'basin','A_type', 'B_type', 'C_type', 'D_type', 'value']]

# print(melted_df)
