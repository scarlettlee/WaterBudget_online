# # import numpy as np
# # import pandas as pd

# # # Create a NumPy array
# # numpy_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# # print(numpy_array)

# # # Convert the NumPy array to a pandas DataFrame with column names
# # column_names = ['A', 'B', 'C']
# # df = pd.DataFrame(data=numpy_array, columns=column_names)

# # print(df)

# # nparr = df.to_numpy()
# # print(nparr)

# import re

# def get_first_single_digit_number(s):
#     # Search for the first occurrence of a single digit in the string
#     match = re.search(r'\d', s)
#     if match:
#         return match.group()
#     else:
#         return None  # or a default value if no digit is found

# # Example usage
# s = "PRC_1234_P_r"
# first_single_digit = get_first_single_digit_number(s)
# print(first_single_digit)  # Output: 1 (the first single digit)

import pandas as pd

# Example DataFrame
data = pd.DataFrame({
    'column1': [1, -2, 3, 0, None, 5, -1, None, 8],
    'column2': [-1, None, 0, 2, 3, None, 4, 5, 6]
})

# List of column names to be considered
filtered = ['column1', 'column2']

# Concatenate the specified columns into a single series, ignoring nulls
concatenated_values = pd.concat([data[column].dropna() for column in filtered])

# Count the total number of non-null values
total_count_non_null = len(concatenated_values)

# Count the number of values greater than 0
count_greater_than_zero = (concatenated_values > 0).sum()

# Calculate the percentage
percentage_greater_than_zero = (count_greater_than_zero / total_count_non_null) * 100 if total_count_non_null > 0 else 0

print(count_greater_than_zero, total_count_non_null)

print(f"Percentage of values greater than 0 across all specified columns: {percentage_greater_than_zero:.2f}%")


