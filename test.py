# import numpy as np
# import pandas as pd

# # Create a NumPy array
# numpy_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(numpy_array)

# # Convert the NumPy array to a pandas DataFrame with column names
# column_names = ['A', 'B', 'C']
# df = pd.DataFrame(data=numpy_array, columns=column_names)

# print(df)

# nparr = df.to_numpy()
# print(nparr)

import re

def get_first_single_digit_number(s):
    # Search for the first occurrence of a single digit in the string
    match = re.search(r'\d', s)
    if match:
        return match.group()
    else:
        return None  # or a default value if no digit is found

# Example usage
s = "PRC_1234_P_r"
first_single_digit = get_first_single_digit_number(s)
print(first_single_digit)  # Output: 1 (the first single digit)