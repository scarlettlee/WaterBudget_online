import numpy as np
import pandas as pd

# Create a NumPy array
numpy_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(numpy_array)

# Convert the NumPy array to a pandas DataFrame with column names
column_names = ['A', 'B', 'C']
df = pd.DataFrame(data=numpy_array, columns=column_names)

print(df)

nparr = df.to_numpy()
print(nparr)