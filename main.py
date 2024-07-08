import pandas as pd
import matplotlib.pyplot as plt

# Import Excel file
excel_file = r'U:\test_ppg\ppg1.xlsx'
data = pd.read_excel(excel_file, header=1)

# Data from 2nd column
column_2 = data.iloc[:, 1]

# Plot
plt.plot(column_2)
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('Smartwatch PPG')
plt.show()
