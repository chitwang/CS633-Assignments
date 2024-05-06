import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data.csv')

# Uncomment to filter out rows where n == 8192
# df_filtered = df[df['n'] != 8192]
# df = df_filtered

# Group by 'n' and 'np' and get the values of 'Time without leader' and 'Time with leader' for each group
grouped_df = df.groupby(['n', 'np']).agg({'0': list, '1': list}).reset_index()

# Create a dictionary with the desired format
data = {}
for index, row in grouped_df.iterrows():
  key = ('0', row['np'], row['n'])
  data[key] = row['0']
  
  key = ('1', row['np'], row['n'])
  data[key] = row['1']

# # Extracting x and y data for plotting
x_data = list(data.keys())
y_data = list(data.values())

# Plotting boxplots
plt.figure(figsize=(10, 6))
plt.boxplot(y_data, labels=x_data)

# Adding labels and title
plt.xlabel('(leader, np, n)')
plt.ylabel('Time (seconds)')
plt.title('Time for each data size per configuration')

plt.grid(True)
plt.tight_layout()
plt.show()
