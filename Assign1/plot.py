import matplotlib.pyplot as plt

# Sample data (replace this with your actual data)
data = {
  ('512', '5'): [0.125784, 0.125389, 0.119695],
  ('512', '9'): [0.250755, 0.204841, 0.194142],
  ('2048', '5'): [1.761244, 1.260418, 1.158923],
  ('2048', '9'): [2.192494, 1.872660, 2.089848],
}

# Extracting x and y data for plotting
x_data = list(data.keys())
y_data = list(data.values())

# Plotting boxplots
plt.figure(figsize=(10, 6))
plt.boxplot(y_data, labels=x_data)

# Adding labels and title
plt.xlabel('(n, stencil)')
plt.ylabel('Time (seconds)')
plt.title('Time for each data size per stencil configuration')

plt.grid(True)
plt.tight_layout()
plt.show()
