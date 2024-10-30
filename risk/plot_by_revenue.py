# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
data = pd.read_csv('risk/risk_data.csv')

# Calculate the average risk by revenue
average_risk_by_revenue = data.groupby('revenue')['risk'].mean().reset_index()

# Create a single figure for the average risk
fig, ax = plt.subplots(figsize=(15, 10))  # Create one figure

# Plotting the average risk data
ax.scatter(average_risk_by_revenue['revenue'], average_risk_by_revenue['risk'], label='Average Risk', color='blue')  # Scatter plot for average risk

# Add polynomial regression line (2nd degree for curvature)
z = np.polyfit(average_risk_by_revenue['revenue'], average_risk_by_revenue['risk'], 2)  # Polynomial regression
p = np.poly1d(z)
ax.plot(average_risk_by_revenue['revenue'], p(average_risk_by_revenue['revenue']), linestyle='--', label='Curved Regression Line', color='red')  # Plot regression line

# Set y-axis to logarithmic scale
ax.set_yscale('log')  # Set y-axis to log scale

# Set titles and labels
ax.set_title('Average Risk vs Revenue')
ax.set_xlabel('Revenue')
ax.set_ylabel('Average Risk')
ax.grid()
ax.legend()  # Add legend

# Save the figure under the risk folder
plt.savefig('risk/average_risk_by_revenue.png')
plt.show()  # Display the plot