# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('risk/risk_data.csv')

# Group by sector and year, then calculate the average risk
average_risk_by_sector = data.groupby(['sector', 'year'])['risk'].mean().reset_index()

# Get unique sectors
sectors = average_risk_by_sector['sector'].unique()

# Create a single figure for all sectors
fig, ax = plt.subplots(figsize=(15, 10))  # Create one figure

# Plotting each sector's data in the same graph
for sector in sectors:
    sector_data = average_risk_by_sector[average_risk_by_sector['sector'] == sector]
    ax.plot(sector_data['year'], sector_data['risk'], marker='o', label=sector)  # Add label for legend

# Set titles and labels
ax.set_title('Average Risk Over Time for All Sectors')
ax.set_xlabel('Year')
ax.set_ylabel('Average Risk')
ax.grid()
ax.set_xticks(sector_data['year'])
ax.legend()  # Add legend to differentiate sectors

# Save the figure under the risk folder
plt.savefig('risk/average_risk_by_sector_combined.png')
plt.show()  # Display the plot