# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('risk/risk_data.csv')

# Group by country and year, then calculate the average risk
average_risk = data.groupby(['country', 'year'])['risk'].mean().reset_index()

# Get unique countries
countries = average_risk['country'].unique()

# Plotting all countries' data in a single subplot
for country in countries:
    country_data = average_risk[average_risk['country'] == country]
    plt.plot(country_data['year'], country_data['risk'], marker='o', label=country)

# Set title and labels for the single plot
plt.title('Average Risk Over Time for All Countries')
plt.xlabel('Year')
plt.ylabel('Average Risk')
plt.grid()
plt.xticks(average_risk['year'].unique())
plt.legend()  # Add a legend to differentiate countries

# Adjust layout
plt.tight_layout()

# Save the figure under the risk folder
plt.savefig('risk/average_risk_by_country.png')  # Save the plot as a PNG file
plt.show()  # Display the plot
