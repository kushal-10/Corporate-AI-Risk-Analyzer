import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_by_country(model: str):
    # Read the data
    df = pd.read_csv(os.path.join('riskv2', model+'.csv'))

    # Calculate average risk by country and year
    risk_by_country = df.groupby(['country', 'year'])['risk'].mean().reset_index()

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=risk_by_country, x='year', y='risk', hue='country', marker='o')

    # Customize the plot
    plt.title('Average Risk by Country Over Time', fontsize=12, pad=15)
    plt.xlabel('Year')
    plt.ylabel('Average Risk')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join('plots', f'{model}_country.png'))
    plt.close()


if __name__ == '__main__':
    plot_by_country('gpt')
    plot_by_country('bert')