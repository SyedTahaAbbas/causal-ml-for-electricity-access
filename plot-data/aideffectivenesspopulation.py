import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create directory
os.makedirs('aid_response_updated/dot_plots_population', exist_ok=True)

# Compute predicted Δ-access in number of people per country
popu_2022 = df_2022['popu_total'].values  # total population per country (millions)
predicted_people = popu_2022 * (predicted_delta_access / 100)

# Combine into DataFrame
df_population = pd.DataFrame({
    'country': df_2022['country'].values,
    'PredictedOptimal': predicted_people
})

# Set group size for plotting multiple countries
group_size = 10
num_groups = int(np.ceil(len(df_population) / group_size))

for i in range(num_groups):
    group_df = df_population.iloc[i*group_size:(i+1)*group_size]
    plt.figure(figsize=(16, 6))
    
    x = np.arange(len(group_df))
    width = 0.6
    
    # Plot predicted only
    bars = plt.bar(x, group_df['PredictedOptimal'], width, color='#ff7f0e', label='Predicted Optimal')
    
    # Add value labels on top of each bar
    for j, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height * 1.02, f"{height:.2f}M",
                 ha='center', va='bottom', fontsize=10)
    
    plt.xticks(x, group_df['country'], rotation=0, ha='center')
    plt.ylabel('Increase in People with Electricity Access (Millions)', fontsize=14)
    plt.title(f'Predicted Increase in Electricity Access per Country (Group {i+1})', fontsize=16)
    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    
    # Save high-quality PNG
    plt.savefig(f'aid_response_updated/dot_plots_population/predicted_access_group_{i+1}.png', dpi=300)
    plt.close()
