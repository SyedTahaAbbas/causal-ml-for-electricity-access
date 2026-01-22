import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create directory for bar charts
os.makedirs('aid_response_updated/bar_charts_population', exist_ok=True)

# Compute predicted Δ-access in number of people per country
popu_2022 = df_2022['popu_total'].values  # total population per country (millions)
predicted_people = popu_2022 * (predicted_delta_access / 100)  # already in millions

# Combine into DataFrame
df_population = pd.DataFrame({
    'country': df_2022['country'].values,
    'PredictedOptimalAccess': predicted_people,
    'PredictedOptimalAid': predicted_optimal_aid  # optional: show aid
})

# Sort by predicted access
df_sorted = df_population.sort_values('PredictedOptimalAccess', ascending=False)

# Top 10 countries
top10 = df_sorted.head(10)
plt.figure(figsize=(10, 6))
bars = plt.barh(top10['country'][::-1], top10['PredictedOptimalAccess'][::-1], color='green')
plt.xlabel('Increase in People with Electricity Access (Millions)', fontsize=14)
plt.title('Top 10 Countries by Predicted Increase in Electricity Access', fontsize=16)

# Add value labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.02*width, bar.get_y() + bar.get_height()/2,
             f"{width:.2f}M", ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.grid(axis='x', alpha=0.2)
plt.savefig('aid_response_updated/bar_charts_population/top10_predicted_access.png', dpi=300)
plt.close()

# Bottom 10 countries
bottom10 = df_sorted.tail(10)
plt.figure(figsize=(10, 6))
bars = plt.barh(bottom10['country'][::-1], bottom10['PredictedOptimalAccess'][::-1], color='red')
plt.xlabel('Increase in People with Electricity Access (Millions)', fontsize=14)
plt.title('Bottom 10 Countries by Predicted Increase in Electricity Access', fontsize=16)

# Add value labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.02*width, bar.get_y() + bar.get_height()/2,
             f"{width:.2f}M", ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.grid(axis='x', alpha=0.2)
plt.savefig('aid_response_updated/bar_charts_population/bottom10_predicted_access.png', dpi=300)
plt.close()
