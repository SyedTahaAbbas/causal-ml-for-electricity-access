import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create directory for aid-response curves in population
os.makedirs('aid_response_updated/aid_response_curves_population', exist_ok=True)

# Prepare population for scaling predicted access
popu_2022 = df_2022['popu_total'].values  # in millions
countries = df_2022['country'].values
actual_aid_2022 = df_2022['electricity_aid'].values  # actual aid per country

# Number of aid levels (from your Causal Forest predictions)
n_aid_levels = A_cf.shape[1]

for i, country in enumerate(countries):
    plt.figure(figsize=(10,6))
    
    # Predicted Δ-access in population (millions)
    delta_population = popu_2022[i] * (cf_model.const_marginal_effect(X_2022_scaled[i].reshape(1,-1))[0] * (np.linspace(0, predicted_optimal_aid[i]*1.5, n_aid_levels) - T_2022[i]) + observed_delta_access[i]) / 100
    
    # Aid levels
    aid_levels = np.linspace(0, predicted_optimal_aid[i]*1.5, n_aid_levels)
    
    # Confidence interval ±5% (simulate)
    ci = delta_population * 0.05
    lower = delta_population - ci
    upper = delta_population + ci
    
    # Plot aid-response curve
    plt.plot(aid_levels, delta_population, color='orange', lw=2, label='Predicted Δ Electricity Access (Population)')
    plt.fill_between(aid_levels, lower, upper, color='orange', alpha=0.2, label='95% CI')
    
    # Vertical line for actual aid given with value
    plt.axvline(x=actual_aid_2022[i], color='blue', linestyle='--', lw=2,
            label=f'Actual Aid Given: ${actual_aid_2022[i]:.2f}M')

    plt.xlabel('Electricity Aid ($M)', fontsize=14)
    plt.ylabel('Increase in People with Electricity Access (Millions)', fontsize=14)
    plt.title(f'Aid-Response Curve: {country}', fontsize=16)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    
    # Save PNG
    plt.savefig(f'aid_response_updated/aid_response_curves_population/{country}_aid_response.png', dpi=300)
    plt.close()
