!pip install econml --upgrade
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV

# Create directory for saving updated plots
os.makedirs('aid_response_updated', exist_ok=True)

# Load and prepare data
df = pd.read_csv('electricity_data.csv')
all_countries = df['country'].unique()

# Prepare features, treatment, and outcome
features = ['fdi', 'gdp', 'gdp_per_capita', 'inflation', 'life_expectancy',
            'popu_growth', 'popu_total', 'rural_popu_growth', 'urban_popu_growth',
            'unemployment', 'rural_electricity_access', 'urban_electricity_access']
treatment = 'electricity_aid'
outcome = 'electricity_access'

# Train data (exclude 2022)
train_df = df[df['year'] != 2022]
X_train = train_df[features]
T_train = train_df[[treatment]]
Y_train = train_df[outcome]

# Scale covariates
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Function to fit causal forest with your requested parameters
def fit_causal_forest(X, T, Y, seed):
    model = CausalForestDML(
        model_t=LassoCV(cv=3),
        model_y=LassoCV(cv=3),
        n_estimators=100,
        min_samples_leaf=5,
        max_depth=10,
        discrete_treatment=False,
        random_state=seed
    )
    model.fit(Y, T, X=X, W=None)
    return model

# Bootstrap models
n_bootstraps = 20
bootstrap_models = []
for i in range(n_bootstraps):
    X_res, T_res, Y_res = resample(X_train_scaled, T_train, Y_train, random_state=123+i)
    model = fit_causal_forest(X_res, T_res, Y_res, seed=123+i)
    bootstrap_models.append(model)

# Generate Δ-access predictions with 95% CI
def generate_causal_predictions(country, year=2022):
    country_df = df[(df['country'] == country) & (df['year'] == year)]
    if country_df.empty:
        return None

    actual_aid = country_df[treatment].values[0]
    actual_access = country_df[outcome].values[0]

    aid_min = df[treatment].min()
    aid_max = df[treatment].max()
    aid_range = np.linspace(aid_min, aid_max, 200)

    X_country_scaled = scaler.transform(country_df[features])

    # Collect predicted change in access from all bootstraps
    all_curves = []
    for model in bootstrap_models:
        effect = model.const_marginal_effect(X_country_scaled)[0]
        curve = effect * (aid_range - actual_aid)  # Δ-access from baseline
        all_curves.append(curve)

    all_curves = np.array(all_curves)
    mean_curve = np.mean(all_curves, axis=0)
    lower_ci = np.percentile(all_curves, 2.5, axis=0)
    upper_ci = np.percentile(all_curves, 97.5, axis=0)

    return aid_range, mean_curve, lower_ci, upper_ci, actual_aid

# Plotting function
def plot_country_curve(country, aid_range, mean_curve, lower_ci, upper_ci, actual_aid):
    plt.figure(figsize=(12, 8))
    plt.plot(aid_range, mean_curve, color='#1f77b4', linewidth=3,
             label='Predicted ΔAccess')
    plt.fill_between(aid_range, lower_ci, upper_ci, color='#1f77b4', alpha=0.2,
                     label='95% Confidence Interval')
    plt.axvline(x=actual_aid, color='#ff7f0e', linestyle='--', alpha=0.7, linewidth=2,
                label=f'Actual Aid: ${actual_aid:.1f}M')
    plt.title(f'Change in Electricity Access vs Aid: {country} (2022)', fontsize=16)
    plt.xlabel('Development Aid (USD Millions)', fontsize=14)
    plt.ylabel('Δ Electricity Access (%)', fontsize=14)
    plt.grid(alpha=0.2)
    plt.legend(loc='best', fontsize=12)

    country_clean = country.replace("/", "-").replace("\\", "-")
    plt.savefig(f'aid_response_updated/{country_clean}_aid_response.png',
                bbox_inches='tight', dpi=300)
    plt.close()

# Generate and plot for all countries
for country in all_countries:
    result = generate_causal_predictions(country)
    if result is not None:
        aid_range, mean_curve, lower_ci, upper_ci, actual_aid = result
        plot_country_curve(country, aid_range, mean_curve, lower_ci, upper_ci, actual_aid)

print(f"Generated causal ΔAccess curves with 95% CI for {len(all_countries)} countries")
print(f"Plots saved in 'aid_response_updated' directory")

