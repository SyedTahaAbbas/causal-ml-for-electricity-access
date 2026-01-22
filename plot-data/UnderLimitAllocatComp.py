import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt

# --------------------------
# 1. Load data
# --------------------------
df = pd.read_csv('electricity_data.csv')
year = 2022
df_2022 = df[df['year'] == year].copy()

features = ['fdi', 'gdp', 'gdp_per_capita', 'inflation', 'life_expectancy',
            'popu_growth', 'popu_total', 'rural_popu_growth', 'urban_popu_growth',
            'unemployment', 'rural_electricity_access', 'urban_electricity_access']
treatment = 'electricity_aid'
outcome = 'electricity_access'

# --------------------------
# 2. Load and prepare training data
# --------------------------
train_df = pd.read_csv('Sim_data_train.csv')
X_train = train_df.iloc[:, 2:].values
T_train = train_df.iloc[:, 1].values.reshape(-1,1)
Y_train = train_df.iloc[:, 0].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --------------------------
# 3. Fit Causal Forest
# --------------------------
cf_model = CausalForestDML(
    model_t=LassoCV(cv=3),
    model_y=LassoCV(cv=3),
    n_estimators=200,
    min_samples_leaf=5,
    max_depth=10,
    discrete_treatment=False,
    random_state=123
)
cf_model.fit(Y_train, T_train, X=X_train_scaled, W=None)

# --------------------------
# 4. Prepare 2022 data for predictions
# --------------------------
X_2022 = df_2022[features].values
T_2022 = df_2022[treatment].values
Y_2022 = df_2022[outcome].values
X_2022_scaled = scaler.transform(X_2022)

# --------------------------
# 5. Compute baseline and effects
# --------------------------
cate_effects = cf_model.const_marginal_effect(X_2022_scaled).flatten()
baseline_access = Y_2022 - cate_effects * T_2022
observed_delta_access = Y_2022 - baseline_access

# --------------------------
# 6. Constrained optimal aid allocation
# --------------------------
total_aid_budget = T_2022.sum()  # Same as actual total aid
max_aid_per_country = 2 * T_2022  # Per-country cap

# Sort countries by treatment effect (descending)
sorted_indices = np.argsort(cate_effects)[::-1]
optimized_aid = np.zeros_like(T_2022)
remaining_budget = total_aid_budget

# Allocate aid greedily to most effective countries first
for idx in sorted_indices:
    if remaining_budget <= 0:
        break
    # Determine allocation for this country
    aid_allocation = min(max_aid_per_country[idx], remaining_budget)
    optimized_aid[idx] = aid_allocation
    remaining_budget -= aid_allocation

# Calculate predicted outcomes
optimized_delta = cate_effects * optimized_aid
observed_delta = cate_effects * T_2022

# --------------------------
# 7. Results comparison
# --------------------------
print(f"Actual Total Aid (2022): ${T_2022.sum():.2f}M")
print(f"Optimized Total Aid (2022): ${optimized_aid.sum():.2f}M")
print(f"Actual Avg Δ Access: {observed_delta.mean():.4f}%")
print(f"Optimized Avg Δ Access: {optimized_delta.mean():.4f}%")

# --------------------------
# 8. Visualization
# --------------------------
plt.figure(figsize=(10, 6))
labels = ['Actual Allocation', 'Optimized Allocation']
values = [observed_delta.mean(), optimized_delta.mean()]

bars = plt.bar(labels, values, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
plt.ylabel('Average Increase in Electricity Access (%)', fontsize=12)
plt.title('Electricity Access Improvement: Actual vs Optimized Aid Allocation', fontsize=14)
plt.ylim(0, max(values)*1.2)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.0002, f'{height:.4f}%', 
             ha='center', va='bottom', fontsize=10)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('aid_response_updated/constrained_optimization_comparison.png', dpi=300)
plt.show()