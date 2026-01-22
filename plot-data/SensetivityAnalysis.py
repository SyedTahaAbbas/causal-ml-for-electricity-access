import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

# Prepare data (reuse your scaled X_train_scaled, T_train, Y_train)

hyperparameter_grid = {
    "min_samples_leaf": [5, 10, 20],
    "max_depth": [5, 10, None],
    "n_estimators": [50, 100, 200]
}

sensitivity_results = []

for min_leaf in hyperparameter_grid['min_samples_leaf']:
    for max_d in hyperparameter_grid['max_depth']:
        for n_est in hyperparameter_grid['n_estimators']:
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

            # Example: average estimated effect for 2022
            X_cf_scaled = scaler.transform(X_cf)  # test set covariates
            effects = cf_model.const_marginal_effect(X_cf_scaled)
            avg_effect = np.mean(effects)

            sensitivity_results.append({
                "min_samples_leaf": min_leaf,
                "max_depth": max_d,
                "n_estimators": n_est,
                "avg_effect": avg_effect
            })

sensitivity_df = pd.DataFrame(sensitivity_results)
print(sensitivity_df)


from sklearn.utils import resample

n_bootstraps = 50
bootstrap_effects = []

for i in range(n_bootstraps):
    X_res, T_res, Y_res = resample(X_train_scaled, T_train, Y_train, random_state=42+i)
    cf_model = CausalForestDML(
        model_t=LassoCV(cv=3),
        model_y=LassoCV(cv=3),
        n_estimators=100,
        min_samples_leaf=5,
        max_depth=10,
        discrete_treatment=False,
        random_state=123+i
    )
    cf_model.fit(Y_res, T_res, X=X_res, W=None)
    effects = cf_model.const_marginal_effect(X_cf_scaled)
    bootstrap_effects.append(np.mean(effects))

bootstrap_effects = np.array(bootstrap_effects)
print(f"Bootstrap mean: {bootstrap_effects.mean():.4f}, std: {bootstrap_effects.std():.4f}")

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Hyperparameter Sensitivity Plot
# -----------------------------
# sensitivity_df: from the hyperparameter analysis above
plt.figure(figsize=(12,6))
sns.lineplot(
    data=sensitivity_df,
    x='n_estimators',
    y='avg_effect',
    hue='min_samples_leaf',
    style='max_depth',
    markers=True,
    dashes=False,
    palette='tab10'
)
plt.title('Causal Forest Sensitivity: Avg Effect vs n_estimators', fontsize=16)
plt.xlabel('Number of Trees (n_estimators)', fontsize=14)
plt.ylabel('Average Estimated Effect (Δ Access)', fontsize=14)
plt.legend(title='min_samples_leaf / max_depth', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('sensitivity_hyperparameters.png', dpi=300)
plt.show()

# -----------------------------
# 2. Bootstrap Sensitivity Plot
# -----------------------------
plt.figure(figsize=(10,6))
sns.histplot(bootstrap_effects, bins=15, kde=True, color='#1f77b4')
plt.axvline(x=np.mean(bootstrap_effects), color='red', linestyle='--', linewidth=2, label='Mean Effect')
plt.title('Causal Forest Sensitivity: Bootstrap Effects', fontsize=16)
plt.xlabel('Average Estimated Effect (Δ Access)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('sensitivity_bootstrap.png', dpi=300)
plt.show()
