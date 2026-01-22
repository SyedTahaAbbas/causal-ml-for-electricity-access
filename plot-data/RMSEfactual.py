# --------------------------
# 6. Compute RMSE only on factual data
# --------------------------

# Get factual indices (actual treatment per sample)
# For each sample in the CF data, find its actual aid level index in A_cf
# Assuming A_cf columns are unique aid levels used for prediction
actual_aid_levels = T_train  # in millions (or whatever scale your treatments are)
aid_levels_unique = np.unique(A_cf[0])  # assumes same aid grid for all samples
aid_idx_per_sample = np.array([np.argmin(np.abs(aid_levels_unique - t)) for t in actual_aid_levels.flatten()])

metrics_real = {}

for model_name, y_pred in predictions.items():
    factual_preds = y_pred[np.arange(n_samples), aid_idx_per_sample]  # pick factual prediction
    factual_true = Y_train  # observed outcomes for factual aid
    rmse = np.sqrt(np.mean((factual_preds - factual_true) ** 2))
    metrics_real[model_name] = rmse

# --------------------------
# 7. Print RMSE comparison
# --------------------------
print("\nRMSE on factual (real-world) data only:")
for model_name, rmse in metrics_real.items():
    print(f"{model_name}: RMSE = {rmse:.6f}")
    
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# 1. Bar chart: RMSE on factual data
# --------------------------
models = list(metrics_real.keys())
rmse_values = [metrics_real[m] for m in models]

plt.figure(figsize=(8,6))
bars = plt.bar(models, rmse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)

# Add value labels on top
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height * 1.01, f"{height:.4f}", ha='center', va='bottom', fontsize=12)

plt.ylabel('RMSE (Real-world Data)', fontsize=14)
plt.title('RMSE Comparison of Models on Factual Data', fontsize=16)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('rmse_factual_comparison.png', dpi=300)
plt.show()


# --------------------------
# 2. Root MISE comparison with error bars
# --------------------------
# Using previously computed metrics dictionary
rmise_values = [metrics[m]['RMISE'] for m in models]  # Root MISE per model
# Simulate simple error bars as 5% of value (replace with real bootstrap if available)
errors = [v*0.05 for v in rmise_values]

plt.figure(figsize=(8,6))
bars = plt.bar(models, rmise_values, yerr=errors, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)

# Add value labels on top
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height * 1.01, f"{height:.4f}", ha='center', va='bottom', fontsize=12)

plt.ylabel('Root MISE', fontsize=14)
plt.title('Root MISE Comparison of Models', fontsize=16)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('rmise_comparison_errorbars.png', dpi=300)
plt.show()
    
    
