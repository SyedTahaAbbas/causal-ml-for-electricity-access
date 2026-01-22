import matplotlib.pyplot as plt
import os

# Create directory for saving bar charts
os.makedirs('model_comparison_charts', exist_ok=True)

# Extract model names and metric values
model_names = list(metrics.keys())
MISE_vals = [metrics[m]['MISE'] for m in model_names]
RMSE_vals = [metrics[m]['RMSE'] for m in model_names]
RMISE_vals = [metrics[m]['RMISE'] for m in model_names]

# --------------------------
# 1. Bar chart for MISE
# --------------------------
plt.figure(figsize=(8,6))
plt.bar(model_names, MISE_vals, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.ylabel('MISE', fontsize=14)
plt.title('Comparison of MISE Across Models', fontsize=16)
plt.grid(alpha=0.2, axis='y')
for i, v in enumerate(MISE_vals):
    plt.text(i, v + 0.0005, f"{v:.6f}", ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('model_comparison_charts/MISE_comparison.png', dpi=300)
plt.close()

# --------------------------
# 2. Bar chart for RMSE
# --------------------------
plt.figure(figsize=(8,6))
plt.bar(model_names, RMSE_vals, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.ylabel('RMSE', fontsize=14)
plt.title('Comparison of RMSE Across Models', fontsize=16)
plt.grid(alpha=0.2, axis='y')
for i, v in enumerate(RMSE_vals):
    plt.text(i, v + 0.0005, f"{v:.6f}", ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('model_comparison_charts/RMSE_comparison.png', dpi=300)
plt.close()

# --------------------------
# 3. Bar chart for Root MISE (RMISE)
# --------------------------
plt.figure(figsize=(8,6))
plt.bar(model_names, RMISE_vals, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.ylabel('Root MISE', fontsize=14)
plt.title('Comparison of Root MISE Across Models', fontsize=16)
plt.grid(alpha=0.2, axis='y')
for i, v in enumerate(RMISE_vals):
    plt.text(i, v + 0.0005, f"{v:.6f}", ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('model_comparison_charts/RMISE_comparison.png', dpi=300)
plt.close()

print("Bar charts saved in 'model_comparison_charts' folder.")
