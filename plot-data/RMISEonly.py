import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV

# --------------------------
# 1. Load training and test data
# --------------------------
train_df = pd.read_csv("Sim_data_train.csv")
data_cf = np.load("data_cf.npz")
A_cf = data_cf['A_cf']       # shape: (n_samples, n_aid_levels)
Y_cf = data_cf['Y_cf']       # true counterfactual electricity access
X_cf = data_cf['X_cf']       # covariates

n_samples, n_aid_levels = A_cf.shape

# --------------------------
# 2. Prepare training data
# --------------------------
X_train = train_df.iloc[:, 2:].values
T_train = train_df.iloc[:, 1].values.reshape(-1,1)
Y_train = train_df.iloc[:, 0].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cf_scaled = scaler.transform(X_cf)

# --------------------------
# 3. Train bootstrapped Causal Forest models
# --------------------------
n_bootstraps = 10  # reduce if needed for speed
bootstrap_models = []

for i in range(n_bootstraps):
    X_res, T_res, Y_res = resample(X_train_scaled, T_train, Y_train, random_state=123+i)
    model = CausalForestDML(
        model_t=LassoCV(cv=3),
        model_y=LassoCV(cv=3),
        n_estimators=100,
        min_samples_leaf=5,
        max_depth=10,
        discrete_treatment=False,
        random_state=123+i
    )
    model.fit(Y_res, T_res, X=X_res, W=None)
    bootstrap_models.append(model)

# --------------------------
# 4. Predict Δ-access for each test aid level (average over bootstraps)
# --------------------------
cf_preds = np.zeros((n_samples, n_aid_levels))
all_preds = np.zeros((n_bootstraps, n_samples, n_aid_levels))

for b, model in enumerate(bootstrap_models):
    marginal_effects = model.const_marginal_effect(X_cf_scaled).flatten()
    for i in range(n_aid_levels):
        all_preds[b, :, i] = marginal_effects * (A_cf[:, i] - T_train.mean()) + Y_train.mean()

# Average over bootstraps
cf_preds = all_preds.mean(axis=0)

# --------------------------
# 5. Load/Generate GPS, DRNet, SCIGAN predictions
# --------------------------
# Replace with your actual predictions
gps_pred = np.random.rand(n_samples, n_aid_levels)
drnet_pred = np.random.rand(n_samples, n_aid_levels)
scigan_pred = np.random.rand(n_samples, n_aid_levels)

predictions = {
    "GPS": gps_pred,
    "DRNet": drnet_pred,
    "SCIGAN": scigan_pred,
    "CausalForest": cf_preds
}

# --------------------------
# 6. Compute MISE, RMSE, and Root MISE
# --------------------------
def compute_metrics(y_true, y_pred):
    squared_errors = (y_pred - y_true)**2
    integrated_errors = np.mean(squared_errors, axis=1)  # integrate per sample
    MISE = np.mean(integrated_errors)                     # mean over samples
    RMSE = np.sqrt(np.mean(squared_errors))              # global RMSE
    RMISE = np.sqrt(MISE)                                # root MISE
    return MISE, RMSE, RMISE

metrics = {}
for model_name, y_pred in predictions.items():
    MISE, RMSE, RMISE = compute_metrics(Y_cf, y_pred)
    metrics[model_name] = {"MISE": MISE, "RMSE": RMSE, "RMISE": RMISE}

# --------------------------
# 7. Print results
# --------------------------
for model_name, vals in metrics.items():
    print(f"{model_name}: MISE = {vals['MISE']:.6f}, RMSE = {vals['RMSE']:.6f}, RMISE = {vals['RMISE']:.6f}")
