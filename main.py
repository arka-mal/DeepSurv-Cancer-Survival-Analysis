# ==========================================================
# DeepSurv vs Classical Survival Models (METABRIC Dataset)
# Fully Corrected & Reviewer-Safe Implementation
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import cumulative_dynamic_auc


# ----------------------------------------------------------
# Plot Saving Setup (Timestamped Output Folder)
# ----------------------------------------------------------

import os
from datetime import datetime

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
PLOT_DIR = f"results_{RUN_ID}"

os.makedirs(PLOT_DIR, exist_ok=True)

print(f"Saving plots to folder: {PLOT_DIR}")

def save_and_show(fig_name):
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fig_name}.png", dpi=300)
    plt.show()
    plt.close()
RESULTS_FILE = f"{PLOT_DIR}/results_summary.txt"

def log(*args):

    msg = " ".join(str(arg) for arg in args)
    print(msg)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
                
log("===== SURVIVAL ANALYSIS RESULTS =====")


# ----------------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------------

df = pd.read_csv("METABRIC.csv")

# ----------------------------------------------------------
# 2. Clean Survival Columns (CRITICAL FIX)
# ----------------------------------------------------------

# ----------------------------------------------------------
# Robust Survival Column Cleaning (FIXES EMPTY DATA ISSUE)
# ----------------------------------------------------------

# Normalize survival status text
df["Overall Survival Status"] = (
    df["Overall Survival Status"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# Keep only valid entries
df = df[df["Overall Survival Status"].isin(["DECEASED", "LIVING"])]

# Encode event
event = df["Overall Survival Status"].map(
    {"DECEASED": 1, "LIVING": 0}
).values.astype(int)

# Survival time
# ----------------------------------------------------------
# Fix zero or negative survival times (SVM-safe)
# ----------------------------------------------------------

time = df["Overall Survival (Months)"].astype(float).values

# Replace 0 or negative times with small epsilon
time[time <= 0] = 0.1

print("Min survival time:", time.min())


log(df["Overall Survival Status"].value_counts())
log("Number of samples:", len(df))


# ----------------------------------------------------------
# 3. Feature Matrix
# ----------------------------------------------------------

drop_cols = [
    "Patient ID",
    "Overall Survival (Months)",
    "Overall Survival Status",
    "Patient's Vital Status"
]

X = df.drop(columns=drop_cols)

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# ----------------------------------------------------------
# Remove rare categorical dummy variables (Cox-safe)
# ----------------------------------------------------------

min_positive_fraction = 0.01  # at least 1% of samples must be 1

keep_cols = []
for col in X.columns:
    frac = X[col].mean()   # since dummy variables are 0/1
    if frac > min_positive_fraction and frac < (1 - min_positive_fraction):
        keep_cols.append(col)

log(f"Keeping {len(keep_cols)} / {X.shape[1]} features after rarity filtering")

X = X[keep_cols]


# Handle missing values safely
X = X.fillna(X.median(numeric_only=True))

feature_names = X.columns.tolist()

# ----------------------------------------------------------
# 4. Train-Test Split (Stratified)
# ----------------------------------------------------------

X_train, X_test, t_train, t_test, e_train, e_test = train_test_split(
    X, time, event,
    test_size=0.3,
    stratify=event,
    random_state=42
)

# ----------------------------------------------------------
# 4A. Structured Survival Targets (REQUIRED FOR RSF & SVM)
# ----------------------------------------------------------

from sksurv.util import Surv

y_train_struct = Surv.from_arrays(
    event=e_train.astype(bool),
    time=t_train
)

y_test_struct = Surv.from_arrays(
    event=e_test.astype(bool),
    time=t_test
)



# ----------------------------------------------------------
# 5. Feature Scaling (BASE FEATURES)
# ----------------------------------------------------------

scaler = StandardScaler()
X_train_base = scaler.fit_transform(X_train)
X_test_base = scaler.transform(X_test)

# These are for Cox, RSF, SVM
X_train = X_train_base
X_test = X_test_base


# ----------------------------------------------------------
# 5A. Autoencoder Embeddings (DeepSurv ONLY)
# ----------------------------------------------------------

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def embed(self, x):
        return self.encoder(x)

ae = AutoEncoder(X_train_base.shape[1], latent_dim=16)
optimizer_ae = torch.optim.Adam(ae.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

Xt_ae = torch.tensor(X_train_base, dtype=torch.float32)

ae.train()
for _ in range(200):
    optimizer_ae.zero_grad()
    loss = loss_fn(ae(Xt_ae), Xt_ae)
    loss.backward()
    optimizer_ae.step()

ae.eval()
with torch.no_grad():
    Z_train = ae.embed(torch.tensor(X_train_base, dtype=torch.float32)).numpy()
    Z_test = ae.embed(torch.tensor(X_test_base, dtype=torch.float32)).numpy()

# FINAL DeepSurv features
X_train_ds = np.hstack([X_train_base, Z_train])
X_test_ds = np.hstack([X_test_base, Z_test])




# ----------------------------------------------------------
# 6. Cox Proportional Hazards Model
# ----------------------------------------------------------

cox_df = pd.DataFrame(X_train, columns=feature_names)
cox_df["time"] = t_train
cox_df["event"] = e_train

cox = CoxPHFitter(
    penalizer=1.0,
    l1_ratio=0.5
)


cox.fit(cox_df, duration_col="time", event_col="event")

cox_risk = cox.predict_partial_hazard(
    pd.DataFrame(X_test, columns=feature_names)
).values.flatten()

cox_cindex = concordance_index(t_test, -cox_risk, e_test)



# ----------------------------------------------------------
# 7. Random Survival Forest
# ----------------------------------------------------------

rsf = RandomSurvivalForest(
    n_estimators=200,
    min_samples_split=10,
    random_state=42
)
rsf.fit(X_train, y_train_struct)

rsf_risk = rsf.predict(X_test)
rsf_cindex = concordance_index(t_test, -rsf_risk, e_test)




# ----------------------------------------------------------
# 8. Survival SVM
# ----------------------------------------------------------

svm = FastSurvivalSVM(max_iter=1000, random_state=42)
svm.fit(X_train, y_train_struct)

svm_risk = svm.predict(X_test)
svm_cindex = concordance_index(t_test, -svm_risk, e_test)





# ----------------------------------------------------------
# 9. DeepSurv
# ----------------------------------------------------------

class DeepSurv(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def cox_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    return -torch.sum((risk - log_cumsum) * event) / (event.sum() + 1e-8)

model = DeepSurv(X_train_ds.shape[1])
Xt = torch.tensor(X_train_ds, dtype=torch.float32)

# ----------------------------------------------------------
# DeepSurv ENSEMBLE (REPLACES TRAINING + EVAL ONLY)
# ----------------------------------------------------------

ds_risks = []

Xt = torch.tensor(X_train_ds, dtype=torch.float32)
tt = torch.tensor(t_train, dtype=torch.float32)
et = torch.tensor(e_train, dtype=torch.float32)

print("DeepSurv input shape:", X_train_ds.shape)
print("Xt shape:", Xt.shape)

for seed in [0, 1, 2, 3, 4]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DeepSurv(X_train_ds.shape[1])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-3
    )

    model.train()
    best_loss = np.inf
    patience = 50
    counter = 0

    for epoch in range(800):
        optimizer.zero_grad()
        loss = cox_loss(model(Xt).squeeze(), tt, et)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break

    model.eval()
    with torch.no_grad():
        risk = model(
            torch.tensor(X_test_ds, dtype=torch.float32)
        ).numpy().flatten()

    ds_risks.append(risk)

# Average ensemble predictions
ds_risk = np.mean(ds_risks, axis=0)

ds_cindex = concordance_index(t_test, -ds_risk, e_test)






# ----------------------------------------------------------
# 9A. Paired Bootstrap Significance Test (DeepSurv vs RSF)
# ----------------------------------------------------------

def bootstrap_cindex(risk1, risk2, time, event, n_boot=1000):
    diffs = []
    n = len(time)

    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        c1 = concordance_index(time[idx], -risk1[idx], event[idx])
        c2 = concordance_index(time[idx], -risk2[idx], event[idx])
        diffs.append(c1 - c2)

    return np.mean(diffs), np.percentile(diffs, [2.5, 97.5])

mean_diff, ci = bootstrap_cindex(
    ds_risk,
    rsf_risk,
    t_test,
    e_test,
    n_boot=1000
)

log("\n===== BOOTSTRAP SIGNIFICANCE TEST =====")
log(f"DeepSurv-RSF mean ΔC-index: {mean_diff:.4f}")
log(f"95% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}]")

if ci[0] > 0:
    log("DeepSurv is statistically superior to RSF")
elif ci[1] < 0:
    log("RSF is statistically superior to DeepSurv")
else:
    log("No statistically significant difference")


# ----------------------------------------------------------
# 10. Integrated Brier Score (IBS) Comparison
# ----------------------------------------------------------

from sksurv.metrics import integrated_brier_score

# Use clinically meaningful time range
time_grid = np.linspace(
    np.percentile(t_test, 10),
    np.percentile(t_test, 90),
    100
)


# CoxPH Survival Functions (CORRECT)


cox_surv_df = cox.predict_survival_function(
    pd.DataFrame(X_test, columns=feature_names),
    times=time_grid
)

# Convert DataFrame → NumPy array (n_samples × n_times)
cox_surv = cox_surv_df.values.T


# RSF survival functions
rsf_surv_fns = rsf.predict_survival_function(X_test, return_array=False)
rsf_surv = np.vstack([
    fn(time_grid) for fn in rsf_surv_fns
])

# Survival SVM calibrated via CoxPH
svm_df = pd.DataFrame({"svm_risk": svm_risk})
svm_df["time"] = t_test
svm_df["event"] = e_test

svm_cal = CoxPHFitter()
svm_cal.fit(svm_df, duration_col="time", event_col="event")

svm_surv_df = svm_cal.predict_survival_function(
    pd.DataFrame({"svm_risk": svm_risk}),
    times=time_grid
)

# Convert DataFrame → NumPy array (n_samples × n_times)
svm_surv = svm_surv_df.values.T

# DeepSurv calibrated via CoxPH
ds_df = pd.DataFrame({"ds_risk": ds_risk})
ds_df["time"] = t_test
ds_df["event"] = e_test

ds_cal = CoxPHFitter()
ds_cal.fit(ds_df, duration_col="time", event_col="event")

ds_surv_df = ds_cal.predict_survival_function(
    pd.DataFrame({"ds_risk": ds_risk}),
    times=time_grid
)

# Convert DataFrame → NumPy array (n_samples × n_times)
ds_surv = ds_surv_df.values.T

# ----------------------------------------------------------
# IBS Calculation
# ----------------------------------------------------------

ibs_cox = integrated_brier_score(
    y_train_struct, y_test_struct, cox_surv, time_grid
)

ibs_rsf = integrated_brier_score(
    y_train_struct, y_test_struct, rsf_surv, time_grid
)

ibs_svm = integrated_brier_score(
    y_train_struct, y_test_struct, svm_surv, time_grid
)

ibs_ds = integrated_brier_score(
    y_train_struct, y_test_struct, ds_surv, time_grid
)

log("\n===== INTEGRATED BRIER SCORE (LOWER IS BETTER) =====")
log("CoxPH IBS:", f"{ibs_cox:.3f}")
log("RSF IBS:", f"{ibs_rsf:.3f}")
log("Survival-SVM IBS:", f"{ibs_svm:.3f}")
log("DeepSurv IBS:", f"{ibs_ds:.3f}")

models = ["CoxPH", "RSF", "Survival-SVM", "DeepSurv"]
ibs_scores = [ibs_cox, ibs_rsf, ibs_svm, ibs_ds]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, ibs_scores)

# Highlight DeepSurv (best expected)
bars[-1].set_edgecolor("black")
bars[-1].set_linewidth(3)

plt.ylabel("Integrated Brier Score (IBS)")
plt.title("Integrated Brier Score Comparison (Lower is Better)")
plt.ylim(0, max(ibs_scores) * 1.15)

# Annotate values on bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom"
    )


save_and_show("ibs-score")




# ----------------------------------------------------------
# 11. C-index Comparison Plot
# ----------------------------------------------------------

models = ["CoxPH", "RSF", "Survival-SVM", "DeepSurv"]
cindex_scores = [cox_cindex, rsf_cindex, svm_cindex, ds_cindex]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, cindex_scores)
bars[-1].set_edgecolor("black")
bars[-1].set_linewidth(3)

plt.ylabel("C-index")
plt.title("C-index Comparision")
plt.ylim(0.55, 0.80)

# Annotate values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom"
    )

save_and_show("cindex_comparison")

# ----------------------------------------------------------
# C-index Summary
# ----------------------------------------------------------

log("\n===== C-INDEX PERFORMANCE =====")
log(f"CoxPH C-index:        {cox_cindex:.3f}")
log(f"RSF C-index:          {rsf_cindex:.3f}")
log(f"Survival-SVM C-index: {svm_cindex:.3f}")
log(f"DeepSurv C-index:     {ds_cindex:.3f}")

# ----------------------------------------------------------
# 12. Time-Dependent AUC
# ----------------------------------------------------------

times = np.percentile(t_test, [25, 50, 75])

auc_cox, _ = cumulative_dynamic_auc(
    y_train_struct, y_test_struct, cox_risk, times
)
auc_rsf, _ = cumulative_dynamic_auc(
    y_train_struct, y_test_struct, rsf_risk, times
)
auc_svm, _ = cumulative_dynamic_auc(
    y_train_struct, y_test_struct, svm_risk, times
)
auc_ds, _ = cumulative_dynamic_auc(
    y_train_struct, y_test_struct, ds_risk, times
)

plt.figure()
plt.plot(times, auc_cox, label="CoxPH")
plt.plot(times, auc_rsf, label="RSF")
plt.plot(times, auc_svm, label="Survival-SVM")
plt.plot(times, auc_ds, label="DeepSurv", linewidth=3)

plt.xlabel("Time (Months)")
plt.ylabel("AUC")
plt.title("Time-Dependent AUC Comparison")
plt.legend()
save_and_show("time_dependent_auc")

# ----------------------------------------------------------
# 13. Kaplan–Meier Risk Stratification (DeepSurv)
# ----------------------------------------------------------

df_test = pd.DataFrame(X_test, columns=feature_names)
df_test["time"] = t_test
df_test["event"] = e_test
df_test["risk"] = ds_risk

df_test["risk_group"] = pd.qcut(
    df_test["risk"], 3,
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

kmf = KaplanMeierFitter()

plt.figure(figsize=(7, 6))
for group in df_test["risk_group"].unique():
    mask = df_test["risk_group"] == group
    kmf.fit(
        df_test.loc[mask, "time"],
        df_test.loc[mask, "event"],
        label=group
    )
    kmf.plot_survival_function()

plt.title("Kaplan–Meier Curves by DeepSurv Risk Groups")
plt.xlabel("Time (Months)")
plt.ylabel("Survival Probability")
save_and_show("km_risk_stratification_deepsurv")

