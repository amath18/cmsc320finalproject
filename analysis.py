"""
End-to-end analysis for the CMSC320 Spring 2026 final tutorial.

Loads the BRFSS 2015 diabetes binary health indicators, runs preprocessing,
EDA hypothesis tests, three classification models with imbalance handling,
and writes:

  docs/figures/*.png   - publication-quality charts for the site
  docs/model.json      - logistic regression coefficients + scaler params
                         for the in-browser risk calculator
  docs/metrics.json    - dataset stats and model metrics quoted in prose

Run from the project root:
    python3 analysis.py
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).parent
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Editorial plotting style
# ---------------------------------------------------------------------------
PALETTE = {
    "bg": "#faf7f0",
    "panel": "#ffffff",
    "ink": "#1a1a1a",
    "muted": "#6b6b6b",
    "rule": "#d8d2c4",
    "primary": "#9b1c1c",      # claret
    "secondary": "#0e7c7b",    # teal
    "tertiary": "#b58900",     # mustard
    "blue": "#1f4e79",
    "neutral": "#cbc5b6",
}

mpl.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor": PALETTE["bg"],
    "savefig.facecolor": PALETTE["bg"],
    "axes.edgecolor": PALETTE["rule"],
    "axes.labelcolor": PALETTE["ink"],
    "axes.titlesize": 14,
    "axes.titleweight": "semibold",
    "axes.titlecolor": PALETTE["ink"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": PALETTE["rule"],
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "axes.axisbelow": True,
    "xtick.color": PALETTE["muted"],
    "ytick.color": PALETTE["muted"],
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.labelsize": 11,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "font.family": ["DejaVu Sans"],
    "font.size": 11,
    "figure.dpi": 110,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
})

SERIF = {"family": "DejaVu Serif", "weight": "semibold"}


def save_fig(fig, name):
    out = FIG_DIR / name
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 1. Load + preprocess
# ---------------------------------------------------------------------------
print("Loading dataset...")
csv_path = ROOT / "diabetes_binary_health_indicators_BRFSS2015.csv"
df = pd.read_csv(csv_path)

df.columns = df.columns.str.strip()
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()
for col in df.columns:
    df[col] = df[col].astype(int)

n_rows, n_cols = df.shape
n_pos = int((df["Diabetes_binary"] == 1).sum())
n_neg = int((df["Diabetes_binary"] == 0).sum())
prevalence = n_pos / n_rows
print(f"  shape: {df.shape}, positives: {n_pos}, negatives: {n_neg}, prevalence: {prevalence:.3f}")


# ---------------------------------------------------------------------------
# 2. EDA: target balance, correlations, three hypothesis tests
# ---------------------------------------------------------------------------
print("\nFigure 1: target class balance")
fig, ax = plt.subplots(figsize=(7.5, 4.2))
counts = df["Diabetes_binary"].value_counts().sort_index()
bars = ax.bar(
    ["No diabetes", "Diabetes / pre-diabetes"],
    counts.values,
    color=[PALETTE["neutral"], PALETTE["primary"]],
    edgecolor=PALETTE["ink"],
    linewidth=0.8,
    width=0.55,
)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 3500, f"{val:,}",
            ha="center", va="bottom", color=PALETTE["ink"], fontsize=11)
ax.set_ylabel("Respondents (BRFSS 2015)")
ax.set_title("Class imbalance in the BRFSS diabetes indicator", loc="left", **SERIF)
ax.set_ylim(0, counts.max() * 1.15)
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
save_fig(fig, "01_class_balance.png")

# Correlations with target
print("Figure 2: correlations with target")
corrs = df.corr(numeric_only=True)["Diabetes_binary"].drop("Diabetes_binary").sort_values()
fig, ax = plt.subplots(figsize=(8, 7))
colors = [PALETTE["primary"] if v > 0 else PALETTE["blue"] for v in corrs.values]
ax.barh(corrs.index, corrs.values, color=colors, edgecolor="white", linewidth=0.6)
ax.axvline(0, color=PALETTE["ink"], linewidth=0.8)
ax.set_xlabel("Pearson correlation with Diabetes_binary")
ax.set_title("Which features move with diabetes status?", loc="left", **SERIF)
legend_elems = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=PALETTE["primary"], markersize=10, label="positive"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=PALETTE["blue"], markersize=10, label="negative"),
]
ax.legend(handles=legend_elems, loc="lower right")
save_fig(fig, "02_correlations.png")

# Hypothesis test 1: BMI t-test
print("Hypothesis test 1: BMI by diabetes status (Welch's one-tailed t)")
no_dm_bmi = df.loc[df["Diabetes_binary"] == 0, "BMI"]
dm_bmi = df.loc[df["Diabetes_binary"] == 1, "BMI"]
t1, p1_two = stats.ttest_ind(dm_bmi, no_dm_bmi, equal_var=False)
p1_one = p1_two / 2 if t1 > 0 else 1 - p1_two / 2
print(f"  mean(no DM)={no_dm_bmi.mean():.2f}, mean(DM)={dm_bmi.mean():.2f}, t={t1:.2f}, p_one={p1_one:.2e}")

fig, ax = plt.subplots(figsize=(7, 4.5))
parts = ax.violinplot(
    [no_dm_bmi.sample(min(len(no_dm_bmi), 8000), random_state=0),
     dm_bmi.sample(min(len(dm_bmi), 8000), random_state=0)],
    showmeans=False, showmedians=True, widths=0.85,
)
for i, body in enumerate(parts["bodies"]):
    body.set_facecolor([PALETTE["neutral"], PALETTE["primary"]][i])
    body.set_alpha(0.65)
    body.set_edgecolor(PALETTE["ink"])
    body.set_linewidth(0.6)
parts["cmedians"].set_color(PALETTE["ink"])
parts["cmedians"].set_linewidth(1.5)
parts["cbars"].set_color(PALETTE["muted"])
parts["cmins"].set_color(PALETTE["muted"])
parts["cmaxes"].set_color(PALETTE["muted"])
ax.set_xticks([1, 2])
ax.set_xticklabels(["No diabetes", "Diabetes / pre-diabetes"])
ax.set_ylabel("BMI")
ax.set_ylim(10, 60)
ax.set_title("BMI distribution by diabetes status", loc="left", **SERIF)
ax.text(0.02, 0.96,
        f"Welch t = {t1:.2f}   p < 1e-100   (one-tailed)",
        transform=ax.transAxes, va="top", color=PALETTE["muted"], fontsize=10)
save_fig(fig, "03_bmi_violin.png")

# Hypothesis test 2: prevalence by sex (z-test of two proportions via t-test on indicator)
print("Hypothesis test 2: prevalence by sex (two-tailed t)")
female = df.loc[df["Sex"] == 0, "Diabetes_binary"]
male = df.loc[df["Sex"] == 1, "Diabetes_binary"]
t2, p2 = stats.ttest_ind(male, female, equal_var=False)
print(f"  rate(F)={female.mean():.4f}, rate(M)={male.mean():.4f}, t={t2:.2f}, p={p2:.2e}")

fig, ax = plt.subplots(figsize=(7, 4.2))
rates = [female.mean(), male.mean()]
bars = ax.bar(["Female", "Male"], rates, color=[PALETTE["secondary"], PALETTE["primary"]],
              edgecolor=PALETTE["ink"], linewidth=0.8, width=0.55)
for bar, r in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2, r + 0.003, f"{r*100:.2f}%",
            ha="center", va="bottom", color=PALETTE["ink"], fontsize=11)
ax.set_ylabel("Diabetes prevalence")
ax.set_ylim(0, max(rates) * 1.25)
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
ax.set_title("Self-reported diabetes prevalence by sex", loc="left", **SERIF)
ax.text(0.02, 0.96, f"Welch t = {t2:.2f}    p = {p2:.2e}",
        transform=ax.transAxes, va="top", color=PALETTE["muted"], fontsize=10)
save_fig(fig, "04_sex_prevalence.png")

# Hypothesis test 3: BMI by age group (ANOVA)
print("Hypothesis test 3: BMI by broad age group (one-way ANOVA)")
def age_bucket(code):
    if code <= 4:
        return "Young (18-39)"
    elif code <= 9:
        return "Middle (40-64)"
    else:
        return "Older (65+)"
df["AgeGroup"] = df["Age"].apply(age_bucket)
groups = [df.loc[df["AgeGroup"] == g, "BMI"] for g in ["Young (18-39)", "Middle (40-64)", "Older (65+)"]]
f_stat, p3 = stats.f_oneway(*groups)
print(f"  means: {[g.mean() for g in groups]}, F={f_stat:.2f}, p={p3:.2e}")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
parts = ax.violinplot([g.sample(min(len(g), 8000), random_state=0) for g in groups],
                       showmeans=False, showmedians=True, widths=0.85)
group_colors = [PALETTE["secondary"], PALETTE["tertiary"], PALETTE["primary"]]
for i, body in enumerate(parts["bodies"]):
    body.set_facecolor(group_colors[i])
    body.set_alpha(0.6)
    body.set_edgecolor(PALETTE["ink"])
    body.set_linewidth(0.6)
parts["cmedians"].set_color(PALETTE["ink"])
parts["cmedians"].set_linewidth(1.5)
parts["cbars"].set_color(PALETTE["muted"])
parts["cmins"].set_color(PALETTE["muted"])
parts["cmaxes"].set_color(PALETTE["muted"])
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["Young (18-39)", "Middle (40-64)", "Older (65+)"])
ax.set_ylabel("BMI")
ax.set_ylim(10, 60)
ax.set_title("BMI distribution across age cohorts", loc="left", **SERIF)
ax.text(0.02, 0.96, f"ANOVA F = {f_stat:.2f}   p < 1e-100",
        transform=ax.transAxes, va="top", color=PALETTE["muted"], fontsize=10)
save_fig(fig, "05_age_bmi_violin.png")


# ---------------------------------------------------------------------------
# 3. Train/test split + three classification models
# ---------------------------------------------------------------------------
print("\nTraining classifiers...")
feature_cols = [c for c in df.columns if c not in ("Diabetes_binary", "AgeGroup")]
X = df[feature_cols].astype(float).values
y = df["Diabetes_binary"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# Model 1: logistic regression with class_weight balanced
lr = LogisticRegression(
    class_weight="balanced", max_iter=2000, solver="lbfgs", n_jobs=-1,
)
lr.fit(X_train_s, y_train)
lr_proba = lr.predict_proba(X_test_s)[:, 1]
lr_auc = roc_auc_score(y_test, lr_proba)
lr_ap = average_precision_score(y_test, lr_proba)
print(f"  LR    ROC-AUC={lr_auc:.4f}  PR-AUC={lr_ap:.4f}")

# Model 2: random forest (subsample for speed)
rf = RandomForestClassifier(
    n_estimators=200, max_depth=14, min_samples_leaf=20,
    class_weight="balanced", n_jobs=-1, random_state=42,
)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_proba)
rf_ap = average_precision_score(y_test, rf_proba)
print(f"  RF    ROC-AUC={rf_auc:.4f}  PR-AUC={rf_ap:.4f}")

# Model 3: gradient boosting (sklearn HistGB would be faster; using GBC for clarity)
from sklearn.ensemble import HistGradientBoostingClassifier
# Use sample_weight to handle imbalance (HistGB doesn't support class_weight directly)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
sample_weight = np.where(y_train == 1, pos_weight, 1.0)
gb = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.07, max_depth=7, random_state=42,
)
gb.fit(X_train, y_train, sample_weight=sample_weight)
gb_proba = gb.predict_proba(X_test)[:, 1]
gb_auc = roc_auc_score(y_test, gb_proba)
gb_ap = average_precision_score(y_test, gb_proba)
print(f"  GBT   ROC-AUC={gb_auc:.4f}  PR-AUC={gb_ap:.4f}")


# ---------------------------------------------------------------------------
# 4. ROC + PR curves
# ---------------------------------------------------------------------------
print("\nFigure 6: ROC curves")
fig, ax = plt.subplots(figsize=(6.5, 5.2))
for proba, name, color, auc in [
    (lr_proba, "Logistic regression", PALETTE["blue"], lr_auc),
    (rf_proba, "Random forest", PALETTE["secondary"], rf_auc),
    (gb_proba, "Gradient boosting", PALETTE["primary"], gb_auc),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"{name}  (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color=PALETTE["muted"], linestyle="--", linewidth=1, label="chance")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC: discriminating diabetes from survey features", loc="left", **SERIF)
ax.legend(loc="lower right")
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
save_fig(fig, "06_roc_curves.png")

print("Figure 7: precision-recall curves")
fig, ax = plt.subplots(figsize=(6.5, 5.2))
for proba, name, color, ap in [
    (lr_proba, "Logistic regression", PALETTE["blue"], lr_ap),
    (rf_proba, "Random forest", PALETTE["secondary"], rf_ap),
    (gb_proba, "Gradient boosting", PALETTE["primary"], gb_ap),
]:
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ax.plot(recall, precision, color=color, linewidth=2.2, label=f"{name}  (AP = {ap:.3f})")
ax.axhline(prevalence, color=PALETTE["muted"], linestyle="--", linewidth=1,
           label=f"baseline rate = {prevalence:.2f}")
ax.set_xlabel("Recall (sensitivity)")
ax.set_ylabel("Precision")
ax.set_title("Precision-recall: the imbalance-honest view", loc="left", **SERIF)
ax.legend(loc="upper right")
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(0, 1)
save_fig(fig, "07_pr_curves.png")


# ---------------------------------------------------------------------------
# 5. Confusion matrix at threshold = 0.5 for the best model (gradient boosting)
# ---------------------------------------------------------------------------
print("Figure 8: confusion matrix")
y_pred_gb = (gb_proba >= 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred_gb)
fig, ax = plt.subplots(figsize=(5.5, 4.8))
im = ax.imshow(cm, cmap="Reds", aspect="auto")
for (i, j), v in np.ndenumerate(cm):
    text_color = "white" if v > cm.max() * 0.55 else PALETTE["ink"]
    ax.text(j, i, f"{v:,}", ha="center", va="center", color=text_color, fontsize=14)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred: no DM", "Pred: DM"])
ax.set_yticklabels(["True: no DM", "True: DM"])
ax.set_title("Gradient boosting confusion (threshold 0.5)", loc="left", **SERIF)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)
save_fig(fig, "08_confusion.png")

# Compute headline classification metrics at threshold 0.5
acc_gb = accuracy_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)
sens_gb = cm[1, 1] / (cm[1, 0] + cm[1, 1])  # recall on positive class
spec_gb = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # specificity
print(f"  GBT@0.5  acc={acc_gb:.4f}  f1={f1_gb:.4f}  sens={sens_gb:.4f}  spec={spec_gb:.4f}")


# ---------------------------------------------------------------------------
# 6. Feature importance from gradient boosting (permutation-light: use built-in)
# ---------------------------------------------------------------------------
print("Figure 9: feature importance (gradient boosting + LR magnitudes)")
# HistGB does not expose feature_importances_; use LR coefficient magnitude on standardized X
# and RF feature importances side by side for richness.
lr_coef = lr.coef_[0]
rf_imp = rf.feature_importances_

imp_df = pd.DataFrame({
    "feature": feature_cols,
    "lr_coef": lr_coef,
    "rf_importance": rf_imp,
}).sort_values("rf_importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 7.5))
ax.barh(imp_df["feature"], imp_df["rf_importance"],
        color=PALETTE["primary"], edgecolor="white", linewidth=0.5)
ax.set_xlabel("Random forest feature importance")
ax.set_title("What signals carry the most weight?", loc="left", **SERIF)
save_fig(fig, "09_feature_importance.png")

# A second figure: standardized LR coefficients (signed) — interpretable direction
print("Figure 10: signed standardized LR coefficients")
coef_df = pd.DataFrame({"feature": feature_cols, "coef": lr_coef}).sort_values("coef")
fig, ax = plt.subplots(figsize=(8, 7.5))
colors = [PALETTE["primary"] if c > 0 else PALETTE["blue"] for c in coef_df["coef"].values]
ax.barh(coef_df["feature"], coef_df["coef"], color=colors, edgecolor="white", linewidth=0.5)
ax.axvline(0, color=PALETTE["ink"], linewidth=0.8)
ax.set_xlabel("Standardized logistic regression coefficient")
ax.set_title("Direction and strength of each feature", loc="left", **SERIF)
save_fig(fig, "10_lr_coefficients.png")


# ---------------------------------------------------------------------------
# 7. Calibration plot for gradient boosting (best model)
# ---------------------------------------------------------------------------
print("Figure 11: calibration curve (gradient boosting)")
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_test, gb_proba, n_bins=10, strategy="quantile")
fig, ax = plt.subplots(figsize=(6.2, 5))
ax.plot([0, 1], [0, 1], color=PALETTE["muted"], linestyle="--", linewidth=1, label="perfect calibration")
ax.plot(prob_pred, prob_true, color=PALETTE["primary"], linewidth=2.2, marker="o",
        markerfacecolor=PALETTE["primary"], markeredgecolor="white", markersize=8,
        label="gradient boosting")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed diabetes rate")
ax.set_title("Calibration: do scores mean what they say?", loc="left", **SERIF)
ax.legend(loc="upper left")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
save_fig(fig, "11_calibration.png")


# ---------------------------------------------------------------------------
# 8. Threshold-vs-cost curve
# ---------------------------------------------------------------------------
print("Figure 12: threshold sweep")
thresholds = np.linspace(0.05, 0.95, 91)
sens_arr, spec_arr, prec_arr = [], [], []
for t in thresholds:
    pred = (gb_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    sens_arr.append(tp / (tp + fn) if (tp + fn) else 0)
    spec_arr.append(tn / (tn + fp) if (tn + fp) else 0)
    prec_arr.append(tp / (tp + fp) if (tp + fp) else 0)
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(thresholds, sens_arr, color=PALETTE["primary"], linewidth=2, label="sensitivity (recall)")
ax.plot(thresholds, spec_arr, color=PALETTE["blue"], linewidth=2, label="specificity")
ax.plot(thresholds, prec_arr, color=PALETTE["secondary"], linewidth=2, label="precision")
ax.axvline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=1)
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Rate")
ax.set_title("Choosing a threshold trades off who you flag", loc="left", **SERIF)
ax.legend(loc="center right")
ax.set_xlim(0.05, 0.95)
ax.set_ylim(0, 1.02)
save_fig(fig, "12_threshold.png")


# ---------------------------------------------------------------------------
# 9. Export model + metrics for the site
# ---------------------------------------------------------------------------
print("\nExporting model + metrics JSON for the in-browser risk calculator...")

# For the calculator we use the LR model (cheap to run client-side).
model_export = {
    "model": "logistic_regression",
    "intercept": float(lr.intercept_[0]),
    "features": feature_cols,
    "coefficients": [float(c) for c in lr_coef],
    "scaler_mean": [float(m) for m in scaler.mean_],
    "scaler_scale": [float(s) for s in scaler.scale_],
    "training_prevalence": prevalence,
    "test_roc_auc": float(lr_auc),
    "notes": (
        "Standardize each input as (x - mean) / scale, then compute "
        "p = sigmoid(intercept + dot(coefficients, x_std))."
    ),
}
(ROOT / "docs" / "model.json").write_text(json.dumps(model_export, indent=2))
print(f"  wrote docs/model.json ({len(feature_cols)} features)")

metrics_export = {
    "dataset": {
        "rows": int(n_rows),
        "columns": int(n_cols),
        "features_used": len(feature_cols),
        "positives": n_pos,
        "negatives": n_neg,
        "prevalence": prevalence,
    },
    "hypothesis_tests": {
        "bmi_by_diabetes": {
            "test": "Welch's one-tailed t-test",
            "mean_no_dm": float(no_dm_bmi.mean()),
            "mean_dm": float(dm_bmi.mean()),
            "t": float(t1),
            "p_one_tailed": float(p1_one),
        },
        "prevalence_by_sex": {
            "test": "Welch's two-tailed t-test on indicator",
            "rate_female": float(female.mean()),
            "rate_male": float(male.mean()),
            "t": float(t2),
            "p": float(p2),
        },
        "bmi_by_age_group": {
            "test": "one-way ANOVA",
            "means": [float(g.mean()) for g in groups],
            "F": float(f_stat),
            "p": float(p3),
        },
    },
    "models": {
        "logistic_regression": {"roc_auc": float(lr_auc), "pr_auc": float(lr_ap)},
        "random_forest": {"roc_auc": float(rf_auc), "pr_auc": float(rf_ap)},
        "gradient_boosting": {
            "roc_auc": float(gb_auc),
            "pr_auc": float(gb_ap),
            "accuracy_at_0.5": float(acc_gb),
            "f1_at_0.5": float(f1_gb),
            "sensitivity_at_0.5": float(sens_gb),
            "specificity_at_0.5": float(spec_gb),
        },
    },
    "top_correlates": {k: float(v) for k, v in corrs.sort_values(key=abs, ascending=False).head(8).items()},
}
(ROOT / "docs" / "metrics.json").write_text(json.dumps(metrics_export, indent=2))
print(f"  wrote docs/metrics.json")
print("\nDone.")
