"""
Build final_tutorial.ipynb from the analysis pipeline, with markdown prose
matching the site, ready for nbconvert.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).parent

def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }

cells = []

cells.append(md("""\
# Predicting Diabetes from Survey Data Alone

**CMSC 320 Spring 2026 — Final Project Tutorial**
University of Maryland — Dr. Fardina Alam

**Authors:** Pranav Bykampadi · Ansh Mathur · Alvin · Addhyan Kumar

---

## Contributions

For each member, the rubric sections (A: Project idea, B: Dataset Curation,
C: Data Exploration & Statistics, D: ML Algorithm Design, E: ML Training &
Evaluation, F: Visualization & Conclusion, G: Final Tutorial Report Creation,
H: Additional) they led:

- **Pranav Bykampadi** — *TBD*
- **Ansh Mathur** — *TBD*
- **Alvin** — *TBD*
- **Addhyan Kumar** — *TBD*

> *Section letters and short narrative descriptions to be filled in by the
> team before Gradescope submission.*

A polished GitHub-Pages version of this tutorial lives at
**[amath18.github.io/cmsc320finalproject](https://amath18.github.io/cmsc320finalproject/)**;
the in-browser interactive risk calculator only runs there. This notebook is
the canonical source of code and prose.
"""))

# ---------------------------------------------------------------------------
cells.append(md("""\
## 1. Introduction

Roughly **one in ten Americans** has diabetes, and another one in three is
pre-diabetic. Most don't know it. The classical screen — a fasting blood
glucose or HbA1c draw — is cheap by hospital standards but still requires a
clinic, a phlebotomist, and a follow-up appointment that an uninsured patient
may never make.

The federal Behavioral Risk Factor Surveillance System (BRFSS) takes the
opposite approach: every year the CDC calls 400,000+ adults and asks the same
22 questions. The data are free, the questions take ninety seconds, and they
cost nothing to administer. The natural question is whether the answers carry
enough signal to flag someone at elevated diabetes risk before they ever set
foot in a clinic.

> **Headline question.** *Can we predict diabetes risk from cheap survey data
> alone, well enough to triage who should go get a real blood test?*

This tutorial walks through the full data-science pipeline expected by
CMSC 320: acquiring and curating the survey, exploring it with summary
statistics and three formal hypothesis tests, training and comparing three
classification algorithms (logistic regression, random forest, and histogram
gradient boosting), and converting the best model into an in-browser risk
calculator on the published GitHub Pages site.
"""))

cells.append(md("""\
## 2. Data curation

The dataset is `diabetes_binary_health_indicators_BRFSS2015.csv`, a cleaned
subset of the 2015 BRFSS survey released by Alex Teboul on
[Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
(original source:
[CDC BRFSS 2015](https://www.cdc.gov/brfss/annual_data/annual_2015.html)).
Each row is one respondent. The target column, `Diabetes_binary`, equals 1 if
the respondent has been told by a health professional that they have diabetes
or pre-diabetes and 0 otherwise. The remaining 21 columns are mostly binary
indicators (high blood pressure, smoker, physical activity), a handful of
ordinal scales (general health, age band, income, education), and one
continuous feature (BMI).
"""))

cells.append(code("""\
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, accuracy_score, f1_score,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Editorial-style plot palette used throughout this tutorial.
PALETTE = {
    "bg": "#faf7f0", "ink": "#1a1a1a", "muted": "#6b6b6b", "rule": "#d8d2c4",
    "primary": "#9b1c1c", "secondary": "#0e7c7b", "tertiary": "#b58900",
    "blue": "#1f4e79", "neutral": "#cbc5b6",
}
mpl.rcParams.update({
    "figure.facecolor": PALETTE["bg"], "axes.facecolor": PALETTE["bg"],
    "savefig.facecolor": PALETTE["bg"], "axes.edgecolor": PALETTE["rule"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": PALETTE["rule"], "grid.alpha": 0.7,
    "axes.axisbelow": True, "font.size": 11, "figure.dpi": 110,
})
"""))

cells.append(md("""\
### 2.1 Load and preprocess

Preprocessing is deliberately light because the file is already harmonized:
strip whitespace from column names, force every column through
`pd.to_numeric`, drop the (zero) rows that fail, then cast to `int` since
every variable is coded numerically by design. We also engineer one helper
column, `AgeGroup`, that buckets the 13-level `Age` ordinal into Young /
Middle / Older for the ANOVA below.
"""))

cells.append(code("""\
df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

df.columns = df.columns.str.strip()
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()
for col in df.columns:
    df[col] = df[col].astype(int)

def age_bucket(code):
    if code <= 4:  return "Young (18-39)"
    if code <= 9:  return "Middle (40-64)"
    return "Older (65+)"

df["AgeGroup"] = df["Age"].apply(age_bucket)

print(f"Shape after cleaning: {df.shape}")
print(f"Missing values: {int(df.isna().sum().sum())}")
df.head()
"""))

# ---------------------------------------------------------------------------
cells.append(md("""\
## 3. Exploratory data analysis

Before fitting any model we want a feel for the data: how lopsided is the
target, which features seem to move with diabetes, and are the differences
big enough to be real or could they be noise? Three formal hypothesis tests
answer the third question.

### 3.1 The target is heavily imbalanced
"""))

cells.append(code("""\
counts = df["Diabetes_binary"].value_counts().sort_index()
prevalence = counts[1] / counts.sum()
print(f"Negatives (no diabetes): {counts[0]:,}")
print(f"Positives (diabetes):    {counts[1]:,}")
print(f"Prevalence:              {prevalence:.3f}")

fig, ax = plt.subplots(figsize=(7.5, 4.2))
bars = ax.bar(["No diabetes", "Diabetes / pre-diabetes"], counts.values,
              color=[PALETTE["neutral"], PALETTE["primary"]],
              edgecolor=PALETTE["ink"], linewidth=0.8, width=0.55)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 3500, f"{val:,}",
            ha="center", va="bottom", fontsize=11, color=PALETTE["ink"])
ax.set_ylabel("Respondents (BRFSS 2015)")
ax.set_title("Class imbalance in the BRFSS diabetes indicator", loc="left")
ax.set_ylim(0, counts.max()*1.15)
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,_: f"{int(x/1000)}k"))
plt.show()
"""))

cells.append(md("""\
Of 253,680 respondents, only **13.9%** reported a diabetes diagnosis. Any
classifier that always predicts *no diabetes* will be 86% accurate and useless
— so we will track ROC-AUC, precision-recall, sensitivity, and specificity
rather than accuracy alone.

### 3.2 Which features even matter?
"""))

cells.append(code("""\
corrs = df.corr(numeric_only=True)["Diabetes_binary"].drop("Diabetes_binary").sort_values()

fig, ax = plt.subplots(figsize=(8, 7))
colors = [PALETTE["primary"] if v > 0 else PALETTE["blue"] for v in corrs.values]
ax.barh(corrs.index, corrs.values, color=colors, edgecolor="white", linewidth=0.6)
ax.axvline(0, color=PALETTE["ink"], linewidth=0.8)
ax.set_xlabel("Pearson correlation with Diabetes_binary")
ax.set_title("Which features move with diabetes status?", loc="left")
ax.legend(handles=[
    Line2D([0],[0], marker="s", color="w", markerfacecolor=PALETTE["primary"], markersize=10, label="positive"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor=PALETTE["blue"], markersize=10, label="negative"),
], loc="lower right")
plt.show()

print("Top 8 by |correlation|:")
print(corrs.reindex(corrs.abs().sort_values(ascending=False).index).head(8).round(3))
"""))

cells.append(md("""\
None of the individual correlations are large in absolute terms — that's
expected for a 21-feature health survey, where most signal lives in
interactions — but the directions are sensible. High blood pressure, BMI,
self-rated general health, age, and high cholesterol carry the most weight.
Physical activity, education, and income point the other way.

### 3.3 Hypothesis test 1 — does BMI separate the groups?

We test whether the *mean* BMI among diabetics is higher than among
non-diabetics. The classical tool here is Welch's *t*-test (no equal-variance
assumption), run as a one-sided test because the directional prediction
(higher BMI for diabetics) is the medically meaningful one.

- **H₀:** mean BMI(DM) ≤ mean BMI(no DM)
- **Hₐ:** mean BMI(DM) > mean BMI(no DM)
"""))

cells.append(code("""\
no_dm_bmi = df.loc[df["Diabetes_binary"] == 0, "BMI"]
dm_bmi    = df.loc[df["Diabetes_binary"] == 1, "BMI"]

t1, p1_two = stats.ttest_ind(dm_bmi, no_dm_bmi, equal_var=False)
p1_one = p1_two/2 if t1 > 0 else 1 - p1_two/2

print(f"mean BMI (no DM): {no_dm_bmi.mean():.2f}")
print(f"mean BMI (DM):    {dm_bmi.mean():.2f}")
print(f"Welch t = {t1:.2f}, one-tailed p = {p1_one:.2e}")

fig, ax = plt.subplots(figsize=(7, 4.5))
parts = ax.violinplot(
    [no_dm_bmi.sample(min(len(no_dm_bmi), 8000), random_state=0),
     dm_bmi.sample(min(len(dm_bmi), 8000), random_state=0)],
    showmeans=False, showmedians=True, widths=0.85,
)
for i, body in enumerate(parts["bodies"]):
    body.set_facecolor([PALETTE["neutral"], PALETTE["primary"]][i])
    body.set_alpha(0.65)
parts["cmedians"].set_color(PALETTE["ink"])
ax.set_xticks([1,2])
ax.set_xticklabels(["No diabetes", "Diabetes / pre-diabetes"])
ax.set_ylabel("BMI"); ax.set_ylim(10, 60)
ax.set_title("BMI distribution by diabetes status", loc="left")
plt.show()
"""))

cells.append(md("""\
The means are 27.8 (no DM) versus 31.9 (DM); *t* ≈ 100, *p* effectively 0.
We reject H₀ decisively. BMI is materially higher among diabetics.

### 3.4 Hypothesis test 2 — does prevalence differ by sex?

A two-sided Welch *t*-test on the diabetes indicator, grouped by sex.

- **H₀:** prevalence is the same for males and females
- **Hₐ:** prevalences differ
"""))

cells.append(code("""\
female = df.loc[df["Sex"] == 0, "Diabetes_binary"]
male   = df.loc[df["Sex"] == 1, "Diabetes_binary"]

t2, p2 = stats.ttest_ind(male, female, equal_var=False)
print(f"Female prevalence: {female.mean()*100:.2f}%")
print(f"Male prevalence:   {male.mean()*100:.2f}%")
print(f"Welch t = {t2:.2f}, two-tailed p = {p2:.2e}")

fig, ax = plt.subplots(figsize=(7, 4.2))
rates = [female.mean(), male.mean()]
bars = ax.bar(["Female", "Male"], rates,
              color=[PALETTE["secondary"], PALETTE["primary"]],
              edgecolor=PALETTE["ink"], linewidth=0.8, width=0.55)
for bar, r in zip(bars, rates):
    ax.text(bar.get_x()+bar.get_width()/2, r+0.003, f"{r*100:.2f}%",
            ha="center", va="bottom", color=PALETTE["ink"], fontsize=11)
ax.set_ylabel("Diabetes prevalence")
ax.set_ylim(0, max(rates)*1.25)
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,_: f"{x*100:.0f}%"))
ax.set_title("Self-reported diabetes prevalence by sex", loc="left")
plt.show()
"""))

cells.append(md("""\
*t* ≈ 15.7, *p* ≈ 1.3 × 10⁻⁵⁵. The 2-percentage-point gap (12.97% vs 15.16%)
is small in absolute terms but the sample is so large that the uncertainty
bands are tight — this is a real, replicable difference.

### 3.5 Hypothesis test 3 — does BMI vary across age groups?

A one-way ANOVA across our three engineered age buckets.

- **H₀:** mean BMI is identical in all three age cohorts
- **Hₐ:** at least one cohort's mean differs
"""))

cells.append(code("""\
groups = [df.loc[df["AgeGroup"]==g, "BMI"]
          for g in ["Young (18-39)", "Middle (40-64)", "Older (65+)"]]
f_stat, p3 = stats.f_oneway(*groups)
print(f"Means: {[round(g.mean(),2) for g in groups]}")
print(f"ANOVA F = {f_stat:.2f}, p = {p3:.2e}")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
parts = ax.violinplot([g.sample(min(len(g), 8000), random_state=0) for g in groups],
                       showmeans=False, showmedians=True, widths=0.85)
for body, c in zip(parts["bodies"], [PALETTE["secondary"], PALETTE["tertiary"], PALETTE["primary"]]):
    body.set_facecolor(c); body.set_alpha(0.6)
parts["cmedians"].set_color(PALETTE["ink"])
ax.set_xticks([1,2,3])
ax.set_xticklabels(["Young (18-39)", "Middle (40-64)", "Older (65+)"])
ax.set_ylabel("BMI"); ax.set_ylim(10, 60)
ax.set_title("BMI distribution across age cohorts", loc="left")
plt.show()
"""))

cells.append(md("""\
*F* ≈ 763, *p* effectively 0. The middle cohort (40–64) has the highest mean
BMI; both younger and older sit lower. The non-monotone shape — BMI rises
through middle age and then *falls* in the 65+ bucket, a pattern combining
survivorship bias and muscle-mass loss in older adults — is a useful nudge
that the age ↔ BMI ↔ diabetes relationship is non-linear, motivating the
tree-based models in §4.
"""))

# ---------------------------------------------------------------------------
cells.append(md("""\
## 4. Primary analysis: three classifiers

The exploration leaves us with a binary classification problem on a heavily
imbalanced target. Three things drive the modeling choices:

1. **Imbalance.** 6 negatives for every positive. We have to weight the loss,
   or the model will learn to predict *no diabetes* for everyone.
2. **Mixed feature types.** Continuous (BMI), ordinal (Age, GenHlth), binary
   (Smoker, HighBP). Tree-based models handle this natively; linear models
   need standardization.
3. **Non-linearity.** §3.5 already showed a U-shape in BMI by age. Whatever
   model we ship had better be able to bend.

We compare three models with progressively more flexibility:

- **Logistic regression** with `class_weight="balanced"` — interpretable
  baseline; coefficients on standardized features tell us direction and
  relative strength.
- **Random forest** (200 trees, max depth 14, balanced class weights) —
  captures non-linearity and interactions; gives a feature-importance
  ranking that doesn't rely on linearity.
- **Histogram gradient boosting** (300 iterations, learning rate 0.07) —
  the off-the-shelf tabular champion; we balance the loss with sample
  weights since `HistGradientBoostingClassifier` doesn't take
  `class_weight` directly.

Eighty-twenty stratified train/test split, fixed seed (42).
"""))

cells.append(code("""\
feature_cols = [c for c in df.columns if c not in ("Diabetes_binary", "AgeGroup")]
X = df[feature_cols].astype(float).values
y = df["Diabetes_binary"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# Logistic regression
lr = LogisticRegression(class_weight="balanced", max_iter=2000, n_jobs=-1)
lr.fit(X_train_s, y_train)
lr_proba = lr.predict_proba(X_test_s)[:, 1]

# Random forest
rf = RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=20,
                            class_weight="balanced", n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]

# Gradient boosting (sample-weight imbalance handling)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
sw = np.where(y_train == 1, pos_weight, 1.0)
gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.07,
                                     max_depth=7, random_state=42)
gb.fit(X_train, y_train, sample_weight=sw)
gb_proba = gb.predict_proba(X_test)[:, 1]

results = pd.DataFrame({
    "Model": ["Logistic regression", "Random forest", "Gradient boosting"],
    "ROC-AUC": [roc_auc_score(y_test, lr_proba),
                 roc_auc_score(y_test, rf_proba),
                 roc_auc_score(y_test, gb_proba)],
    "PR-AUC":  [average_precision_score(y_test, lr_proba),
                 average_precision_score(y_test, rf_proba),
                 average_precision_score(y_test, gb_proba)],
}).round(4)
results
"""))

cells.append(md("""\
### 4.1 ROC and precision-recall

ROC summarizes the sensitivity-vs-specificity trade-off; precision-recall is
the imbalance-honest companion (with positive prevalence at 13.9%, a random
classifier sits at PR ≈ 0.14, not 0.5).
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for proba, name, color in [(lr_proba, "Logistic regression", PALETTE["blue"]),
                            (rf_proba, "Random forest",       PALETTE["secondary"]),
                            (gb_proba, "Gradient boosting",   PALETTE["primary"])]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, color=color, linewidth=2.2,
            label=f"{name} (AUC = {roc_auc_score(y_test, proba):.3f})")
ax.plot([0,1], [0,1], color=PALETTE["muted"], linestyle="--", linewidth=1, label="chance")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC", loc="left"); ax.legend(loc="lower right")

ax = axes[1]
for proba, name, color in [(lr_proba, "Logistic regression", PALETTE["blue"]),
                            (rf_proba, "Random forest",       PALETTE["secondary"]),
                            (gb_proba, "Gradient boosting",   PALETTE["primary"])]:
    p, r, _ = precision_recall_curve(y_test, proba)
    ax.plot(r, p, color=color, linewidth=2.2,
            label=f"{name} (AP = {average_precision_score(y_test, proba):.3f})")
prevalence = y_test.mean()
ax.axhline(prevalence, color=PALETTE["muted"], linestyle="--", linewidth=1,
           label=f"baseline = {prevalence:.2f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-recall", loc="left"); ax.legend(loc="upper right")

plt.tight_layout(); plt.show()
"""))

cells.append(md("""\
All three models are materially better than chance; gradient boosting wins
narrowly with ROC-AUC = 0.827. The fact that all three converge to within
0.01 of each other is a clue that the *survey*, not the algorithm, is the
binding constraint — adding more sophisticated models would not unlock the
next tier of accuracy; adding better questions might.

### 4.2 Confusion at threshold = 0.5

We pick gradient boosting as our headline model and look at its decisions
at the default threshold.
"""))

cells.append(code("""\
y_pred_gb = (gb_proba >= 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred_gb)
tn, fp, fn, tp = cm.ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)
print(f"Accuracy:    {accuracy_score(y_test, y_pred_gb):.4f}")
print(f"F1:          {f1_score(y_test, y_pred_gb):.4f}")
print(f"Sensitivity: {sens:.4f}")
print(f"Specificity: {spec:.4f}")

fig, ax = plt.subplots(figsize=(5.5, 4.8))
ax.imshow(cm, cmap="Reds", aspect="auto")
for (i, j), v in np.ndenumerate(cm):
    color = "white" if v > cm.max()*0.55 else PALETTE["ink"]
    ax.text(j, i, f"{v:,}", ha="center", va="center", color=color, fontsize=14)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Pred: no DM", "Pred: DM"])
ax.set_yticklabels(["True: no DM", "True: DM"])
ax.set_title("Gradient boosting confusion (threshold 0.5)", loc="left")
ax.grid(False)
for s in ax.spines.values(): s.set_visible(False)
plt.show()
"""))

cells.append(md("""\
At a 0.5 cutoff the gradient-boosting model catches **79.5%** of true
diabetics (sensitivity) at the cost of flagging **29.4%** of healthy
respondents as positive. For a *screening* workflow — *who should go get a
blood test?* — that trade-off is roughly the right shape: high sensitivity
even if specificity sags.

### 4.3 Threshold tuning and calibration

The choice of decision threshold is where machine learning meets clinical
priority. A screening tool wants to err toward false positives; a
self-diagnostic widget might want the opposite. The sweep below shows the
trade-off explicitly. The calibration curve checks whether the model's
predicted probabilities mean what they claim.
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

# Threshold sweep
ax = axes[0]
thresholds = np.linspace(0.05, 0.95, 91)
sens_arr, spec_arr, prec_arr = [], [], []
for t in thresholds:
    pred = (gb_proba >= t).astype(int)
    _tn, _fp, _fn, _tp = confusion_matrix(y_test, pred).ravel()
    sens_arr.append(_tp / (_tp + _fn) if (_tp + _fn) else 0)
    spec_arr.append(_tn / (_tn + _fp) if (_tn + _fp) else 0)
    prec_arr.append(_tp / (_tp + _fp) if (_tp + _fp) else 0)
ax.plot(thresholds, sens_arr, color=PALETTE["primary"],   linewidth=2, label="sensitivity")
ax.plot(thresholds, spec_arr, color=PALETTE["blue"],       linewidth=2, label="specificity")
ax.plot(thresholds, prec_arr, color=PALETTE["secondary"], linewidth=2, label="precision")
ax.axvline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=1)
ax.set_xlabel("Decision threshold"); ax.set_ylabel("Rate")
ax.set_title("Threshold trade-off", loc="left"); ax.legend(loc="center right")
ax.set_xlim(0.05, 0.95); ax.set_ylim(0, 1.02)

# Calibration
ax = axes[1]
prob_true, prob_pred = calibration_curve(y_test, gb_proba, n_bins=10, strategy="quantile")
ax.plot([0,1], [0,1], color=PALETTE["muted"], linestyle="--", linewidth=1, label="perfect calibration")
ax.plot(prob_pred, prob_true, color=PALETTE["primary"], linewidth=2.2,
        marker="o", markerfacecolor=PALETTE["primary"], markeredgecolor="white", markersize=8,
        label="gradient boosting")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed diabetes rate")
ax.set_title("Calibration", loc="left"); ax.legend(loc="upper left")
ax.set_xlim(0,1); ax.set_ylim(0,1)

plt.tight_layout(); plt.show()
"""))

cells.append(md("""\
## 5. What the model learned

Two complementary views of which features carry weight. The random forest's
impurity-based importance is non-negative and direction-blind; the
standardized logistic-regression coefficients tell us the *sign* and the
relative strength on a common scale.
"""))

cells.append(code("""\
lr_coef = lr.coef_[0]
rf_imp  = rf.feature_importances_

imp_df = pd.DataFrame({"feature": feature_cols, "lr_coef": lr_coef,
                       "rf_importance": rf_imp})

fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))

# RF importances
order = imp_df.sort_values("rf_importance")
axes[0].barh(order["feature"], order["rf_importance"],
              color=PALETTE["primary"], edgecolor="white", linewidth=0.5)
axes[0].set_xlabel("Random forest importance")
axes[0].set_title("Impurity-based importance (RF)", loc="left")

# Signed LR coefficients
order = imp_df.sort_values("lr_coef")
colors = [PALETTE["primary"] if c > 0 else PALETTE["blue"] for c in order["lr_coef"].values]
axes[1].barh(order["feature"], order["lr_coef"], color=colors, edgecolor="white", linewidth=0.5)
axes[1].axvline(0, color=PALETTE["ink"], linewidth=0.8)
axes[1].set_xlabel("Standardized LR coefficient")
axes[1].set_title("Direction & strength (logistic regression)", loc="left")

plt.tight_layout(); plt.show()
"""))

cells.append(md("""\
The most influential features — BMI, self-rated `GenHlth`, age, high blood
pressure, high cholesterol — are exactly the questions a primary-care doctor
would ask first. Reassuring rather than surprising. The most counter-intuitive
signal is that `HvyAlcoholConsump` carries a *protective* coefficient. That
is almost certainly an artifact of how the survey codes heavy drinking in a
population skewed toward older respondents, not a causal relationship.

## 6. The interactive risk calculator

The published GitHub-Pages site at
**[amath18.github.io/cmsc320finalproject](https://amath18.github.io/cmsc320finalproject/)**
exports the trained logistic regression as `model.json` and runs predictions
client-side, so a reader can plug in their own answers and see a probability
in real time. The export step:
"""))

cells.append(code("""\
model_export = {
    "model": "logistic_regression",
    "intercept": float(lr.intercept_[0]),
    "features": feature_cols,
    "coefficients": [float(c) for c in lr_coef],
    "scaler_mean": [float(m) for m in scaler.mean_],
    "scaler_scale": [float(s) for s in scaler.scale_],
}
with open("docs/model.json", "w") as f:
    json.dump(model_export, f, indent=2)
print(f"Exported {len(feature_cols)} features.")
"""))

# ---------------------------------------------------------------------------
cells.append(md("""\
## 7. Insights and conclusions

**The questionnaire works — for triage.** A 90-second self-report can hit
ROC-AUC = 0.83 and catch four out of five diabetics at a threshold tuned for
sensitivity. That is not good enough to *diagnose* anyone — false positives
outnumber true positives at almost every operating point — but it is plenty
good enough to triage: to flag the people who should go get a real blood
test, ahead of the people who probably don't need one.

**The signal is in the survey, not the algorithm.** All three models — a
linear baseline, a random forest, and a gradient-boosted ensemble — converged
within 0.01 of each other on ROC-AUC. That is the signature of a feature
ceiling. The most informative existing features are the ones a primary-care
doctor would ask first: BMI, self-rated health, age, blood pressure,
cholesterol. The features the doctor would not bother asking — fruit
consumption, daily veggie intake, sex — pulled almost no weight.

**Be honest about what you can't see.** The data are self-reported,
cross-sectional, and from 2015. We learn what BRFSS respondents *say* about
themselves, not what shows up on a glucose tolerance test. Heavy alcohol
consumption pulled a *protective* coefficient — not because drinking prevents
diabetes, but because it correlates with younger, healthier respondents in
this sample. A more careful study would disentangle that with stratification
or causal modeling. We flag it here as a caution against reading the
coefficients as policy.

A reader who came in cold should leave with an intuition for the
data-science pipeline — hypothesis tests, model comparison, threshold
trade-offs, calibration — and a working sense of what a 13.9%-prevalence
binary classifier can and cannot tell them. A reader who came in knowing the
BRFSS dataset should leave with one new specific number: **0.827 ROC-AUC** is
attainable from the questionnaire alone, and the binding constraint isn't
the model.

---

## 8. Citations

- Centers for Disease Control and Prevention. *BRFSS 2015 Codebook.*
  [cdc.gov/brfss/annual_data/annual_2015.html](https://www.cdc.gov/brfss/annual_data/annual_2015.html).
- Teboul, A. *Diabetes Health Indicators Dataset.* Kaggle, 2021.
  [kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).
- Pedregosa, F. *et al.* *Scikit-learn: Machine Learning in Python.*
  *JMLR* **12**, 2825–2830 (2011).
- Niculescu-Mizil, A. & Caruana, R. *Predicting Good Probabilities with
  Supervised Learning.* ICML 2005 — calibration background.
- Hosmer, D. W. & Lemeshow, S. *Applied Logistic Regression* (2nd ed.).
  Wiley, 2000.
- Chen, T. & Guestrin, C. *XGBoost: A Scalable Tree Boosting System.*
  KDD 2016 — gradient boosting background.
- American Diabetes Association. *Standards of Medical Care in Diabetes —
  2024.* [diabetesjournals.org](https://diabetesjournals.org/care/issue/47/Supplement_1).

Python libraries used: `pandas`, `numpy`, `scipy.stats`, `matplotlib`,
`scikit-learn`. Cited via the import block at the top of §2.
"""))

# ---------------------------------------------------------------------------
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "final_tutorial.ipynb"
out.write_text(json.dumps(notebook, indent=1))
print(f"wrote {out.relative_to(ROOT)} ({len(cells)} cells)")
