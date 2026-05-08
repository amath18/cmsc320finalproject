# Predicting Diabetes from Survey Data Alone

CMSC 320 — Spring 2026 final project.
Pranav Bykampadi · Ansh Mathur · Alvin Persaud · Adhyyan Kumar.

We took the 2015 BRFSS health survey — 253,680 people answering the same
22 questions the CDC has been asking for decades — and asked whether a
90-second questionnaire carries enough signal to flag someone at elevated
risk for diabetes before they ever see a clinic.

The short version: a gradient-boosting model gets to ROC-AUC ≈ 0.83 and
catches roughly four out of five real diabetics at a sensitivity-tuned
threshold. Not enough to *diagnose* anyone — but plenty good enough to
triage who should go get a real blood test. The full walkthrough lives at:

**[https://amath18.github.io/cmsc320finalproject/](https://amath18.github.io/cmsc320finalproject/)**

There's an interactive risk calculator at the bottom of the page that runs
the trained model in your browser. No data leaves the page.

## What's in here

```
analysis.py                # the whole pipeline in one file (preprocessing,
                           #   3 hypothesis tests, 3 ML models, all the
                           #   plots, exports model.json + metrics.json)
build_notebook.py          # builds final_tutorial.ipynb from the analysis
final_tutorial.ipynb       # canonical notebook — every section, executed
cmsc320checkpoint2.ipynb   # what we submitted for checkpoint 2 (left as-is)
diabetes_binary_health_indicators_BRFSS2015.csv   # the dataset
docs/                      # GitHub Pages serves this folder
  index.html, styles.css, script.js
  notebook.html            # nbconvert HTML of the notebook
  model.json, metrics.json
  figures/                 # every chart used on the site
```

## Running it locally

The CSV needs to be in the project root. If you don't have it, grab it from
[Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
(original source: [CDC BRFSS 2015](https://www.cdc.gov/brfss/annual_data/annual_2015.html)).

```bash
pip3 install pandas numpy scipy matplotlib scikit-learn jupyter nbconvert

# regenerate every figure + the model JSON
python3 analysis.py

# rebuild the notebook from the analysis (optional — the committed copy is
# already executed)
python3 build_notebook.py
jupyter nbconvert --to notebook --execute final_tutorial.ipynb \
        --output final_tutorial.ipynb --ExecutePreprocessor.timeout=600

# preview the site
python3 -m http.server --directory docs 8000
# then http://localhost:8000
```

If `nbconvert` complains about a missing template, set `JUPYTER_PATH` to
the homebrew share dir before running — the templates ship with the
homebrew jupyter package but jupyter doesn't always find them on macOS:

```bash
JUPYTER_PATH=/opt/homebrew/share/jupyter jupyter nbconvert --to html ...
```

## Deploying

GitHub repo → Settings → Pages → "Deploy from a branch" → branch `main`,
folder `/docs` → Save. The site is live at
`https://amath18.github.io/cmsc320finalproject/` a minute or so later.
There's no Jekyll, no Actions workflow — `docs/` is just static files.

## Libraries used

`pandas`, `numpy`, `scipy.stats`, `matplotlib`, `scikit-learn`, `nbconvert`.
Per the course rules, the citation for these is the import block at the top
of `analysis.py` and §2 of `final_tutorial.ipynb`.

## Acknowledgements

CDC BRFSS for the data. Alex Teboul for the cleaned Kaggle release. Dr.
Fardina Alam and the CMSC 320 staff for the project structure.
