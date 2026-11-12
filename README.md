# Explainable Machine Learning for Islamic FinTech Credit Decisions Using Synthetic Data

Code and data for the conference paper by Amina Ebrahim Mekhaimer, Nour A.Sattar Amri and Mohamed Al Mannai (University of Bahrain), presented at SIBF'26 (Bahrain, 8-9 December 2026).

## What is here

| File | What it does |
|---|---|
| `generate_dataset.py` | Creates the 1,200 synthetic records (random seed 42) using the rules in Table I of the paper |
| `islamic_fintech_credit_dataset.csv` | The dataset produced by `generate_dataset.py` |
| `run_analysis.py` | Trains Logistic Regression and the Decision Tree, runs cross-validation and clustering, and saves `results.json` |
| `make_figures.py` | Recreates Figs. 3 to 6 of the paper (plus a bar chart of Table III) |
| `make_diagrams.py` | Draws Fig. 1 (framework) and Fig. 2 (proposed default handling) |
| `robustness.py` | Revision analysis (Section IV-C, Table V): accuracy ceiling, 30 regenerated datasets at 0%, 10% and 20% label noise, default ROC-AUC / average precision and cutoff analysis; saves `robustness_results.json` |

## Important

The data is **synthetic**. All distributions and rules are assumptions chosen by the authors.
They were not estimated from the Kaggle Credit Risk Dataset (Lao Tse), which only inspired the
choice of variables. The records are not real Bahraini bank or SME records, and the contract
labels are not a Shari'ah compliance check.

## How to run

```bash
pip install -r requirements.txt
python generate_dataset.py
python run_analysis.py
python make_figures.py
python make_diagrams.py
python robustness.py
```
