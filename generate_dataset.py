"""
Synthetic data generator for:
"Explainable Machine Learning for Islamic FinTech Credit Decisions Using Synthetic Data"

All distributions and rules below are assumptions chosen by the authors (see Table I).
They were NOT estimated from the Kaggle Credit Risk Dataset, which only inspired the
choice of variables. The records are not real Bahraini SME or bank records.

Run:  python generate_dataset.py      -> writes islamic_fintech_credit_dataset.csv
"""
import numpy as np
import pandas as pd

SEED = 42
N = 1200
LABEL_NOISE = 0.10          # share of contract labels replaced at random
TARGET_DEFAULT_RATE = 0.18  # expected default rate used to calibrate the intercept

rng = np.random.default_rng(SEED)

# 1. Latent financial strength, z ~ N(0, 1). It links the inputs so they are not independent.
z = rng.normal(0, 1, N)

# 2. Six numeric inputs (Table I)
monthly_income = np.clip(rng.lognormal(6.55 + 0.12 * z, 0.30), 400, 3000)           # BHD
credit_score   = np.clip(680 + 45 * z + rng.normal(0, 25, N), 300, 850)
debt_to_equity = np.clip(0.9 * rng.beta(2.2, 5, N) - 0.03 * z, 0.05, 0.80)
tangible_assets = np.clip(monthly_income * rng.uniform(5, 15, N) * np.exp(0.15 * z),
                          1500, 30000)                                                  # BHD
active_products = rng.choice([1, 2, 3, 4], N, p=[0.20, 0.40, 0.28, 0.12])
# Late payments counted in the 12 months BEFORE the observation date
late_payments  = rng.poisson(np.exp(-1.1 - 0.5 * z + debt_to_equity))

# 3. ContractType: threshold rule, then 10% random label noise (analyst disagreement)
contract = np.where(tangible_assets < 8000, "Murabaha",
            np.where((debt_to_equity <= 0.22) & (monthly_income >= 700), "Musharakah", "Ijarah"))
contract = contract.astype(object)
labels = np.array(["Murabaha", "Ijarah", "Musharakah"])
flip = rng.random(N) < LABEL_NOISE
for i in np.where(flip)[0]:
    others = labels[labels != contract[i]]
    contract[i] = others[rng.integers(0, 2)]

# 4. DefaultFlag: random draw from a logistic risk function.
#    Default is counted in the 12 months AFTER the observation date.
def risk_score(intercept):
    return (intercept
            - 0.020 * (credit_score - 680)
            + 0.45 * late_payments
            + 2.0 * (debt_to_equity - 0.3)
            - 0.0010 * (monthly_income - 700))

lo, hi = -10.0, 10.0                      # calibrate intercept once by bisection
for _ in range(100):
    mid = (lo + hi) / 2
    if (1 / (1 + np.exp(-risk_score(mid)))).mean() < TARGET_DEFAULT_RATE:
        lo = mid
    else:
        hi = mid
intercept = (lo + hi) / 2
p_default = 1 / (1 + np.exp(-risk_score(intercept)))
default_flag = (rng.random(N) < p_default).astype(int)

# 5. Save
df = pd.DataFrame({
    "CustomerID": np.arange(1001, 1001 + N),
    "MonthlyIncome_BHD": monthly_income.round(2),
    "CreditScore": credit_score.round().astype(int),
    "DebtToEquity": debt_to_equity.round(3),
    "TangibleAssetValue_BHD": tangible_assets.round(2),
    "LatePayments_12M": late_payments,
    "ActiveProducts": active_products,
    "ContractType": contract.astype(str),
    "DefaultFlag": default_flag,
})

if __name__ == "__main__":
    df.to_csv("islamic_fintech_credit_dataset.csv", index=False)
    print(f"Intercept calibrated to {intercept:.3f}")
    print(df["ContractType"].value_counts().to_dict(), "defaults:", int(df["DefaultFlag"].sum()))
    print(df.describe().loc[["min", "max"]].round(2).to_string())
