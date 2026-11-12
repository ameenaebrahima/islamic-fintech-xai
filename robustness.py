"""Robustness and default-cutoff analysis added in the revision (Section IV-C, Table V).

It re-uses the generator of generate_dataset.py unchanged, but with the random seed and
the label-noise rate as parameters. With seed 42 and noise 0.10 it reproduces
islamic_fintech_credit_dataset.csv exactly (checked below).

Run:  python robustness.py      -> prints the numbers and writes robustness_results.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
                             precision_score, recall_score)

N = 1200
FEATS = ["MonthlyIncome_BHD", "CreditScore", "DebtToEquity", "TangibleAssetValue_BHD",
         "LatePayments_12M", "ActiveProducts"]


def generate(seed=42, label_noise=0.10, target_default_rate=0.18):
    """Same steps and same order of random draws as generate_dataset.py."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, N)
    monthly_income = np.clip(rng.lognormal(6.55 + 0.12 * z, 0.30), 400, 3000)
    credit_score = np.clip(680 + 45 * z + rng.normal(0, 25, N), 300, 850)
    debt_to_equity = np.clip(0.9 * rng.beta(2.2, 5, N) - 0.03 * z, 0.05, 0.80)
    tangible_assets = np.clip(monthly_income * rng.uniform(5, 15, N) * np.exp(0.15 * z), 1500, 30000)
    active_products = rng.choice([1, 2, 3, 4], N, p=[0.20, 0.40, 0.28, 0.12])
    late_payments = rng.poisson(np.exp(-1.1 - 0.5 * z + debt_to_equity))

    rule = np.where(tangible_assets < 8000, "Murabaha",
                    np.where((debt_to_equity <= 0.22) & (monthly_income >= 700), "Musharakah", "Ijarah"))
    contract = rule.astype(object).copy()
    labels = np.array(["Murabaha", "Ijarah", "Musharakah"])
    flip = rng.random(N) < label_noise
    for i in np.where(flip)[0]:
        others = labels[labels != contract[i]]
        contract[i] = others[rng.integers(0, 2)]

    def risk_score(b):
        return (b - 0.020 * (credit_score - 680) + 0.45 * late_payments
                + 2.0 * (debt_to_equity - 0.3) - 0.0010 * (monthly_income - 700))
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-risk_score(mid)))).mean() < target_default_rate:
            lo = mid
        else:
            hi = mid
    p_default = 1 / (1 + np.exp(-risk_score((lo + hi) / 2)))
    default_flag = (rng.random(N) < p_default).astype(int)

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
    return df, rule.astype(str), p_default


def lr(cw=None):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight=cw))


def dt():
    return DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)


def contract_run(df, rule, seed):
    """Test accuracy of LR and DT, and the 'rule ceiling': how often the noise-free
    generation rule agrees with the (noisy) test labels."""
    idx = np.arange(N)
    tr, te = train_test_split(idx, test_size=0.2, stratify=df.ContractType, random_state=seed)
    X, y = df[FEATS], df.ContractType
    r = {}
    for name, m in [("LR", lr()), ("DT", dt())]:
        m.fit(X.iloc[tr], y.iloc[tr])
        r[name] = accuracy_score(y.iloc[te], m.predict(X.iloc[te]))
    r["ceiling"] = float((rule[te] == y.values[te]).mean())
    return r


def default_run(df, p_true, seed):
    idx = np.arange(N)
    y = df.DefaultFlag.values
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    X = df[FEATS]
    r = {}
    for name, m in [("LR", lr()), ("LRw", lr("balanced")), ("DT", dt())]:
        m.fit(X.iloc[tr], y[tr])
        s = m.predict_proba(X.iloc[te])[:, 1]
        p = (s >= 0.5).astype(int)
        r[name] = dict(auc=roc_auc_score(y[te], s), ap=average_precision_score(y[te], s),
                       rec=recall_score(y[te], p, zero_division=0),
                       prec=precision_score(y[te], p, zero_division=0))
    # Oracle: the true default probability used to draw the labels
    r["oracle"] = dict(auc=roc_auc_score(y[te], p_true[te]), ap=average_precision_score(y[te], p_true[te]))
    r["base_rate"] = float(y[te].mean())
    return r


def ms(vals):
    a = np.asarray(vals, float)
    return [round(float(a.mean()), 4), round(float(a.std(ddof=1)), 4)]


if __name__ == "__main__":
    out = {}

    # 0. Check: seed 42, noise 0.10 reproduces the published dataset
    df42, rule42, p42 = generate(42, 0.10)
    ref, _, _ = generate(42, 0.10)
    try:
        pub = pd.read_csv("islamic_fintech_credit_dataset.csv")
        out["reproduces_published_csv"] = bool(pub.equals(df42))
    except FileNotFoundError:
        out["reproduces_published_csv"] = "csv not found"

    # 1. Rule ceiling on the published split (seed 42)
    out["seed42_contract"] = contract_run(df42, rule42, 42)
    out["seed42_noise_flips_all"] = int((rule42 != df42.ContractType.values).sum())

    # 2. Default analysis on the published split: AUC, AP, recall/precision at several cutoffs
    y = df42.DefaultFlag.values
    tr, te = train_test_split(np.arange(N), test_size=0.2, stratify=y, random_state=42)
    X = df42[FEATS]
    cut = {}
    for name, m in [("LR", lr()), ("LRw", lr("balanced"))]:
        m.fit(X.iloc[tr], y[tr])
        s = m.predict_proba(X.iloc[te])[:, 1]
        cut[name] = {"auc": roc_auc_score(y[te], s), "ap": average_precision_score(y[te], s)}
        for c in [0.2, 0.3, 0.4, 0.5]:
            p = (s >= c).astype(int)
            cut[name][f"cut_{c}"] = dict(rec=recall_score(y[te], p, zero_division=0),
                                         prec=precision_score(y[te], p, zero_division=0),
                                         flagged=int(p.sum()))
    m = dt().fit(X.iloc[tr], y[tr]); s = m.predict_proba(X.iloc[te])[:, 1]
    cut["DT"] = {"auc": roc_auc_score(y[te], s), "ap": average_precision_score(y[te], s)}
    cut["oracle"] = {"auc": roc_auc_score(y[te], p42[te]), "ap": average_precision_score(y[te], p42[te])}
    for c in [0.2, 0.3, 0.5]:
        p = (p42[te] >= c).astype(int)
        cut["oracle"][f"cut_{c}"] = dict(rec=recall_score(y[te], p, zero_division=0),
                                         prec=precision_score(y[te], p, zero_division=0))
    out["seed42_default"] = cut

    # 3. Robustness: 30 generation seeds x 3 label-noise levels (new data AND new split each time)
    SEEDS = range(1, 31)
    rob = {}
    for noise in [0.0, 0.10, 0.20]:
        rows = [contract_run(*generate(s, noise)[:2], s) for s in SEEDS]
        rob[str(noise)] = {k: ms([r[k] for r in rows]) for k in ["LR", "DT", "ceiling"]}
        rob[str(noise)]["DT_gap_to_ceiling"] = ms([r["ceiling"] - r["DT"] for r in rows])
    out["robust_contract"] = rob

    drows = []
    for s in SEEDS:
        df, _, p = generate(s, 0.10)
        drows.append(default_run(df, p, s))
    out["robust_default"] = {
        name: {k: ms([r[name][k] for r in drows]) for k in drows[0][name]}
        for name in ["LR", "LRw", "DT", "oracle"]
    }
    out["robust_default"]["base_rate"] = ms([r["base_rate"] for r in drows])

    json.dump(out, open("robustness_results.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=lambda o: round(float(o), 4)))
