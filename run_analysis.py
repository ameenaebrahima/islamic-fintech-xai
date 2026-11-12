"""Runs every experiment reported in the paper on islamic_fintech_credit_dataset.csv
and prints the numbers used in the text, tables and figures."""
import json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score, precision_score, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

SEED = 42
df = pd.read_csv("islamic_fintech_credit_dataset.csv")
FEATS = ["MonthlyIncome_BHD", "CreditScore", "DebtToEquity", "TangibleAssetValue_BHD", "LatePayments_12M", "ActiveProducts"]
X = df[FEATS]          # CustomerID, ContractType and DefaultFlag are never inputs
out = {}

def lr(cw=None): return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight=cw))
def dt(depth=4): return DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=SEED)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

# ---------- ContractType ----------
y = df["ContractType"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
res = {}
for name, m in [("LR", lr()), ("DT", dt())]:
    m.fit(Xtr, ytr); p = m.predict(Xte)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="macro", zero_division=0)
    per = dict(zip(["Ijarah","Murabaha","Musharakah"], recall_score(yte, p, average=None, labels=["Ijarah","Murabaha","Musharakah"])))
    res[name] = dict(acc=accuracy_score(yte,p), prec=pr, rec=rc, f1=f1, per_class_recall=per,
                     cv=cross_val_score(m, Xtr, ytr, cv=cv).mean(), model=m)
out["contract"] = {k:{kk:vv for kk,vv in v.items() if kk!="model"} for k,v in res.items()}
out["contract_test_counts"] = yte.value_counts().to_dict()
out["depth_sweep"] = {d: cross_val_score(dt(d), Xtr, ytr, cv=cv).mean() for d in range(2, 9)}
lrm = res["LR"]["model"][-1]
out["lr_contract_coef"] = dict(zip(FEATS, np.abs(lrm.coef_).mean(axis=0)))
dtm = res["DT"]["model"]
out["dt_contract_imp"] = dict(zip(FEATS, dtm.feature_importances_))
t = dtm.tree_
out["root"] = dict(feature=FEATS[t.feature[0]], thr=t.threshold[0])
def node(i): 
    v = t.value[i][0]*t.n_node_samples[i] if t.value[i][0].sum()<=1.0001 else t.value[i][0]
    return dict(n=int(t.n_node_samples[i]), counts=dict(zip(dtm.classes_, np.round(v).astype(int).tolist())),
                feature=FEATS[t.feature[i]] if t.feature[i]>=0 else None, thr=float(t.threshold[i]))
out["tree_nodes"] = {i: node(i) for i in range(min(t.node_count, 15))}
out["tree_children"] = {i:(int(t.children_left[i]), int(t.children_right[i])) for i in range(min(t.node_count,15))}

# ---------- DefaultFlag ----------
y2 = df["DefaultFlag"]
Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y2, test_size=0.2, stratify=y2, random_state=SEED)
def score(m, Xa, Xb):
    m.fit(Xa, ytr2); p = m.predict(Xb)
    pr, rc, f1, _ = precision_recall_fscore_support(yte2, p, average="binary", zero_division=0)
    return dict(acc=accuracy_score(yte2,p), prec=pr, rec=rc, f1=f1)
NO_LP = [f for f in FEATS if f!="LatePayments_12M"]
out["default"] = {"LR": score(lr(), Xtr2, Xte2), "DT": score(dt(), Xtr2, Xte2)}
out["default_noLP"] = {"LR": score(lr(), Xtr2[NO_LP], Xte2[NO_LP]), "DT": score(dt(), Xtr2[NO_LP], Xte2[NO_LP])}
out["default_LR_balanced"] = score(lr("balanced"), Xtr2, Xte2)
out["default_LR_balanced_noLP"] = score(lr("balanced"), Xtr2[NO_LP], Xte2[NO_LP])
out["default_test_pos"] = int(yte2.sum())
m = lr().fit(Xtr2, ytr2)
out["lr_default_coef"] = dict(zip(FEATS, m[-1].coef_[0]))

# ---------- Clustering (all 1,200 records, no labels) ----------
Z = StandardScaler().fit_transform(X)
sil = {"kmeans": {}, "ward": {}}
for k in range(2, 7):
    sil["kmeans"][k] = silhouette_score(Z, KMeans(k, n_init=10, random_state=SEED).fit_predict(Z))
    sil["ward"][k] = silhouette_score(Z, AgglomerativeClustering(k, linkage="ward").fit_predict(Z))
out["silhouette"] = sil
km = KMeans(2, n_init=10, random_state=SEED).fit(Z)
lab = km.labels_
# order clusters so cluster 0 = lower income
if df.loc[lab==0,"MonthlyIncome_BHD"].mean() > df.loc[lab==1,"MonthlyIncome_BHD"].mean(): lab = 1-lab
prof = {}
for c in [0,1]:
    s = df[lab==c]
    prof[c] = dict(n=len(s), inc=s.MonthlyIncome_BHD.mean(), ta=s.TangibleAssetValue_BHD.mean(), de=s.DebtToEquity.mean(),
                   contracts=s.ContractType.value_counts().to_dict())
out["clusters"] = prof
pca = PCA(2).fit(Z); out["pca_var"] = pca.explained_variance_ratio_.tolist()
np.save("cluster_labels.npy", lab)

json.dump(out, open("results.json","w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=lambda o: round(float(o),4)))
