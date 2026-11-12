"""Regenerates Figs. 3-7 from results.json and the dataset."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif","Times New Roman","DejaVu Serif"],
                     "font.size":8,"axes.spines.top":False,"axes.spines.right":False})
NAVY,ORANGE,RED="#1f4e79","#d95f02","#a51d24"
R=json.load(open("results.json")); df=pd.read_csv("islamic_fintech_credit_dataset.csv")
FEATS=["MonthlyIncome_BHD","CreditScore","DebtToEquity","TangibleAssetValue_BHD","LatePayments_12M","ActiveProducts"]
NICE=dict(zip(FEATS,["Monthly Income","Credit Score","Debt-to-Equity","Tangible Assets","Late Payments","Active Products"]))
def grid(ax): ax.yaxis.grid(True,color="#d7d7d7",lw=0.6,ls=":"); ax.set_axisbelow(True)
def lab(ax,bars):
    for b in bars: ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.015,f"{b.get_height():.2f}",ha="center",va="bottom",fontsize=6)

# Fig 3 performance
fig,axs=plt.subplots(2,1,figsize=(3.4,4.1))
for ax,key,title,mets in [(axs[0],"contract","(a) ContractType (macro-averaged)",["acc","prec","rec","f1"]),
                          (axs[1],"default","(b) DefaultFlag (defaulter class)",["acc","prec","rec","f1"])]:
    x=np.arange(4); w=0.36
    b1=ax.bar(x-w/2,[R[key]["LR"][m] for m in mets],w,color=NAVY,label="Logistic Regression")
    b2=ax.bar(x+w/2,[R[key]["DT"][m] for m in mets],w,color=ORANGE,label="Decision Tree")
    lab(ax,b1); lab(ax,b2); ax.set_xticks(x,["Accuracy","Precision","Recall","F1-score"]); ax.set_ylim(0,1.1)
    ax.set_ylabel("Score"); ax.set_title(title,loc="left",fontsize=8); grid(ax)
axs[0].legend(fontsize=6,ncol=2,frameon=False,loc="upper center")
fig.tight_layout(); fig.savefig("fig_performance.png",dpi=300); plt.close()

# Fig 4 ablation
fig,ax=plt.subplots(figsize=(3.4,2.5)); x=np.arange(3); w=0.36
allv=[R["default"]["LR"]["rec"],R["default_LR_balanced"]["rec"],R["default"]["DT"]["rec"]]
nolp=[R["default_noLP"]["LR"]["rec"],R["default_LR_balanced_noLP"]["rec"],R["default_noLP"]["DT"]["rec"]]
b1=ax.bar(x-w/2,allv,w,color=NAVY,label="All six inputs"); b2=ax.bar(x+w/2,nolp,w,color=RED,label="Without Late Payments")
lab(ax,b1); lab(ax,b2); ax.set_xticks(x,["Logistic Regression","LR (balanced weights)","Decision Tree"],fontsize=6.5)
ax.set_ylim(0,1.0); ax.set_ylabel("Defaulter recall"); ax.legend(fontsize=6,frameon=False); grid(ax)
fig.tight_layout(); fig.savefig("fig_ablation.png",dpi=300); plt.close()

# Fig 5 features
fig,axs=plt.subplots(2,1,figsize=(3.4,3.5))
for ax,key,col,title in [(axs[0],"lr_contract_coef",NAVY,"(a) Logistic Regression: mean |standardized coefficient|"),
                         (axs[1],"dt_contract_imp",ORANGE,"(b) Decision Tree: feature importance")]:
    items=sorted(R[key].items(),key=lambda kv:kv[1])
    bars=ax.barh([NICE[k] for k,_ in items],[v for _,v in items],color=col)
    for b in bars: ax.text(b.get_width()+0.01,b.get_y()+b.get_height()/2,f"{b.get_width():.2f}",va="center",fontsize=6)
    ax.set_title(title,loc="left",fontsize=7.5); ax.xaxis.grid(True,color="#d7d7d7",lw=0.6,ls=":"); ax.set_axisbelow(True)
    ax.set_xlim(0,max(R[key].values())*1.15)
fig.tight_layout(); fig.savefig("fig_features.png",dpi=300); plt.close()

# Fig 6 tree (same split/model as run_analysis.py)
y=df["ContractType"]; Xtr,_,ytr,_=train_test_split(df[FEATS],y,test_size=0.2,stratify=y,random_state=42)
dt=DecisionTreeClassifier(criterion="gini",max_depth=4,random_state=42).fit(Xtr,ytr)
fig,ax=plt.subplots(figsize=(7.0,2.8))
plot_tree(dt,max_depth=2,feature_names=[NICE[f] for f in FEATS],class_names=list(dt.classes_),filled=True,
          impurity=False,proportion=False,rounded=True,fontsize=6,ax=ax)
fig.tight_layout(); fig.savefig("fig_tree.png",dpi=300); plt.close()

# Fig 7 clustering
fig,axs=plt.subplots(2,1,figsize=(3.4,4.3)); ks=[2,3,4,5,6]
km=[R["silhouette"]["kmeans"][str(k)] for k in ks]; wd=[R["silhouette"]["ward"][str(k)] for k in ks]
axs[0].plot(ks,km,"o-",color="#7570b3",label="K-Means"); axs[0].plot(ks,wd,"s--",color="#1b7c3d",label="Hierarchical")
for k,v in zip(ks,km): axs[0].text(k,v+0.004,f"{v:.3f}",ha="center",fontsize=5.5)
axs[0].set_xticks(ks); axs[0].set_xlabel("Number of clusters (k)"); axs[0].set_ylabel("Silhouette score")
axs[0].set_title("(a) Silhouette score by k",loc="left",fontsize=8); axs[0].legend(fontsize=6,frameon=False); grid(axs[0])
Z=StandardScaler().fit_transform(df[FEATS]); P=PCA(2).fit(Z); pts=P.transform(Z); lab_=np.load("cluster_labels.npy")
for c,col in [(0,"#7570b3"),(1,"#1b7c3d")]:
    m=lab_==c; axs[1].scatter(pts[m,0],pts[m,1],s=3,alpha=0.45,color=col,label=f"Cluster {c} (n = {m.sum()})")
v=P.explained_variance_ratio_; axs[1].set_xlabel(f"PC1 ({v[0]*100:.0f}% variance)"); axs[1].set_ylabel(f"PC2 ({v[1]*100:.0f}% variance)")
axs[1].set_title("(b) K-Means (k = 2) in PCA space",loc="left",fontsize=8); axs[1].legend(fontsize=6,frameon=False,markerscale=3)
fig.tight_layout(); fig.savefig("fig_clustering.png",dpi=300); plt.close()
print("figures done")
