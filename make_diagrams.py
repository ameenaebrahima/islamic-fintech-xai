"""Draws Fig. 1 (framework) and Fig. 2 (proposed default handling)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif","Times New Roman","DejaVu Serif"]})
def box(ax,x,y,w,h,text,color,dashed=False,fs=6.6):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.08",fc=color,ec="#222222",
                 lw=1.3 if dashed else 1.0,ls=(0,(4,2)) if dashed else "-",alpha=0.85 if dashed else 1))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",color="white",fontsize=fs,fontweight="bold",linespacing=1.25)
def arrow(ax,p,q):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=7,color="#666666",lw=1.1))

# ---------- Fig. 1 ----------
fig,ax=plt.subplots(figsize=(7,2.6)); ax.set_xlim(0,21); ax.set_ylim(0,8.1); ax.axis("off")
NAVY,BLUE,RED,GREEN,PURPLE,STEEL="#1f4e79","#2c5d8f","#b5444b","#1e7e3e","#7570b3","#4f7397"
box(ax,0.1,3.0,2.5,2.0,"Synthetic credit\ndata\n1,200 records",NAVY)
box(ax,3.1,3.0,2.5,2.0,"Riba filter\n(proposed)\nTangible asset\nbacking",RED,dashed=True)
# supervised branch (top)
box(ax,6.3,5.3,2.7,2.2,"80/20 stratified\nsplit\n960 train / 240 test",BLUE)
box(ax,9.5,5.3,2.8,2.2,"Z-score scaling\nfitted on training\nrecords only\n(inside each CV fold)",BLUE)
box(ax,12.8,5.3,3.5,2.2,"Supervised models\nLR and Decision Tree\nContractType (5-fold CV)\nDefaultFlag (test set)",GREEN)
box(ax,16.8,4.2,4.1,3.3,"AAOIFI decision layer\n(proposed)\nMurabaha (SS 8)\nMusharakah (SS 12)\nIjarah (SS 9)",STEEL,dashed=True)
# unsupervised branch (bottom)
box(ax,6.3,0.6,6.0,2.0,"All 1,200 records, no labels\nZ-score scaling on all records",PURPLE)
box(ax,12.8,0.6,3.5,2.0,"K-Means and\nHierarchical\nk = 2 to 6, Silhouette",PURPLE)
box(ax,16.8,0.6,4.1,2.0,"Customer segments\n(exploratory)",PURPLE)
arrow(ax,(2.6,4.0),(3.1,4.0)); arrow(ax,(5.6,4.4),(6.3,6.2)); arrow(ax,(5.6,3.6),(6.3,1.8))
arrow(ax,(9.0,6.4),(9.5,6.4)); arrow(ax,(12.3,6.4),(12.8,6.4)); arrow(ax,(16.3,6.4),(16.8,6.2))
arrow(ax,(12.3,1.6),(12.8,1.6)); arrow(ax,(16.3,1.6),(16.8,1.6))
ax.text(6.3,7.85,"Supervised pipeline (tested)",fontsize=6.6,style="italic",color="#333333")
ax.text(6.3,2.85,"Unsupervised pipeline (tested)",fontsize=6.6,style="italic",color="#333333")
fig.text(0.5,0.015,"Solid boxes: programmed and tested.  Dashed boxes: proposed, not tested.",ha="center",fontsize=6.6,style="italic",color="#555555")
fig.subplots_adjust(0.005,0.06,0.995,0.995); fig.savefig("fig_framework.png",dpi=300); plt.close()

# ---------- Fig. 2 ----------
fig,ax=plt.subplots(figsize=(3.4,3.5)); ax.set_xlim(0,10); ax.set_ylim(0,10.4); ax.axis("off")
box(ax,1.6,9.0,6.8,1.15,"Model output: DefaultFlag = Yes\n(risk flag only, no action)","#1f4e79",fs=6.8)
box(ax,0.3,6.55,9.4,1.85,"Human review before any action:\nconfirm the payment is overdue, then check\nability to pay and the reason for the delay","#4f7397",dashed=True,fs=6.6)
box(ax,0.2,3.75,4.4,1.3,"Procrastinating\ndebtor","#e07b2a",dashed=True,fs=6.8)
box(ax,5.4,3.75,4.4,1.3,"Insolvent\ndebtor","#8e88c2",dashed=True,fs=6.8)
box(ax,0.2,0.8,4.4,2.2,"Proposed: pre-agreed\ndonation to charity\n(SS 3, cl. 2/1/8),\nnever bank income","#3d8f5a",dashed=True,fs=6.4)
box(ax,5.4,0.8,4.4,2.2,"Proposed: grace\nperiod, no penalty\n(SS 3, App. B)","#3d8f5a",dashed=True,fs=6.4)
arrow(ax,(5,9.0),(5,8.4)); arrow(ax,(3.4,6.55),(2.4,5.05)); arrow(ax,(6.6,6.55),(7.6,5.05))
arrow(ax,(2.4,3.75),(2.4,3.0)); arrow(ax,(7.6,3.75),(7.6,3.0))
ax.text(0.15,5.75,"Overdue, able to pay,\nno valid reason",fontsize=5.8,ha="left",va="center",linespacing=1.15)
ax.text(9.85,5.75,"Unable to pay",fontsize=5.8,ha="right",va="center")
ax.text(5,0.25,"Solid: tested. Dashed: proposed, not tested.",ha="center",fontsize=6.2,style="italic",color="#555555")
fig.subplots_adjust(0.01,0.01,0.99,0.99); fig.savefig("fig_ss3.png",dpi=300); plt.close()
print("ok")
