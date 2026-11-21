import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#777","axes.labelcolor":"#222","xtick.color":"#555","ytick.color":"#555","axes.grid":True,"grid.color":"#e6e6e6","grid.linewidth":.6,"axes.axisbelow":True})
C={"m":"#2a78d6","g":"#eb6834","p":"#1baf7a"}; NM={"m":"Meningioma","g":"Glioma","p":"Pituitary"}
OUT="/mnt/project-files/clahe-brisc/figures"; import os; os.makedirs(OUT,exist_ok=True)
# Fig 1 Hamming
h=[3982,0,544,0,8,0,1,0,13,0,50,0,80,0,77,0,29,0,5,0,4]
fig,ax=plt.subplots(figsize=(6,3.2)); x=np.arange(len(h))
ax.bar(x[:7],h[:7],color="#2a78d6",width=.8,label="Confirmed copy (r ≥ 0.95)")
ax.bar(x[7:],h[7:],color="#9a9a9a",width=.8,label="Not a copy")
ax.set_yscale("log"); ax.axvline(6.5,color="#444",ls="--",lw=1); ax.text(6.7,2000,"threshold = 6",fontsize=8,color="#444")
ax.set_xlabel("Minimum pHash Hamming distance to Figshare (bits)"); ax.set_ylabel("BRISC tumour images (log scale)")
for i in (0,2): ax.text(i,h[i]*1.15,f"{h[i]:,}",ha="center",fontsize=8)
ax.legend(frameon=False,fontsize=8,loc="upper right"); fig.tight_layout(); fig.savefig(f"{OUT}/Fig1_hamming.png",dpi=300); plt.close()
# Fig 2 Dice by class
H={"m":[1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,2,0,1,3,2,1,4,1,5,2,2,6,10,21,39,63,130,208,189,13],
   "g":[51,4,4,5,7,4,5,8,7,6,4,9,7,6,12,9,11,18,11,12,18,19,22,12,20,27,32,36,35,44,39,60,62,79,95,122,139,124,50,2],
   "p":[7,0,0,0,1,2,0,1,2,3,3,1,4,3,10,1,7,5,5,4,6,6,16,10,12,16,16,25,17,24,38,44,62,78,105,119,134,98,45,0]}
N={"m":706,"g":1237,"p":930}; med={"m":.934,"g":.824,"p":.859}
fig,axs=plt.subplots(3,1,figsize=(6,5.2),sharex=True); b=np.linspace(0,1,41)
for ax,k in zip(axs,["m","p","g"]):
    v=np.array(H[k])/N[k]*100; ax.bar(b[:-1],v,width=.025,align="edge",color=C[k],edgecolor="white",linewidth=.5)
    ax.axvline(med[k],color="#222",lw=1); ax.text(med[k]-.01,ax.get_ylim()[1]*.85 if ax.get_ylim()[1]>0 else 1,f"median {med[k]:.3f}",ha="right",fontsize=8)
    ax.set_title(f"{NM[k]} (n = {N[k]:,})",fontsize=9,loc="left"); ax.set_ylabel("% of slices")
axs[-1].set_xlabel("Dice between Figshare and BRISC masks of the same slice"); fig.tight_layout(); fig.savefig(f"{OUT}/Fig2_dice_by_class.png",dpi=300); plt.close()
# Fig 3 Bland-Altman
P=[p.split(",") for p in open("pts.txt").read().strip().split(";")]
fig,ax=plt.subplots(figsize=(6,3.8))
for k in ["m","p","g"]:
    q=np.array([[float(a),float(d)] for a,d,c in P if c==k]); ax.scatter(q[:,0],q[:,1],s=10,color=C[k],alpha=.6,edgecolor="white",linewidth=.3,label=NM[k])
mu,sd=-61.2,583.4
ax.axhline(mu,color="#222",lw=1); [ax.axhline(mu+s*1.96*sd,color="#222",lw=1,ls="--") for s in (-1,1)]
xm=ax.get_xlim()[1]
ax.text(xm,mu+40,f"mean {mu:.0f}",ha="right",fontsize=8); ax.text(xm,mu+1.96*sd+40,f"+1.96 SD {mu+1.96*sd:.0f}",ha="right",fontsize=8); ax.text(xm,mu-1.96*sd-160,f"−1.96 SD {mu-1.96*sd:.0f}",ha="right",fontsize=8)
ax.set_xlabel("Mean tumour area of the two masks (mm²)"); ax.set_ylabel("BRISC − Figshare area (mm²)")
ax.legend(frameon=False,fontsize=8,loc="lower left"); fig.tight_layout(); fig.savefig(f"{OUT}/Fig3_bland_altman.png",dpi=300); plt.close()
# Fig 4 CLAHE
d=pd.read_csv("../all_seeds.csv"); d=d[d.calibration=="raw"]
MC={"resnet50":("ResNet50","#2a78d6","o"),"densenet121":("DenseNet121","#eb6834","s"),"efficientnet_b0":("EfficientNet-B0","#1baf7a","^")}
conds=["none","clahe_c1","clahe_c2","clahe_c4"]; lab=["None","Clip 1","Clip 2","Clip 4"]
fig,axs=plt.subplots(1,2,figsize=(7.2,3.3))
for ax,sp,t in zip(axs,["figshare_test","brisc_clean"],["Figshare test (patient-level split)","261 non-copied BRISC images"]):
    for j,(m,(n,c,mk)) in enumerate(MC.items()):
        g=d[(d.split==sp)&(d.model==m)].groupby("condition").acc.agg(["mean","std"]).reindex(conds)
        xx=np.arange(4)+(j-1)*.08
        ax.errorbar(xx,g["mean"],yerr=g["std"],color=c,marker=mk,ms=6,lw=2,capsize=3,label=n,mec="white",mew=.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(lab); ax.set_title(t,fontsize=9,loc="left"); ax.set_xlabel("CLAHE setting")
axs[0].set_ylabel("Accuracy (mean ± SD, 3 seeds)"); axs[0].legend(frameon=False,fontsize=8,loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/Fig5_clahe_accuracy.png",dpi=300); plt.close()
print(os.listdir(OUT))
