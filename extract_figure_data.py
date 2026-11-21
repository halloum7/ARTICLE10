# Data for Figures 1-5 (mm conversion, Bland-Altman, Dice<0.1 diagnosis, mask examples).
# Run in the same Python session after src/pipeline.py, which defines AUD, dup, AGR, FIG, BR, PAIR, BI,
# brisc_mask_on_fig and agreement. Prints JSON blocks read by the plotting scripts.

# ==== extraction compacte pour figures, mm et coupes Dice<0,1 ====
import json as _j
def _f(x): return None if x is None or (isinstance(x,float) and np.isnan(x)) else round(float(x),4)
OUT={}
OUT["ham_hist"]=np.bincount(AUD.hamming.clip(0,33),minlength=34).tolist()
OUT["corr_dup_min"]=_f(dup.pixel_corr.min())
OUT["corr_nondup_q"]=[_f(q) for q in AUD[~AUD.duplicate].pixel_corr.quantile([0,.5,.9,1])]
OUT["ham_nondup_q"]=[_f(q) for q in AUD[~AUD.duplicate].hamming.quantile([0,.1,.5,1])]
OUT["copies_per_slice"]=dup.groupby("fig_idx").size().value_counts().sort_index().to_dict()
OUT["copies_per_slice"]={str(k):int(v) for k,v in OUT["copies_per_slice"].items()}
size=np.array([FIG["img"][i].shape[0] for i in AGR.fig_idx]); AGR["size"]=size
AGR["px_mm"]=0.49*512/AGR["size"]
AGR["hd95_mm"]=AGR.hd95_px*AGR.px_mm; AGR["assd_mm"]=AGR.assd_px*AGR.px_mm
OUT["size_counts"]={str(k):int(v) for k,v in pd.Series(size).value_counts().items()}
OUT["mm"]={c:{k:[_f(q) for q in AGR[AGR.cls==c][k].quantile([.25,.5,.75])] for k in ["hd95_mm","assd_mm"]} for c in CLASSES}
OUT["mm"]["all"]={k:[_f(q) for q in AGR[k].quantile([.25,.5,.75])] for k in ["hd95_mm","assd_mm"]}
OUT["n_cls"]=AGR.cls.value_counts().to_dict()
bins=np.linspace(0,1,41)
OUT["dice_hist"]={c:np.histogram(AGR[AGR.cls==c].dice.dropna(),bins)[0].tolist() for c in CLASSES}
m_=(AGR.area_fig+AGR.area_brisc)/2*AGR.px_mm**2; d_=(AGR.area_brisc-AGR.area_fig)*AGR.px_mm**2
OUT["ba"]=dict(mean=_f(d_.mean()),sd=_f(d_.std()))
rng=np.random.default_rng(0); idx=rng.choice(len(AGR),min(1500,len(AGR)),replace=False)
OUT["ba_pts"]=[[round(float(m_.iloc[i]),1),round(float(d_.iloc[i]),1),AGR.cls.iloc[i][0]] for i in idx]
# coupes Dice<0,1 : diagnostic
low=AGR[AGR.dice<0.1]; rows=[]
allc=dup.groupby("fig_idx")
for _,r in low.iterrows():
    fi=int(r.fig_idx); mf=FIG["mask"][fi].astype(bool)
    best_other=0.0; best_flip=0.0
    for _,c in allc.get_group(fi).iterrows():
        bi=BI[c.brisc]
        best_other=max(best_other,agreement(mf,brisc_mask_on_fig(bi,fi,int(c.variant)))["dice"])
        for v in range(4):
            best_flip=max(best_flip,agreement(mf,brisc_mask_on_fig(bi,fi,v))["dice"])
    mb=PAIR[fi].astype(bool)
    cf=np.argwhere(mf).mean(0) if mf.any() else np.array([np.nan,np.nan]); cb=np.argwhere(mb).mean(0) if mb.any() else np.array([np.nan,np.nan])
    rows.append(dict(cls=r.cls,dice=_f(r.dice),af=int(mf.sum()),ab=int(mb.sum()),cd=_f(np.linalg.norm(cf-cb)*0.49*512/mf.shape[0]),
                     other=_f(best_other),flip=_f(best_flip),n=int(len(allc.get_group(fi)))))
OUT["low"]=rows
print("<<<FIGDATA"+_j.dumps(OUT)+"FIGDATA>>>")


# ==== mask examples (Figure 4) ====
import base64, io, json as _j
from PIL import Image as _I
S=180
def jpg(a):
    im=_I.fromarray(a).resize((S,S), _I.BILINEAR); b=io.BytesIO(); im.save(b,"JPEG",quality=75); return base64.b64encode(b.getvalue()).decode()
def poly(m, sc):
    cs,_=cv2.findContours((m>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return [(cv2.approxPolyDP(c,1.0,True)[:,0,:]*sc).round(1).tolist() for c in cs if len(c)>=3]
def crop_box(ms, shape):
    u=np.zeros(shape,bool)
    for m in ms: u|=m.astype(bool)
    ys,xs=np.where(u); cy,cx=(ys.min()+ys.max())/2,(xs.min()+xs.max())/2
    h=max(ys.max()-ys.min(),xs.max()-xs.min())*1.6; h=int(max(h,shape[0]*0.35)); h=min(h,shape[0])
    y0=int(np.clip(cy-h/2,0,shape[0]-h)); x0=int(np.clip(cx-h/2,0,shape[1]-h)); return y0,x0,h
def panel(fi, masks, label):
    img=FIG["img"][fi]; y0,x0,h=crop_box(masks,img.shape); sc=S/h
    return dict(label=label, img=jpg(img[y0:y0+h,x0:x0+h]), polys=[poly(m[y0:y0+h,x0:x0+h],sc) for m in masks])
A=AGR.set_index("fig_idx")
def near(cls,val):
    s=AGR[AGR.cls==cls]; return int(s.iloc[(s.dice-val).abs().argsort().iloc[0]].fig_idx)
P=[]
fa=near("meningioma",0.934); P.append(panel(fa,[FIG["mask"][fa],PAIR[fa]],f"Meningioma, Dice {A.dice[fa]:.2f}"))
fb=near("glioma",0.824); P.append(panel(fb,[FIG["mask"][fb],PAIR[fb]],f"Glioma, Dice {A.dice[fb]:.2f}"))
low=AGR[(AGR.dice==0)&(AGR.cls=="glioma")].sort_values("area_fig")
fc=int(low.iloc[len(low)//2].fig_idx); P.append(panel(fc,[FIG["mask"][fc],PAIR[fc]],f"Glioma, Dice {A.dice[fc]:.2f}"))
best=None
for fi,g in dup.groupby("fig_idx"):
    if len(g)<2: continue
    a,b=g.iloc[0],g.iloc[1]; ma=brisc_mask_on_fig(BI[a.brisc],int(fi),int(a.variant)); mb=brisc_mask_on_fig(BI[b.brisc],int(fi),int(b.variant))
    dd=agreement(ma,mb)["dice"]
    if 0.78<dd<0.82 and a.brisc_split!=b.brisc_split: best=(int(fi),ma,mb,dd); break
fd,ma,mb,dd=best; P.append(panel(fd,[FIG["mask"][fd],ma,mb],f"Two BRISC copies, Dice {dd:.2f}"))
OUT={"masks":P}
print("<<<F5"+_j.dumps(OUT)+"F5>>>")
