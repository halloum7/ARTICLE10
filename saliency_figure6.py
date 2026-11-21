# Figure 6 (saliency examples).
# Run in the same Python session after src/pipeline.py (or the notebook), which defines
# FIG, PAIR, TE, M_FIG, M_BR, apply_condition, train_one, xai_scores, ArrDS and the XAI imports.
# Prints a JSON block between <<<F6 and F6>>> that src/plot_figure_6_saliency.py turns into the figure.

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
# saliency: ResNet50, no CLAHE, seed 0
Fx=np.stack([apply_condition(i, None) for i in FIG["img"]])
m=train_one("resnet50", Fx, 0)
tl=[target_layer(m,"resnet50")]; cams={"gradcam":GradCAM(model=m,target_layers=tl),"gradcampp":GradCAMPlusPlus(model=m,target_layers=tl)}; ig=IntegratedGradients(m)
SAL=[]
for c in range(3):
    for i in TE:
        i=int(i)
        if FIG["y"][i]!=c or i not in M_BR: continue
        x=ArrDS(Fx[[i]],FIG["y"][[i]])[0][0].unsqueeze(0).to(DEVICE)
        with torch.no_grad(): pred=int(m(x).argmax(1))
        if pred!=c: continue
        if M_FIG[i].mean()<0.01: continue
        maps={k:cc(input_tensor=x,targets=[ClassifierOutputTarget(pred)])[0] for k,cc in cams.items()}
        att=ig.attribute(x,baselines=torch.zeros_like(x),target=pred,n_steps=IG_STEPS,internal_batch_size=IG_STEPS)
        maps["intgrad"]=cv2.GaussianBlur(att.abs().sum(1)[0].detach().cpu().numpy(),(0,0),2)
        enc={}
        for k,s in maps.items():
            s=np.clip(s,0,None); s=cv2.resize(s,(40,40),interpolation=cv2.INTER_AREA); s=(255*s/max(s.max(),1e-9)).astype(np.uint8)
            enc[k]=base64.b64encode(s.tobytes()).decode()
        sc=S/IMG_SIZE
        SAL.append(dict(cls=CLASSES[c],img=jpg(Fx[i]),maps=enc,polys=[poly(M_FIG[i],sc),poly(M_BR[i],sc)],
                        sc={k:round(float(xai_scores(v,M_FIG[i])["relevance"]),2) for k,v in maps.items()}))
        break
print("<<<F6"+_j.dumps({"sal":SAL})+"F6>>>")
