import json,base64,io,matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; from PIL import Image
from matplotlib.lines import Line2D
d=json.load(open("f5.json"))["masks"]
C=["#1baf7a","#eb6834","#e87ba4"]
fig,axs=plt.subplots(1,4,figsize=(9,2.9))
for ax,p,t in zip(axs,d,"abcd"):
    im=Image.open(io.BytesIO(base64.b64decode(p["img"]))); ax.imshow(im,cmap="gray",extent=(0,180,180,0))
    for k,polys in enumerate(p["polys"]):
        for pl in polys:
            xs=[q[0] for q in pl]+[pl[0][0]]; ys=[q[1] for q in pl]+[pl[0][1]]
            ax.plot(xs,ys,color=C[k],lw=1.6,ls="-" if k<2 else "--")
    ax.set_title(f"({t}) {p['label']}",fontsize=8.5,loc="left"); ax.axis("off")
h=[Line2D([],[],color=C[0],lw=2,label="Figshare mask"),Line2D([],[],color=C[1],lw=2,label="BRISC mask"),Line2D([],[],color=C[2],lw=2,ls="--",label="Second BRISC copy (d)")]
fig.legend(handles=h,loc="lower center",ncol=3,frameon=False,fontsize=8.5,bbox_to_anchor=(.5,-.02))
fig.tight_layout(rect=(0,.08,1,1)); fig.savefig("/mnt/project-files/clahe-brisc/figures/Fig5_mask_examples.png",dpi=300); fig.savefig("../up/Fig5_mask_examples.png",dpi=300)
