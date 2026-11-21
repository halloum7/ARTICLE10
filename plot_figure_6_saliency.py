import json, base64, io, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

D = json.load(open(sys.argv[1]))["sal"]
S = 180
G, O = "#1baf7a", "#eb6834"
names = {"gradcam": "Grad-CAM", "gradcampp": "Grad-CAM++", "intgrad": "Integrated Gradients"}
cls_name = {"meningioma": "Meningioma", "glioma": "Glioma", "pituitary": "Pituitary"}
fig, ax = plt.subplots(len(D), 4, figsize=(10, 2.6 * len(D) + 0.4))
ax = np.atleast_2d(ax)
for r, d in enumerate(D):
    img = np.array(Image.open(io.BytesIO(base64.b64decode(d["img"]))).convert("L"))
    panels = [None, "gradcam", "gradcampp", "intgrad"]
    for c, k in enumerate(panels):
        a = ax[r, c]
        a.imshow(img, cmap="gray", extent=(0, S, S, 0))
        if k:
            n = int(round(len(base64.b64decode(d["maps"][k])) ** 0.5))
            m = np.frombuffer(base64.b64decode(d["maps"][k]), np.uint8).reshape(n, n) / 255.0
            m = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize((S, S), Image.BICUBIC)) / 255.0
            rgba = plt.get_cmap("jet")(m); rgba[..., 3] = np.clip(m, 0, 1) * 0.6
            a.imshow(rgba, extent=(0, S, S, 0))
            a.set_title(f"{names[k]}\nrel. F {d['sc'][k]:.2f} · B {d['sc2'][k]:.2f}", fontsize=9)
        else:
            a.set_title(f"{cls_name[d['cls']]} (input)", fontsize=9)
        for polys, col in zip(d["polys"], [G, O]):
            for p in polys:
                a.add_patch(Polygon(np.array(p), closed=True, fill=False, ec=col, lw=1.3))
        a.set_xlim(0, S); a.set_ylim(S, 0); a.axis("off")
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], c=G, lw=2), Line2D([], [], c=O, lw=2)], ["Figshare mask", "BRISC mask"],
           loc="lower center", ncol=2, frameon=False, fontsize=9)
plt.tight_layout(rect=(0, 0.04, 1, 1))
plt.savefig(sys.argv[2], dpi=300)
