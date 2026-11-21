# %% [md]
# # Doublons BRISC 2025 / Figshare, accord entre deux annotations expertes, et effet du CLAHE
# Partie A : audit des doublons (pHash + corrélation de pixels), fuites dans le découpage officiel de BRISC.
# Partie B : accord entre les masques Figshare et BRISC sur les mêmes coupes (Dice, IoU, HD95, surfaces).
# Partie C : effet du CLAHE (4 conditions × 3 modèles) sur Figshare découpé par patient ; calibration ;
#            fidélité des explications mesurée avec le masque Figshare ET le masque BRISC.
# Résultats : dossier /content/CLAHE_BRISC (zip téléchargé à la fin).

# %%
SMOKE_TEST   = False
SEEDS        = [0, 1, 2]
MODELS       = ["resnet50", "densenet121", "efficientnet_b0"]
CONDITIONS   = {"none": None, "clahe_c1": 1.0, "clahe_c2": 2.0, "clahe_c4": 4.0}
CLAHE_TILES  = (8, 8)
IMG_SIZE     = 224
EPOCHS       = 12
BATCH        = 32
LR           = 1e-4
DUP_MAXDIST  = 6       # distance de Hamming pHash (64 bits) : candidat doublon
DUP_MINCORR  = 0.95    # corrélation de pixels minimale pour confirmer un doublon
IG_STEPS     = 32
CLASSES      = ["meningioma", "glioma", "pituitary"]
USE_DRIVE    = False

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "grad-cam", "captum", "imagehash", "h5py", "kagglehub", "statsmodels"], check=True)

# %%
import os, glob, zipfile, json, random, time, shutil, warnings, urllib.request
import numpy as np, pandas as pd, cv2, h5py, imagehash
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, cohen_kappa_score
from scipy.stats import wilcoxon, spearmanr
from scipy.ndimage import distance_transform_edt, binary_erosion
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")
try:
    display
except NameError:
    display = print

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Appareil :", DEVICE)
ROOT = os.path.abspath("CLAHE_BRISC")
DATA = "/content/data" if os.path.isdir("/content") else os.path.abspath("data")
for d in [ROOT, f"{ROOT}/results", f"{ROOT}/figures", DATA]:
    os.makedirs(d, exist_ok=True)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# %% [md]
# ## 1. Téléchargement

# %%
FIG_DIR = f"{DATA}/figshare"
if "fig_mats" not in globals():
    if not glob.glob(f"{FIG_DIR}/**/*.mat", recursive=True):
        os.makedirs(FIG_DIR, exist_ok=True)
        r = json.load(urllib.request.urlopen("https://api.figshare.com/v2/articles/1512427"))
        for f in r["files"]:
            if not f["name"].endswith(".zip"):
                continue
            out = f"{DATA}/{f['name']}"
            if not (os.path.exists(out) and os.path.getsize(out) == f["size"]):
                subprocess.run(["wget", "-q", "-O", out, f["download_url"]], check=True)
            zipfile.ZipFile(out).extractall(FIG_DIR)
    fig_mats = sorted(glob.glob(f"{FIG_DIR}/**/*.mat", recursive=True))
    fig_mats = [m for m in fig_mats if os.path.basename(m).split(".")[0].isdigit()]
    import kagglehub
    BRISC_DIR = kagglehub.dataset_download("briscdataset/brisc2025")
    seg_imgs = sorted(glob.glob(f"{BRISC_DIR}/**/segmentation_task/*/images/*.jpg", recursive=True))
print("Figshare .mat :", len(fig_mats), " BRISC images avec masque :", len(seg_imgs))

# %% [md]
# ## 2. Lecture des données

# %%
CODE2CLS = {"me": "meningioma", "gl": "glioma", "pi": "pituitary"}

def to_uint8(img):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [0.5, 99.5])
    return (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)

def read_figshare(path):
    with h5py.File(path, "r") as f:
        cj = f["cjdata"]
        img = np.array(cj["image"]).T
        mask = np.array(cj["tumorMask"]).T.astype(np.uint8)
        label = int(np.array(cj["label"]).flatten()[0])
        pid = "".join(chr(int(c)) for c in np.array(cj["PID"]).flatten())
    return to_uint8(img), mask, label - 1, pid, os.path.basename(path)

mask_of = lambda p: p.replace(f"{os.sep}images{os.sep}", f"{os.sep}masks{os.sep}").rsplit(".", 1)[0] + ".png"

def read_brisc(path):
    img = to_uint8(np.array(Image.open(path).convert("L")))
    mask = (np.array(Image.open(mask_of(path)).convert("L")) > 127).astype(np.uint8)
    if mask.shape != img.shape:
        mask = cv2.resize(mask, img.shape[::-1], interpolation=cv2.INTER_NEAREST)
    parts = os.path.basename(path).split("_")
    return img, mask, CLASSES.index(CODE2CLS[parts[3]]), parts[4], parts[1], os.path.basename(path)

def objarr(lst):
    a = np.empty(len(lst), dtype=object); a[:] = lst; return a

if "FIG" not in globals():
    fm = fig_mats[:150] if SMOKE_TEST else fig_mats
    rows = [read_figshare(p) for p in tqdm(fm, desc="Figshare")]
    FIG = dict(img=objarr([r[0] for r in rows]), mask=objarr([r[1] for r in rows]),
               y=np.array([r[2] for r in rows]), pid=np.array([r[3] for r in rows]), name=np.array([r[4] for r in rows]))
    sel = [p for p in seg_imgs if os.path.basename(p).split("_")[3] in CODE2CLS and os.path.exists(mask_of(p))]
    if SMOKE_TEST: sel = sel[:150]
    rows = [read_brisc(p) for p in tqdm(sel, desc="BRISC")]
    BR = dict(img=objarr([r[0] for r in rows]), mask=objarr([r[1] for r in rows]), y=np.array([r[2] for r in rows]),
              plane=np.array([r[3] for r in rows]), split=np.array([r[4] for r in rows]), name=np.array([r[5] for r in rows]))
    del rows
print("Figshare :", len(FIG["y"]), "coupes,", len(set(FIG["pid"])), "patients ; BRISC tumeurs :", len(BR["y"]))

# %% [md]
# ## Partie A — Audit des doublons BRISC / Figshare

# %%
def phash_bits(img):
    return np.array(imagehash.phash(Image.fromarray(cv2.resize(img, (256, 256)))).hash, dtype=bool).flatten()

VARIANTS = [lambda a: a, np.fliplr, np.flipud, lambda a: np.flipud(np.fliplr(a))]
POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
Fp = np.packbits(np.array([[phash_bits(v(im)) for v in VARIANTS] for im in tqdm(FIG["img"], desc="pHash Figshare")]).reshape(-1, 64), axis=1)
Bp = np.packbits(np.array([phash_bits(im) for im in tqdm(BR["img"], desc="pHash BRISC")]), axis=1)

def pix_corr(a, b):
    a = cv2.resize(a, (128, 128)).astype(np.float32).ravel(); b = cv2.resize(b, (128, 128)).astype(np.float32).ravel()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0

audit = []
for bi in tqdm(range(len(BR["y"])), desc="Appariement"):
    d = POP[np.bitwise_xor(Fp, Bp[bi])].sum(1)
    # candidats proches en pHash, puis choix final par corrélation de pixels (évite les ex aequo)
    cand = np.argsort(d, kind="stable")[:10]
    cand = [int(c) for c in cand if d[c] <= DUP_MAXDIST] or [int(cand[0])]
    cors = [pix_corr(BR["img"][bi], VARIANTS[c % len(VARIANTS)](FIG["img"][c // len(VARIANTS)])) for c in cand]
    j = cand[int(np.argmax(cors))]; fi, v = divmod(j, len(VARIANTS)); bestcorr = max(cors)
    audit.append(dict(brisc=BR["name"][bi], brisc_split=BR["split"][bi], brisc_plane=BR["plane"][bi],
                      brisc_class=CLASSES[BR["y"][bi]], fig_idx=fi, fig_name=FIG["name"][fi], fig_pid=FIG["pid"][fi],
                      fig_class=CLASSES[FIG["y"][fi]], variant=v, hamming=int(d[j]), pixel_corr=bestcorr))
AUD = pd.DataFrame(audit)
AUD["duplicate"] = (AUD.hamming <= DUP_MAXDIST) & (AUD.pixel_corr >= DUP_MINCORR)
AUD["label_agree"] = AUD.brisc_class == AUD.fig_class
AUD.to_csv(f"{ROOT}/results/A_audit_pairs.csv", index=False)

dup = AUD[AUD.duplicate]
A = {
    "BRISC tumeurs": len(AUD),
    "BRISC copies de Figshare": int(AUD.duplicate.sum()),
    "% BRISC copies": round(100 * AUD.duplicate.mean(), 1),
    "coupes Figshare distinctes retrouvées": int(dup.fig_idx.nunique()),
    "% Figshare retrouvé": round(100 * dup.fig_idx.nunique() / len(FIG["y"]), 1),
    "patients Figshare concernés": int(dup.fig_pid.nunique()),
    "BRISC non copiées": int((~AUD.duplicate).sum()),
    "doublons : étiquette de classe différente": int((~dup.label_agree).sum()),
}
# doublons internes à BRISC (plusieurs images BRISC -> même coupe Figshare) et fuites train/test officielles
g = dup.groupby("fig_idx")
A["coupes Figshare présentes plusieurs fois dans BRISC"] = int((g.size() > 1).sum())
A["coupes Figshare présentes à la fois dans train ET test BRISC"] = int(g.brisc_split.nunique().gt(1).sum())
A["images test BRISC dont la coupe est aussi dans train BRISC"] = int(dup[dup.brisc_split == "test"].fig_idx.isin(dup[dup.brisc_split == "train"].fig_idx).sum())
# fuite au niveau patient : un patient Figshare présent dans train et test BRISC
gp = dup.groupby("fig_pid").brisc_split.nunique()
A["patients Figshare répartis entre train et test BRISC"] = int((gp > 1).sum())
pd.Series(A).to_csv(f"{ROOT}/results/A_audit_summary.csv")
display(pd.Series(A, name="Audit"))
tabA = pd.crosstab([AUD.brisc_split, AUD.brisc_class], AUD.duplicate, margins=True)
tabA.to_csv(f"{ROOT}/results/A_by_split_class.csv"); display(tabA)
display(pd.crosstab(dup.brisc_class, dup.fig_class))
pd.crosstab(dup.brisc_class, dup.fig_class).to_csv(f"{ROOT}/results/A_label_confusion.csv")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].hist(AUD.hamming, bins=range(0, 34)); ax[0].axvline(DUP_MAXDIST + .5, c="r"); ax[0].set_xlabel("distance pHash minimale à Figshare"); ax[0].set_ylabel("images BRISC")
ax[1].scatter(AUD.hamming, AUD.pixel_corr, s=4, alpha=.3); ax[1].axhline(DUP_MINCORR, c="r"); ax[1].set_xlabel("distance pHash"); ax[1].set_ylabel("corrélation de pixels")
plt.tight_layout(); plt.savefig(f"{ROOT}/figures/A_duplicates.png", dpi=200); plt.show()

# %% [md]
# ## Partie B — Accord entre les deux annotations expertes (Figshare vs BRISC)
# Le masque BRISC est ramené dans la géométrie de la coupe Figshare (retournement inverse + redimensionnement).

# %%
def brisc_mask_on_fig(bi, fi, v):
    m = VARIANTS[v](BR["mask"][bi])            # les retournements sont leurs propres inverses
    h, w = FIG["img"][fi].shape
    return (cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)

def surface_dists(a, b):
    a = a.astype(bool); b = b.astype(bool)
    if not a.any() or not b.any(): return np.array([np.nan])
    sa = a ^ binary_erosion(a); sb = b ^ binary_erosion(b)
    return np.concatenate([distance_transform_edt(~sb)[sa], distance_transform_edt(~sa)[sb]])

def agreement(a, b):
    a = a.astype(bool); b = b.astype(bool); inter = (a & b).sum(); s = a.sum() + b.sum()
    sd = surface_dists(a, b)
    return dict(dice=2 * inter / s if s else np.nan, iou=inter / (a | b).sum() if (a | b).any() else np.nan,
                hd95_px=float(np.nanpercentile(sd, 95)), assd_px=float(np.nanmean(sd)),
                area_fig=int(a.sum()), area_brisc=int(b.sum()))

# un seul appariement par coupe Figshare (le plus proche)
best = dup.sort_values(["hamming", "pixel_corr"], ascending=[True, False]).drop_duplicates("fig_idx")
BI = {n: i for i, n in enumerate(BR["name"])}
PAIR = {}                     # fig_idx -> masque BRISC dans la géométrie Figshare
agr = []
for _, r in tqdm(best.iterrows(), total=len(best), desc="Accord des masques"):
    bi = BI[r.brisc]; fi = int(r.fig_idx)
    mb = brisc_mask_on_fig(bi, fi, int(r.variant)); PAIR[fi] = mb
    agr.append(dict(fig_idx=fi, fig_name=r.fig_name, fig_pid=r.fig_pid, cls=r.fig_class, plane=r.brisc_plane,
                    **agreement(FIG["mask"][fi], mb)))
AGR = pd.DataFrame(agr)
# taille de pixel Figshare : 0,49 mm (512 px) ; on reste en pixels (images 512×512 ou 256×256)
AGR.to_csv(f"{ROOT}/results/B_mask_agreement.csv", index=False)
sumB = AGR.groupby("cls")[["dice", "iou", "hd95_px", "assd_px"]].describe(percentiles=[.25, .5, .75]).round(3)
sumB.to_csv(f"{ROOT}/results/B_agreement_by_class.csv"); display(AGR[["dice", "iou", "hd95_px", "assd_px"]].describe().round(3)); display(sumB)
print("Coupes avec Dice < 0,5 :", int((AGR.dice < .5).sum()), "/", len(AGR))

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for c in CLASSES:
    ax[0].hist(AGR[AGR.cls == c].dice.dropna(), bins=40, alpha=.5, label=c)
ax[0].set_xlabel("Dice Figshare vs BRISC"); ax[0].legend()
m_ = (AGR.area_fig + AGR.area_brisc) / 2; d_ = AGR.area_brisc - AGR.area_fig
ax[1].scatter(m_, d_, s=3, alpha=.3); ax[1].axhline(d_.mean(), c="k")
for k in (-1.96, 1.96): ax[1].axhline(d_.mean() + k * d_.std(), c="r", ls="--")
ax[1].set_xlabel("surface moyenne (px)"); ax[1].set_ylabel("BRISC − Figshare (px)"); ax[1].set_title("Bland-Altman des surfaces")
ex = AGR.sort_values("dice").iloc[[0, len(AGR) // 4, len(AGR) // 2]].fig_idx.values
ax[2].axis("off")
plt.tight_layout(); plt.savefig(f"{ROOT}/figures/B_agreement.png", dpi=200); plt.show()
fig, ax = plt.subplots(1, len(ex), figsize=(4 * len(ex), 4))
for a, fi in zip(np.atleast_1d(ax), ex):
    a.imshow(FIG["img"][fi], cmap="gray"); a.contour(FIG["mask"][fi], [.5], colors="lime"); a.contour(PAIR[fi], [.5], colors="red")
    a.set_title(f"Dice={AGR.set_index('fig_idx').dice[fi]:.2f} (vert Figshare, rouge BRISC)", fontsize=8); a.axis("off")
plt.tight_layout(); plt.savefig(f"{ROOT}/figures/B_examples.png", dpi=200); plt.show()

# %% [md]
# ## Partie C — Effet du CLAHE (Figshare, découpage par patient)

# %%
def patient_split(y, groups, seed=0):
    folds = list(StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))
    te, va = folds[0][1], folds[1][1]
    return np.setdiff1d(np.arange(len(y)), np.concatenate([te, va])), va, te
TR, VA, TE = patient_split(FIG["y"], FIG["pid"])
assert not set(FIG["pid"][TR]) & set(FIG["pid"][TE])
print("train/val/test :", len(TR), len(VA), len(TE), "; test avec masque BRISC :", sum(int(i) in PAIR for i in TE))

def apply_condition(img, clip):
    if clip is not None:
        img = cv2.createCLAHE(clipLimit=clip, tileGridSize=CLAHE_TILES).apply(img)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
rs = lambda m: cv2.resize(m.astype(np.uint8), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
M_FIG = {int(i): rs(FIG["mask"][i]) for i in TE}
M_BR = {int(i): rs(PAIR[int(i)]) for i in TE if int(i) in PAIR}
BR_CLEAN = np.where(~AUD.duplicate.values)[0]          # images BRISC non copiées (test complémentaire)

MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
class ArrDS(torch.utils.data.Dataset):
    def __init__(self, X, y, train=False):
        self.X, self.y, self.train = X, y, train
        self.aug = T.Compose([T.RandomHorizontalFlip(), T.RandomRotation(10), T.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0), antialias=True)])
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).float().div(255).unsqueeze(0).repeat(3, 1, 1)
        if self.train: x = self.aug(x)
        return (x - MEAN) / STD, int(self.y[i])

def build(name, pretrained=True):
    W = "DEFAULT" if pretrained else None
    if name == "resnet50":
        m = torchvision.models.resnet50(weights=W); m.fc = nn.Linear(m.fc.in_features, 3)
    elif name == "densenet121":
        m = torchvision.models.densenet121(weights=W); m.classifier = nn.Linear(m.classifier.in_features, 3)
    else:
        m = torchvision.models.efficientnet_b0(weights=W); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 3)
    return m
def target_layer(m, name):
    return m.layer4[-1] if name == "resnet50" else m.features[-1]

@torch.no_grad()
def logits_of(m, X, bs=64):
    m.eval(); out = []
    for xb, _ in torch.utils.data.DataLoader(ArrDS(X, np.zeros(len(X))), batch_size=bs):
        with torch.autocast("cuda", enabled=DEVICE == "cuda"):
            out.append(m(xb.to(DEVICE)).float().cpu())
    return torch.cat(out).numpy()

def train_one(name, Fx, seed):
    set_seed(seed)
    m = build(name, pretrained=not SMOKE_TEST or PRETRAINED_IN_SMOKE).to(DEVICE)
    y = FIG["y"]
    w = torch.tensor(len(TR) / (3 * np.maximum(np.bincount(y[TR], minlength=3), 1)), dtype=torch.float32, device=DEVICE)
    dl = torch.utils.data.DataLoader(ArrDS(Fx[TR], y[TR], train=True), batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    ep = 1 if SMOKE_TEST else EPOCHS
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=ep * max(len(dl), 1))
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE == "cuda")
    best, state = -1, None
    for e in range(ep):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.autocast("cuda", enabled=DEVICE == "cuda"):
                loss = F.cross_entropy(m(xb), yb, weight=w)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        f1 = f1_score(y[VA], logits_of(m, Fx[VA]).argmax(1), average="macro")
        if f1 > best: best, state = f1, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        print(f"  {name} époque {e+1}/{ep} F1 val={f1:.3f}")
    m.load_state_dict(state); return m.eval()
PRETRAINED_IN_SMOKE = True

softmax = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
def fit_temperature(logits, y):
    t = torch.zeros(1, requires_grad=True); z, yy = torch.tensor(logits), torch.tensor(y)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)
    def clos():
        opt.zero_grad(); l = F.cross_entropy(z / t.exp(), yy); l.backward(); return l
    opt.step(clos); return float(t.exp())
def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1); e = 0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any(): e += s.mean() * abs((pred[s] == y[s]).mean() - conf[s].mean())
    return e
def metrics(p, y):
    pred = p.argmax(1)
    try: auc = roc_auc_score(y, p, multi_class="ovr", average="macro", labels=[0, 1, 2])
    except ValueError: auc = np.nan
    return dict(acc=accuracy_score(y, pred), f1=f1_score(y, pred, average="macro", labels=[0, 1, 2]), auc=auc,
                ece=ece(p, y), brier=((p - np.eye(3)[y]) ** 2).sum(1).mean(), nll=-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
def boot_ci(p, y, key, n=1000):
    rng = np.random.default_rng(0)
    v = [metrics(p[i], y[i])[key] for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.nanpercentile(v, [2.5, 97.5])

TOL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
def xai_scores(sal, mask):
    sal = np.clip(sal, 0, None); s = sal.sum(); m = mask.astype(bool)
    if s <= 0 or not m.any(): return dict(energy=np.nan, relevance=np.nan, pointing=np.nan, iou=np.nan)
    energy = sal[m].sum() / s; b = (sal / sal.max()) >= 0.5
    return dict(energy=energy, relevance=energy / m.mean(), pointing=float(cv2.dilate(mask.astype(np.uint8), TOL).astype(bool).flat[np.argmax(sal)]),
                iou=(b & m).sum() / max((b | m).sum(), 1))

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from captum.attr import IntegratedGradients

rows, xrows, PRED = [], [], {}
yT = FIG["y"][TE]
for cond, clip in CONDITIONS.items():
    Fx = np.stack([apply_condition(i, clip) for i in FIG["img"]])
    Bc = np.stack([apply_condition(BR["img"][i], clip) for i in BR_CLEAN]) if len(BR_CLEAN) else None
    for name in MODELS:
        for seed in SEEDS:
            t0 = time.time()
            m = train_one(name, Fx, seed)
            zV, zT = logits_of(m, Fx[VA]), logits_of(m, Fx[TE]); temp = fit_temperature(zV, FIG["y"][VA])
            evals = [("figshare_test", zT, yT)]
            if Bc is not None and len(BR_CLEAN) > 10:
                evals.append(("brisc_clean", logits_of(m, Bc), BR["y"][BR_CLEAN]))
            for split, z, y in evals:
                for cal, tt in [("raw", 1.0), ("temp_scaled", temp)]:
                    p = softmax(z / tt); r = dict(model=name, condition=cond, seed=seed, split=split, calibration=cal, T=tt, **metrics(p, y))
                    if cal == "raw":
                        for k in ["acc", "f1", "auc"]:
                            r[f"{k}_lo"], r[f"{k}_hi"] = boot_ci(p, y, k, n=100 if SMOKE_TEST else 1000)
                    rows.append(r)
                PRED[f"{name}|{cond}|{seed}|{split}"] = softmax(z / temp)
            if seed == SEEDS[0]:
                tl = [target_layer(m, name)]
                cams = {"gradcam": GradCAM(model=m, target_layers=tl), "gradcampp": GradCAMPlusPlus(model=m, target_layers=tl)}
                ig = IntegratedGradients(m)
                for i in tqdm(TE, desc=f"XAI {name}/{cond}"):
                    i = int(i); x = ArrDS(Fx[[i]], FIG["y"][[i]])[0][0].unsqueeze(0).to(DEVICE)
                    with torch.no_grad(): pred = int(m(x).argmax(1))
                    maps = {k: c(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0] for k, c in cams.items()}
                    att = ig.attribute(x, baselines=torch.zeros_like(x), target=pred, n_steps=IG_STEPS, internal_batch_size=IG_STEPS)
                    maps["intgrad"] = cv2.GaussianBlur(att.abs().sum(1)[0].detach().cpu().numpy(), (0, 0), 2)
                    for meth, s in maps.items():
                        for ref, M in [("figshare", M_FIG), ("brisc", M_BR)]:
                            if i in M:
                                xrows.append(dict(model=name, condition=cond, method=meth, img=i, y=int(FIG["y"][i]), pred=pred,
                                                  correct=int(pred == FIG["y"][i]), ref=ref, **xai_scores(s, M[i])))
                del cams, ig
            print(f"{name}/{cond}/s{seed} : {time.time()-t0:.0f}s"); del m; torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(f"{ROOT}/results/C_classification_calibration.csv", index=False)
    pd.DataFrame(xrows).to_csv(f"{ROOT}/results/C_xai_two_annotators.csv", index=False)
RES, XAI = pd.DataFrame(rows), pd.DataFrame(xrows)
display(RES[RES.calibration == "raw"].groupby(["split", "model", "condition"])[["acc", "f1", "auc", "ece"]].mean().round(3))

# %% [md]
# ### Tests statistiques et effet de l'annotateur

# %%
st = []
for name in MODELS:
    base = PRED[f"{name}|none|{SEEDS[0]}|figshare_test"].argmax(1) == yT
    for cond in [c for c in CONDITIONS if c != "none"]:
        cur = PRED[f"{name}|{cond}|{SEEDS[0]}|figshare_test"].argmax(1) == yT
        tbl = [[np.sum(base & cur), np.sum(base & ~cur)], [np.sum(~base & cur), np.sum(~base & ~cur)]]
        st.append(dict(family="accuracy", model=name, method="-", ref="-", condition=cond, delta=cur.mean() - base.mean(),
                       p=mcnemar(tbl, exact=False, correction=True).pvalue))
for (name, meth, ref), g in XAI.groupby(["model", "method", "ref"]):
    piv = g.pivot_table(index="img", columns="condition", values="relevance", dropna=False).reindex(columns=list(CONDITIONS))
    for cond in [c for c in CONDITIONS if c != "none"]:
        d = piv[[cond, "none"]].dropna()
        p = wilcoxon(d[cond], d["none"]).pvalue if len(d) > 10 and (d[cond] != d["none"]).any() else np.nan
        st.append(dict(family="xai_relevance", model=name, method=meth, ref=ref, condition=cond, delta=(d[cond] - d["none"]).median(), p=p))
ST = pd.DataFrame(st)
for fam, g in ST.groupby("family"):
    ok = g.p.notna()
    if ok.any(): ST.loc[g.index[ok], "p_holm"] = multipletests(g.p[ok], method="holm")[1]
ST["significatif_5pct"] = ST.p_holm < 0.05
ST.to_csv(f"{ROOT}/results/C_statistical_tests.csv", index=False); display(ST.round(4))

# l'annotateur change-t-il le classement des méthodes d'explication / conditions ?
both = XAI.pivot_table(index=["model", "condition", "method", "img"], columns="ref", values="relevance").dropna()
agg = both.groupby(level=[0, 1, 2]).mean()
rho = spearmanr(agg["figshare"], agg["brisc"]).correlation if len(agg) > 2 else np.nan
gap = (both["figshare"] - both["brisc"])
REF = dict(configs=len(agg), spearman_classement=round(float(rho), 3),
           ecart_median_relevance=round(float(gap.median()), 3),
           p_wilcoxon_ecart=float(wilcoxon(both["figshare"], both["brisc"]).pvalue) if len(both) > 10 else np.nan)
pd.Series(REF).to_csv(f"{ROOT}/results/C_annotator_effect.csv"); display(pd.Series(REF, name="Effet de l'annotateur"))
agg.round(3).to_csv(f"{ROOT}/results/C_xai_by_config_and_annotator.csv")

summary = (RES[(RES.split == "figshare_test") & (RES.calibration == "raw")].groupby(["model", "condition"])[["acc", "f1", "auc", "ece"]].mean()
           .join(XAI[(XAI.method == "gradcam")].pivot_table(index=["model", "condition"], columns="ref", values="relevance").add_prefix("relevance_"))
           .round(3))
summary.to_csv(f"{ROOT}/results/summary.csv"); display(summary)

# %%
os.makedirs("/content/export" if os.path.isdir("/content") else "export", exist_ok=True)
EXP = "/content/export" if os.path.isdir("/content") else "export"
for d in ["results", "figures"]:
    shutil.copytree(f"{ROOT}/{d}", f"{EXP}/{d}", dirs_exist_ok=True)
zp = shutil.make_archive(f"{EXP}/../CLAHE_BRISC_resultats", "zip", EXP)
try:
    from google.colab import files; files.download(zp)
except Exception as e:
    print("zip :", zp)
print("TERMINE")
