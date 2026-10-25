# %% [markdown]
# # PolarFuse-Net — full study
# **Cartesian–Polar Dual-View Network with Cross-View Attention and Hierarchical Heads
# for Papilledema vs Pseudopapilledema vs Normal**
#
# Dataset: https://www.kaggle.com/datasets/shashwatwork/identification-of-pseudopapilledema (1,369 disc-centred fundus images)
#
# | Part | What | Split |
# |---|---|---|
# | **A** | **Comparative study** — PolarFuse-Net vs **10 well-known models** (CNNs + Transformers) | fixed 70/15/15 |
# | **B** | **Ablation study** — **12 variants** of PolarFuse-Net (remove / swap every component) | fixed 70/15/15 |
# | **C** | **5-fold cross-validation** — PolarFuse-Net vs the best baseline, paired t-test | 5 folds |
# | Stats | McNemar test of PolarFuse-Net vs every model, bootstrap 95% CI | — |
#
# Everything (all metrics, all figures, per-model summary images, Excel) → **`results_full_study.zip`**.
#
# ### How to run on Kaggle (takes ~3–4 h on a T4, that is fine)
# 1. Add Data → this dataset. Settings → Accelerator **GPU T4 x1** (or P100). Internet **ON** (to download pretrained weights).
# 2. Click **Save Version → "Save & Run All (Commit)"**. It runs in the background, so it does not stop if you close the tab.
# 3. When it finishes, open the version → **Output** → download `results_full_study.zip`.

# %%
import os, glob, random, time, copy, shutil, math, json, gc
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import cv2
from PIL import Image, ImageDraw, ImageFont

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, cohen_kappa_score, matthews_corrcoef, log_loss)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from scipy.stats import binomtest, ttest_rel

# %% ---------------- Config ----------------
WORK = "/kaggle/working" if os.path.exists("/kaggle/working") else "./outputs"
CFG = dict(
    data_root    = "/kaggle/input",
    img_size     = 224,
    batch_size   = 24,
    epochs       = 40,
    patience     = 10,          # early stopping on val macro-F1
    warmup_ep    = 2,
    lr_head      = 1e-3,
    weight_decay = 1e-4,
    seed         = 42,
    num_workers  = 2,
    embed_dim    = 256,
    supcon_w     = 0.3,
    hier_w       = 0.5,
    supcon_temp  = 0.1,
    n_bootstrap  = 1000,
    n_folds      = 5,
    pretrained   = True,
    run_comparative = True,     # Part A
    run_ablation    = True,     # Part B
    run_cv          = True,     # Part C
    results_dir  = f"{WORK}/results_full_study",
    weights_dir  = f"{WORK}/weights_full_study",
)
RES = CFG["results_dir"]
if os.path.exists(RES): shutil.rmtree(RES)
for d in [RES, f"{RES}/A_comparative", f"{RES}/B_ablation", f"{RES}/C_cross_validation", f"{RES}/data", f"{RES}/summaries", CFG["weights_dir"]]:
    os.makedirs(d, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "")

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
seed_everything(CFG["seed"])

def savefig(path, show=True):
    plt.tight_layout()
    kw = {"pil_kwargs": {"quality": 88}} if path.endswith(".jpg") else {}
    plt.savefig(path, dpi=150, bbox_inches="tight", **kw)
    if show: plt.show()
    plt.close("all")

def free():
    gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

# %% ---------------- Data ----------------
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
def class_of(path):
    for p in reversed([p.lower() for p in path.split(os.sep)][:-1]):
        if p.startswith("pseudo"): return "Pseudopapilledema"
        if p.startswith("papill"): return "Papilledema"
        if p.startswith("normal") or p.startswith("control"): return "Normal"
    return None

files = [f for f in glob.glob(os.path.join(CFG["data_root"], "**", "*"), recursive=True) if f.lower().endswith(IMG_EXT)]
df = pd.DataFrame({"path": files}); df["label_name"] = df.path.map(class_of)
df = df.dropna().drop_duplicates("path").reset_index(drop=True)
CLASSES = ["Normal", "Papilledema", "Pseudopapilledema"]
df["label"] = df.label_name.map({c: i for i, c in enumerate(CLASSES)})
print(len(df), df.label_name.value_counts().to_dict())

# fixed split — identical to the first baseline notebook (same seed, same calls)
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df.label, random_state=CFG["seed"])
val_df, test_df   = train_test_split(temp_df, test_size=0.50, stratify=temp_df.label, random_state=CFG["seed"])
split_tbl = pd.DataFrame({n: d.label_name.value_counts().reindex(CLASSES) for n, d in [("train", train_df), ("val", val_df), ("test", test_df)]})
split_tbl.loc["Total"] = split_tbl.sum(); print(split_tbl); split_tbl.to_csv(f"{RES}/data/split_counts.csv")

fig, ax = plt.subplots(figsize=(5, 3.5)); sns.countplot(x="label_name", data=df, order=CLASSES, ax=ax)
for c in ax.containers: ax.bar_label(c)
ax.set_xlabel(""); ax.set_title("Class distribution"); savefig(f"{RES}/data/class_distribution.png")

# %% ---------------- Polar transform ----------------
S = CFG["img_size"]
def to_polar(img_np):
    """Disc-centred Cartesian image -> polar image (rows = angle 0..360°, cols = radius centre..edge)."""
    h, w = img_np.shape[:2]
    return cv2.warpPolar(img_np, (S, S), (w / 2, h / 2), min(h, w) / 2, cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR)

def from_polar(pol_np, size):
    return cv2.warpPolar(pol_np, (size, size), (size / 2, size / 2), size / 2,
                         cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

fig, ax = plt.subplots(3, 4, figsize=(12, 9))
for r, c in enumerate(CLASSES):
    for k, p in enumerate(df[df.label_name == c].sample(2, random_state=1).path):
        im = np.array(Image.open(p).convert("RGB").resize((S, S)))
        ax[r, 2 * k].imshow(im); ax[r, 2 * k].set_title(f"{c} — Cartesian", fontsize=9)
        ax[r, 2 * k + 1].imshow(to_polar(im)); ax[r, 2 * k + 1].set_title("Polar (y = angle, x = radius)", fontsize=9)
for a in ax.ravel(): a.axis("off")
savefig(f"{RES}/data/cartesian_vs_polar.jpg")

# %% ---------------- Dataset ----------------
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
geo_aug = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
                     T.RandomAffine(0, translate=(0.04, 0.04), scale=(0.92, 1.08)), T.ColorJitter(0.2, 0.2, 0.1)])
norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

class DualViewDS(Dataset):
    """Returns (cartesian, polar, label). Augmentation is done in Cartesian space, polar is derived from it."""
    def __init__(self, d, train): self.p, self.y, self.train = d.path.values, d.label.values, train
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        img = Image.open(self.p[i]).convert("RGB").resize((S, S))
        if self.train: img = geo_aug(img)
        cart = np.array(img)
        return norm(Image.fromarray(cart)), norm(Image.fromarray(to_polar(cart))), int(self.y[i])

def make_loader(d, train):
    return DataLoader(DualViewDS(d, train), batch_size=CFG["batch_size"], shuffle=train,
                      drop_last=train and len(d) > CFG["batch_size"], num_workers=CFG["num_workers"], pin_memory=True)

# %% ---------------- Models ----------------
class CrossViewAttentionFusion(nn.Module):
    """Tokens of each view attend to the other view; a learned gate mixes the two enriched descriptors."""
    def __init__(self, dim, heads=4, n_tok=49):
        super().__init__()
        self.c2p = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
        self.p2c = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.pos_c = nn.Parameter(torch.zeros(1, n_tok, dim)); self.pos_p = nn.Parameter(torch.zeros(1, n_tok, dim))
        self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.Sigmoid())
    def forward(self, tc, tp):
        tc = tc + self.pos_c; tp = tp + self.pos_p
        c = self.n1(tc + self.c2p(tc, tp, tp)[0])       # Cartesian queries Polar
        p = self.n2(tp + self.p2c(tp, tc, tc)[0])       # Polar queries Cartesian
        c, p = c.mean(1), p.mean(1)
        g = self.gate(torch.cat([c, p], 1))
        return g * c + (1 - g) * p

class PolarFuseNet(nn.Module):
    """views: cart | polar | dual ; fusion: concat | cvaf ; hier: hierarchical heads ; share: one backbone for both views"""
    def __init__(self, views="dual", fusion="cvaf", hier=True, share=False, backbone="densenet121", dim=256, pretrained=True):
        super().__init__()
        self.views, self.fusion, self.hier = views, fusion, hier
        mk = lambda: timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        if views in ("cart", "dual"): self.bc = mk()
        if views in ("polar", "dual"): self.bp = self.bc if (share and views == "dual") else mk()
        ch = (self.bc if views != "polar" else self.bp).num_features
        mkp = lambda: nn.Sequential(nn.Conv2d(ch, dim, 1), nn.BatchNorm2d(dim), nn.GELU())
        self.proj_c = mkp() if views != "polar" else None
        self.proj_p = mkp() if views != "cart" else None
        if views == "dual" and fusion == "cvaf": self.fuse = CrossViewAttentionFusion(dim)
        fdim = 2 * dim if (views == "dual" and fusion == "concat") else dim
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(fdim, 3)
        if hier: self.head_bin, self.head_sub = nn.Linear(fdim, 2), nn.Linear(fdim, 2)
        self.proj_con = nn.Sequential(nn.Linear(fdim, fdim), nn.ReLU(), nn.Linear(fdim, 128))
        self.feats = {}

    def _tokens(self, x, backbone, proj, key):
        f = backbone(x)
        if f.requires_grad: f.retain_grad()
        self.feats[key] = f                                   # kept for Grad-CAM
        f = F.adaptive_avg_pool2d(proj(f), 7)
        return f.flatten(2).transpose(1, 2)

    def forward(self, xc, xp):
        if self.views == "cart":    z = self._tokens(xc, self.bc, self.proj_c, "cart").mean(1)
        elif self.views == "polar": z = self._tokens(xp, self.bp, self.proj_p, "polar").mean(1)
        else:
            tc, tp = self._tokens(xc, self.bc, self.proj_c, "cart"), self._tokens(xp, self.bp, self.proj_p, "polar")
            z = torch.cat([tc.mean(1), tp.mean(1)], 1) if self.fusion == "concat" else self.fuse(tc, tp)
        zd = self.drop(z)
        out = {"logits": self.head(zd), "emb": F.normalize(self.proj_con(z), dim=1)}
        if self.hier: out["bin"], out["sub"] = self.head_bin(zd), self.head_sub(zd)
        return out

class Baseline(nn.Module):
    """Any timm model, standard fine-tuning on the Cartesian image only."""
    def __init__(self, name, pretrained=True):
        super().__init__()
        self.net = timm.create_model(name, pretrained=pretrained, num_classes=3)
        self.hier, self.feats = False, {}
        # ViT classifies from the CLS token, so patch tokens at the very last layer get no gradient.
        # For Grad-CAM we take the tokens entering the last block instead (standard ViT Grad-CAM practice).
        self.is_vit = name.startswith(("vit", "deit"))
        if self.is_vit: self.net.blocks[-1].norm1.register_forward_hook(self._vit_hook)
    def _vit_hook(self, m, i, o):
        if o.requires_grad: o.retain_grad()
        self.feats["cart"] = o
    def forward(self, xc, xp=None):
        f = self.net.forward_features(xc)
        if not self.is_vit:
            if f.requires_grad: f.retain_grad()
            self.feats["cart"] = f
        emb = self.net.forward_head(f, pre_logits=True)
        return {"logits": self.net.forward_head(f), "emb": F.normalize(emb.flatten(1), dim=1)}
    def head_params(self): return list(self.net.get_classifier().parameters())

def final_probs(out, hier):
    p = torch.softmax(out["logits"].float(), 1)
    if hier:
        pb, ps = torch.softmax(out["bin"].float(), 1), torch.softmax(out["sub"].float(), 1)
        p = 0.5 * (p + torch.stack([pb[:, 0], pb[:, 1] * ps[:, 0], pb[:, 1] * ps[:, 1]], 1))
    return p

class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., NeurIPS 2020)."""
    def __init__(self, t=0.1): super().__init__(); self.t = t
    def forward(self, z, y):
        eye = torch.eye(len(y), device=z.device, dtype=torch.bool)
        pos = (y[:, None] == y[None, :]) & ~eye
        sim = (z @ z.T / self.t).masked_fill(eye, -1e9)
        logp = sim - torch.logsumexp(sim, 1, keepdim=True)
        n = pos.sum(1); v = n > 0
        return -((logp * pos).sum(1)[v] / n[v]).mean() if v.any() else z.sum() * 0

# %% ---------------- Experiment definitions ----------------
# Part A — 10 comparison models (ImageNet-pretrained, same data / augmentation / schedule / early stopping / flip-TTA)
# lr_bb: backbone learning rate (Transformers & ConvNeXt need a smaller one)
COMPARISON = {
    "VGG16-BN":         dict(kind="base", backbone="vgg16_bn",                      lr_bb=1e-4),
    "ResNet50":         dict(kind="base", backbone="resnet50",                      lr_bb=1e-4),
    "DenseNet121":      dict(kind="base", backbone="densenet121",                   lr_bb=1e-4),
    "InceptionV3":      dict(kind="base", backbone="inception_v3",                  lr_bb=1e-4),
    "MobileNetV3-L":    dict(kind="base", backbone="mobilenetv3_large_100",         lr_bb=2e-4),
    "EfficientNet-B0":  dict(kind="base", backbone="efficientnet_b0",               lr_bb=2e-4),
    "EfficientNet-B3":  dict(kind="base", backbone="efficientnet_b3",               lr_bb=1e-4),
    "ConvNeXt-Tiny":    dict(kind="base", backbone="convnext_tiny",                 lr_bb=5e-5),
    "ViT-B/16":         dict(kind="base", backbone="vit_base_patch16_224",          lr_bb=3e-5),
    "Swin-T":           dict(kind="base", backbone="swin_tiny_patch4_window7_224",  lr_bb=5e-5),
}
PROPOSED = "PolarFuse-Net (proposed)"
FULL_CFG = dict(kind="dual", views="dual", fusion="cvaf", hier=True, supcon=True, share=False, backbone="densenet121", lr_bb=1e-4, tta=True)

# Part B — 12 ablation variants (each removes or swaps ONE thing of the full model)
def V(**kw): c = dict(FULL_CFG); c.update(kw); return c
ABLATION = {
    "B01 Cartesian only":                 V(views="cart",  hier=False, supcon=False),
    "B02 Polar only":                     V(views="polar", hier=False, supcon=False),
    "B03 Dual + Concat":                  V(fusion="concat", hier=False, supcon=False),
    "B04 Dual + CVAF":                    V(hier=False, supcon=False),
    "B05 Dual + CVAF + Hier":             V(supcon=False),
    "B06 Dual + CVAF + SupCon":           V(hier=False),
    "B07 Dual + Concat + Hier + SupCon":  V(fusion="concat"),
    "B08 Full, shared backbone":          V(share=True),
    "B09 Full, ResNet50 backbone":        V(backbone="resnet50"),
    "B10 Full, EfficientNet-B0 backbone": V(backbone="efficientnet_b0", lr_bb=2e-4),
    "B11 Full without TTA":               "no_tta",          # same trained weights as B12, evaluated without TTA
    "B12 Full PolarFuse-Net":             "full",
}

# %% ---------------- Metrics ----------------
RAW_COLS = ["Log Loss", "MCC", "Cohen Kappa", "Pap<->Pseudo confusions", "Params (M)", "Train time (min)",
            "Best epoch", "Inference (ms/img)"]

def overall_metrics(y, prob):
    pred = prob.argmax(1); yb = label_binarize(y, classes=range(3))
    cm = confusion_matrix(y, pred, labels=range(3))
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp; tn = cm.sum() - tp - fp - fn
    spec = tn / np.maximum(tn + fp, 1); npv = tn / np.maximum(tn + fn, 1)
    m = {"Accuracy": accuracy_score(y, pred), "Balanced Accuracy": balanced_accuracy_score(y, pred)}
    for avg in ["macro", "weighted"]:
        p, r, f, _ = precision_recall_fscore_support(y, pred, average=avg, zero_division=0)
        m[f"Precision ({avg})"], m[f"Recall/Sensitivity ({avg})"], m[f"F1 ({avg})"] = p, r, f
    m["Specificity (macro)"] = spec.mean(); m["NPV (macro)"] = npv.mean()
    m["MCC"] = matthews_corrcoef(y, pred); m["Cohen Kappa"] = cohen_kappa_score(y, pred)
    m["AUC-ROC (macro)"] = roc_auc_score(y, prob, multi_class="ovr")
    m["AUC-PR (macro)"] = average_precision_score(yb, prob, average="macro")
    m["Log Loss"] = log_loss(y, np.clip(prob, 1e-7, 1), labels=range(3))
    m["Pap<->Pseudo confusions"] = int(cm[1, 2] + cm[2, 1])
    return m

def per_class_metrics(y, prob):
    pred = prob.argmax(1); yb = label_binarize(y, classes=range(3)); cm = confusion_matrix(y, pred, labels=range(3)); rows = []
    for k, c in enumerate(CLASSES):
        TP = cm[k, k]; FP = cm[:, k].sum() - TP; FN = cm[k].sum() - TP; TN = cm.sum() - TP - FP - FN
        pr, rc = TP / max(TP + FP, 1), TP / max(TP + FN, 1)
        rows.append({"Class": c, "Support": int(cm[k].sum()), "TP": TP, "FP": FP, "FN": FN, "TN": TN,
                     "Precision": pr, "Recall/Sensitivity": rc, "Specificity": TN / max(TN + FP, 1),
                     "NPV": TN / max(TN + FN, 1), "F1": 2 * pr * rc / max(pr + rc, 1e-12),
                     "FPR": FP / max(FP + TN, 1), "FNR": FN / max(FN + TP, 1),
                     "AUC-ROC": roc_auc_score(yb[:, k], prob[:, k]), "AUC-PR": average_precision_score(yb[:, k], prob[:, k])})
    return pd.DataFrame(rows)

def bootstrap_ci(y, prob, n, seed=0):
    rng = np.random.default_rng(seed); keys = ["Accuracy", "F1 (macro)", "AUC-ROC (macro)", "Cohen Kappa"]; v = {k: [] for k in keys}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 3: continue
        m = overall_metrics(y[i], prob[i]); [v[k].append(m[k]) for k in keys]
    return {k: (np.percentile(x, 2.5), np.percentile(x, 97.5)) for k, x in v.items()}

def mcnemar(y, p_a, p_b):
    a, b = p_a.argmax(1) == y, p_b.argmax(1) == y
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    return n01, n10, (binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0)

def to_pct_table(rows):
    t = pd.DataFrame(rows).set_index("Model")
    pc = [c for c in t.columns if c not in RAW_COLS]
    t[pc] = (t[pc] * 100).round(2)
    for c in RAW_COLS:
        if c in t: t[c] = t[c].astype(float).round(4)
    return t

# %% ---------------- Build / train / predict ----------------
def build(cfg):
    if cfg["kind"] == "base": return Baseline(cfg["backbone"], CFG["pretrained"])
    return PolarFuseNet(cfg["views"], cfg.get("fusion", "cvaf"), cfg["hier"], cfg.get("share", False),
                        cfg["backbone"], CFG["embed_dim"], CFG["pretrained"])

def param_groups(model, cfg):
    if cfg["kind"] == "base":
        hid = {id(p) for p in model.head_params()}
    else:
        hid = {id(p) for n, p in model.named_parameters() if not n.startswith(("bc.", "bp."))}
    bb = [p for p in model.parameters() if id(p) not in hid]; hd = [p for p in model.parameters() if id(p) in hid]
    return [{"params": bb, "lr": cfg["lr_bb"]}, {"params": hd, "lr": CFG["lr_head"]}]

@torch.no_grad()
def predict(model, dl, tta=False):
    model.eval(); P, Y = [], []
    for xc, xp, y in dl:
        xc, xp = xc.to(DEVICE), xp.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            p = final_probs(model(xc, xp), model.hier)
            if tta:  # vertical flip in Cartesian (θ -> -θ) == reversing the angle axis of the polar image
                p = (p + final_probs(model(xc.flip(2), xp.flip(2)), model.hier)) / 2
        P.append(p.float().cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Y), np.concatenate(P)

def inference_ms(model, n=30):
    model.eval(); x = torch.randn(1, 3, S, S, device=DEVICE)
    with torch.no_grad():
        for _ in range(5): model(x, x)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t = time.time()
        for _ in range(n): model(x, x)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
    return (time.time() - t) / n * 1000

def train(cfg, tr_df, va_df, tag):
    seed_everything(CFG["seed"])
    model = build(cfg).to(DEVICE)
    opt = torch.optim.AdamW(param_groups(model, cfg), weight_decay=CFG["weight_decay"])
    tr_dl, va_dl = make_loader(tr_df, True), make_loader(va_df, False)
    steps = CFG["epochs"] * len(tr_dl); warm = CFG["warmup_ep"] * len(tr_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    cnt = np.bincount(tr_df.label, minlength=3)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(len(tr_df) / (3 * cnt), dtype=torch.float32, device=DEVICE), label_smoothing=0.05)
    ce2 = nn.CrossEntropyLoss(); supcon = SupConLoss(CFG["supcon_temp"])
    use_hier, use_sc = cfg.get("hier", False) and cfg["kind"] == "dual", cfg.get("supcon", False) and cfg["kind"] == "dual"
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    hist, best, best_state, best_ep, wait, t0 = [], (-1, 1e9), None, 0, 0, time.time()
    for ep in range(1, CFG["epochs"] + 1):
        model.train(); tl, n, corr, te = 0.0, 0, 0, time.time()
        for xc, xp, y in tr_dl:
            xc, xp, y = xc.to(DEVICE), xp.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                out = model(xc, xp); loss = ce(out["logits"], y)
                if use_hier:
                    loss = loss + CFG["hier_w"] * ce2(out["bin"], (y > 0).long())
                    ab = y > 0
                    if ab.any(): loss = loss + CFG["hier_w"] * ce2(out["sub"][ab], y[ab] - 1)
            if use_sc: loss = loss + CFG["supcon_w"] * supcon(out["emb"].float(), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(y); n += len(y); corr += (out["logits"].argmax(1) == y).sum().item()
        yv, pv = predict(model, va_dl); vm = overall_metrics(yv, pv)
        hist.append(dict(epoch=ep, train_loss=tl / n, train_acc=corr / n, val_loss=vm["Log Loss"], val_acc=vm["Accuracy"], val_f1=vm["F1 (macro)"]))
        print(f"[{tag}] ep {ep:02d} loss {tl/n:.4f} acc {corr/n:.4f} | val acc {vm['Accuracy']:.4f} F1 {vm['F1 (macro)']:.4f} "
              f"loss {vm['Log Loss']:.4f} | {time.time()-te:.0f}s")
        if vm["F1 (macro)"] > best[0] or (vm["F1 (macro)"] == best[0] and vm["Log Loss"] < best[1]):
            best, best_state, best_ep, wait = (vm["F1 (macro)"], vm["Log Loss"]), copy.deepcopy(model.state_dict()), ep, 0
        else:
            wait += 1
            if wait >= CFG["patience"]: print("early stop"); break
    model.load_state_dict(best_state)
    safe = "".join(ch if ch.isalnum() else "_" for ch in tag)
    torch.save(best_state, f"{CFG['weights_dir']}/{safe}.pth")
    info = {"Params (M)": sum(p.numel() for p in model.parameters()) / 1e6, "Train time (min)": (time.time() - t0) / 60,
            "Best epoch": best_ep, "Inference (ms/img)": inference_ms(model)}
    return model, pd.DataFrame(hist), info

# %% ---------------- Figures ----------------
def plot_cm(y, prob, title, out, show=True):
    cm = confusion_matrix(y, prob.argmax(1), labels=range(3)); cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=CLASSES, yticklabels=CLASSES, ax=ax[0])
    sns.heatmap(cmn, annot=True, fmt=".2%", cmap="Blues", cbar=False, vmin=0, vmax=1, xticklabels=CLASSES, yticklabels=CLASSES, ax=ax[1])
    ax[0].set_title(f"{title} — confusion matrix"); ax[1].set_title(f"{title} — normalized")
    for a in ax: a.set_xlabel("Predicted"); a.set_ylabel("True")
    savefig(out, show)

def plot_roc_pr(y, prob, title, out, show=True):
    yb = label_binarize(y, classes=range(3)); fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, c in enumerate(CLASSES):
        f, t, _ = roc_curve(yb[:, k], prob[:, k]); ax[0].plot(f, t, label=f"{c} (AUC={roc_auc_score(yb[:, k], prob[:, k]):.3f})")
        pr, rc, _ = precision_recall_curve(yb[:, k], prob[:, k]); ax[1].plot(rc, pr, label=f"{c} (AP={average_precision_score(yb[:, k], prob[:, k]):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set_title(f"{title} — ROC"); ax[1].set_title(f"{title} — Precision-Recall")
    ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    for a in ax: a.legend(fontsize=8); a.grid(alpha=.3)
    savefig(out, show)

def plot_hist(h, title, out, show=True):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(h.epoch, h.train_loss, "o-", ms=3, label="train loss"); ax[0].plot(h.epoch, h.val_loss, "o-", ms=3, label="val log-loss")
    ax[1].plot(h.epoch, h.train_acc, "o-", ms=3, label="train acc"); ax[1].plot(h.epoch, h.val_acc, "o-", ms=3, label="val acc")
    ax[1].plot(h.epoch, h.val_f1, "o-", ms=3, label="val macro-F1")
    ax[0].set_title(f"{title} — loss"); ax[1].set_title(f"{title} — accuracy / F1")
    for a in ax: a.set_xlabel("epoch"); a.legend(fontsize=8); a.grid(alpha=.3)
    savefig(out, show)

def plot_per_class(pc, title, out, show=True):
    cols = ["Precision", "Recall/Sensitivity", "Specificity", "NPV", "F1", "AUC-ROC"]
    ax = (pc.set_index("Class")[cols] * 100).plot.bar(figsize=(11, 4), rot=0, width=.8)
    ax.set_ylim(0, 105); ax.set_ylabel("%"); ax.set_title(f"{title} — per-class metrics"); ax.legend(ncol=3, fontsize=8, loc="lower right")
    savefig(out, show)

@torch.no_grad()
def plot_tsne(model, title, out, show=True):
    model.eval(); Z, Yl = [], []
    for xc, xp, y in make_loader(test_df, False):
        Z.append(model(xc.to(DEVICE), xp.to(DEVICE))["emb"].float().cpu().numpy()); Yl.append(y.numpy())
    Z, Yl = np.concatenate(Z), np.concatenate(Yl)
    e = TSNE(2, perplexity=min(30, len(Z) - 1), random_state=0, init="pca").fit_transform(Z)
    plt.figure(figsize=(6, 5))
    for k, c in enumerate(CLASSES): plt.scatter(e[Yl == k, 0], e[Yl == k, 1], s=14, label=c, alpha=.8)
    plt.legend(); plt.title(f"{title} — t-SNE (test)"); plt.xticks([]); plt.yticks([]); savefig(out, show)

def _cam_from(feat, model_ref):
    a, g = feat.detach(), feat.grad
    if a.ndim == 3:                                   # ViT tokens (B, N, C)
        npre = getattr(model_ref, "num_prefix_tokens", 1)
        a, g = a[:, npre:], g[:, npre:]; s = int(round(math.sqrt(a.shape[1])))
        a = a.transpose(1, 2).reshape(a.shape[0], -1, s, s); g = g.transpose(1, 2).reshape(g.shape[0], -1, s, s)
    elif a.ndim == 4 and a.shape[-1] > a.shape[1] and a.shape[1] == a.shape[2]:   # Swin NHWC
        a, g = a.permute(0, 3, 1, 2), g.permute(0, 3, 1, 2)
    c = torch.relu((g.mean((2, 3), keepdim=True) * a).sum(1))[0].float().cpu().numpy()
    return cv2.resize((c - c.min()) / (c.max() - c.min() + 1e-8), (S, S))

GC_SAMPLES = test_df.groupby("label_name").sample(2, random_state=CFG["seed"])
def plot_gradcam(model, title, out, show=True):
    model.eval(); dual = isinstance(model, PolarFuseNet) and model.views == "dual"
    rows = 3 if dual else 2; n = len(GC_SAMPLES); fig, ax = plt.subplots(rows, n, figsize=(2.6 * n, 2.8 * rows))
    ref = model.net if isinstance(model, Baseline) else None
    for k, (_, r) in enumerate(GC_SAMPLES.iterrows()):
        img = np.array(Image.open(r.path).convert("RGB").resize((S, S))); pol = to_polar(img)
        xc = norm(Image.fromarray(img))[None].to(DEVICE).requires_grad_(True); xp = norm(Image.fromarray(pol))[None].to(DEVICE).requires_grad_(True)
        with torch.enable_grad():
            o = model(xc, xp); cls = final_probs(o, model.hier).argmax(1).item()
            model.zero_grad(); o["logits"][0, cls].backward()
        ok = CLASSES[cls] == r.label_name
        ax[0, k].imshow(img); ax[0, k].set_title(f"True: {r.label_name}", fontsize=8)
        key = "cart" if "cart" in model.feats else "polar"
        cam = _cam_from(model.feats[key], ref)
        if key == "polar": cam = from_polar(cam.astype(np.float32), S)
        ax[1, k].imshow(img); ax[1, k].imshow(cam, cmap="jet", alpha=.45)
        ax[1, k].set_title(f"{'Cartesian' if key == 'cart' else 'Polar'} CAM | Pred: {CLASSES[cls]}", fontsize=7, color="green" if ok else "red")
        if dual:
            cp = from_polar(_cam_from(model.feats["polar"], None).astype(np.float32), S)
            ax[2, k].imshow(img); ax[2, k].imshow(cp, cmap="jet", alpha=.45); ax[2, k].set_title("Polar CAM (mapped back)", fontsize=7)
        for a in ax[:, k]: a.axis("off")
    plt.suptitle(f"{title} — Grad-CAM", fontweight="bold"); savefig(out, show)

def _font(s):
    try: return ImageFont.load_default(size=s)
    except TypeError: return ImageFont.load_default()

def montage(paths, out, title, ncols=2, width=1400, pad=30):
    ims = [Image.open(p).convert("RGB") for p in paths if p and os.path.exists(p)]
    if not ims: return
    ims = [im.resize((width, int(im.height * width / im.width))) for im in ims]
    rows = [ims[i:i + ncols] for i in range(0, len(ims), ncols)]; hs = [max(i.height for i in r) for r in rows]
    W = ncols * width + (ncols + 1) * pad; H = 110 + sum(hs) + (len(rows) + 1) * pad
    cv = Image.new("RGB", (W, H), "white"); ImageDraw.Draw(cv).text((pad, 30), title, fill="black", font=_font(56))
    y = 110 + pad
    for r, h in zip(rows, hs):
        x = pad
        for im in r: cv.paste(im, (x, y)); x += width + pad
        y += h + pad
    cv.save(out, quality=88, optimize=True)

def safe(s): return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")

def save_all_for_model(name, y, prob, hist, model, outdir, info=None):
    """All per-model figures + predictions + one summary picture."""
    d = f"{outdir}/{safe(name)}"; os.makedirs(d, exist_ok=True)
    pd.DataFrame({"path": test_df.path.values, "true": [CLASSES[i] for i in y], "pred": [CLASSES[i] for i in prob.argmax(1)],
                  **{f"prob_{c}": prob[:, k] for k, c in enumerate(CLASSES)}}).to_csv(f"{d}/predictions_test.csv", index=False)
    plot_cm(y, prob, name, f"{d}/confusion_matrix.png", False)
    plot_roc_pr(y, prob, name, f"{d}/roc_pr.png", False)
    plot_per_class(per_class_metrics(y, prob), name, f"{d}/per_class.png", False)
    figs = [f"{d}/confusion_matrix.png", f"{d}/roc_pr.png", f"{d}/per_class.png"]
    if hist is not None:
        hist.to_csv(f"{d}/history.csv", index=False); plot_hist(hist, name, f"{d}/history.png", False); figs.append(f"{d}/history.png")
    if model is not None:
        try: plot_tsne(model, name, f"{d}/tsne.png", False); figs.append(f"{d}/tsne.png")
        except Exception as e: print("t-SNE skipped:", e)
        try: plot_gradcam(model, name, f"{d}/gradcam.jpg", False); figs.append(f"{d}/gradcam.jpg")
        except Exception as e: print("Grad-CAM skipped:", e)
    montage(figs, f"{RES}/summaries/_SUMMARY_{safe(name)}.jpg", f"{name} - test results")
    return d

# %% [markdown]
# ## Train the proposed model once (it is used in Part A, Part B and as reference for the stats)

# %%
test_dl = make_loader(test_df, False)
PROBS, ROWS_A, ROWS_B, PC_ALL, CI = {}, [], [], [], {}

print("=" * 70 + f"\n{PROPOSED}\n" + "=" * 70)
model, hist, info = train(FULL_CFG, train_df, val_df, PROPOSED)
Y_TEST, P_FULL = predict(model, test_dl, tta=True)
_, P_FULL_NOTTA = predict(model, test_dl, tta=False)
FULL_INFO, FULL_HIST = info, hist
print(classification_report(Y_TEST, P_FULL.argmax(1), target_names=CLASSES, digits=4))
save_all_for_model(PROPOSED, Y_TEST, P_FULL, hist, model, f"{RES}/A_comparative", info)
del model; free()

def add_row(rows, name, prob, info, group):
    m = overall_metrics(Y_TEST, prob); m.update(info or {}); m["Model"] = name; rows.append(m)
    PROBS[name] = prob; CI[name] = bootstrap_ci(Y_TEST, prob, CFG["n_bootstrap"])
    pc = per_class_metrics(Y_TEST, prob); pc.insert(0, "Model", name); pc.insert(0, "Study", group); PC_ALL.append(pc)

# %% [markdown]
# ## Part A — Comparative study (10 models + proposed)

# %%
if CFG["run_comparative"]:
    for name, cfg in COMPARISON.items():
        print("\n" + "=" * 70 + f"\n[A] {name}\n" + "=" * 70)
        try:
            model, hist, info = train(cfg, train_df, val_df, name)
        except Exception as e:
            print(f"!! {name} failed: {e}"); free(); continue
        _, prob = predict(model, test_dl, tta=True)   # same flip-TTA as the proposed model (fair comparison)
        print(classification_report(Y_TEST, prob.argmax(1), target_names=CLASSES, digits=4))
        add_row(ROWS_A, name, prob, info, "A")
        save_all_for_model(name, Y_TEST, prob, hist, model, f"{RES}/A_comparative", info)
        del model; free()
    add_row(ROWS_A, PROPOSED, P_FULL, FULL_INFO, "A")

# %% [markdown]
# ## Part B — Ablation study (12 variants)

# %%
if CFG["run_ablation"]:
    for name, cfg in ABLATION.items():
        print("\n" + "=" * 70 + f"\n[B] {name}\n" + "=" * 70)
        if cfg == "full":
            add_row(ROWS_B, name, P_FULL, FULL_INFO, "B"); continue
        if cfg == "no_tta":
            add_row(ROWS_B, name, P_FULL_NOTTA, FULL_INFO, "B"); continue
        try:
            model, hist, info = train(cfg, train_df, val_df, name)
        except Exception as e:
            print(f"!! {name} failed: {e}"); free(); continue
        _, prob = predict(model, test_dl, tta=cfg.get("tta", False))
        print(classification_report(Y_TEST, prob.argmax(1), target_names=CLASSES, digits=4))
        add_row(ROWS_B, name, prob, info, "B")
        save_all_for_model(name, Y_TEST, prob, hist, model, f"{RES}/B_ablation", info)
        del model; free()

# %% ---------------- Tables + statistics for A and B ----------------
def finish_table(rows, tag):
    t = to_pct_table(rows)
    for nme in t.index:
        ci = CI[nme]
        t.loc[nme, "Accuracy 95% CI"] = f"{ci['Accuracy'][0]*100:.2f}–{ci['Accuracy'][1]*100:.2f}"
        t.loc[nme, "F1 95% CI"] = f"{ci['F1 (macro)'][0]*100:.2f}–{ci['F1 (macro)'][1]*100:.2f}"
        t.loc[nme, "AUC 95% CI"] = f"{ci['AUC-ROC (macro)'][0]*100:.2f}–{ci['AUC-ROC (macro)'][1]*100:.2f}"
    ref = PROPOSED if PROPOSED in t.index else "B12 Full PolarFuse-Net"
    for nme in t.index:
        if nme == ref: t.loc[nme, "McNemar p vs proposed"] = np.nan; continue
        n01, n10, p = mcnemar(Y_TEST, PROBS[nme], PROBS[ref])
        t.loc[nme, "McNemar p vs proposed"] = round(p, 4)
        t.loc[nme, "only this model right"] = n01; t.loc[nme, "only proposed right"] = n10
    print(f"\n===== {tag} =====\n" + t.to_string())
    return t

TAB_A = finish_table(ROWS_A, "Part A — comparative study (test)") if ROWS_A else None
TAB_B = finish_table(ROWS_B, "Part B — ablation study (test)") if ROWS_B else None

MAIN = ["Accuracy", "Precision (macro)", "Recall/Sensitivity (macro)", "Specificity (macro)", "F1 (macro)", "AUC-ROC (macro)"]

def study_figures(tab, outdir, title):
    names = list(tab.index)
    # 1 bar chart
    ax = tab[MAIN].plot.bar(figsize=(max(12, 1.2 * len(names)), 5.5), rot=30, width=.85)
    ax.set_ylim(max(0, tab[MAIN].values.min() - 5), 100.5); ax.set_ylabel("%"); ax.set_title(f"{title} — test set")
    ax.legend(ncol=3, fontsize=8, loc="lower right"); ax.grid(axis="y", alpha=.3)
    plt.setp(ax.get_xticklabels(), ha="right"); savefig(f"{outdir}/bar_main_metrics.png")
    # 2 heatmap all metrics
    hm = tab[[c for c in tab.columns if c in MAIN + ["Balanced Accuracy", "Precision (weighted)", "Recall/Sensitivity (weighted)",
                                                   "F1 (weighted)", "NPV (macro)", "AUC-PR (macro)"]]].astype(float)
    hm["MCC ×100"] = tab["MCC"] * 100; hm["Kappa ×100"] = tab["Cohen Kappa"] * 100
    plt.figure(figsize=(14, 0.55 * len(names) + 2.5)); sns.heatmap(hm, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title(f"{title} — all metrics (%)"); plt.xticks(rotation=40, ha="right"); savefig(f"{outdir}/heatmap_all_metrics.png")
    # 3 hard-pair confusions
    plt.figure(figsize=(9, 0.45 * len(names) + 1.5))
    sns.barplot(x=tab["Pap<->Pseudo confusions"].values, y=names, color="#c0504d")
    plt.xlabel("# test images where Papilledema and Pseudopapilledema were confused (lower = better)")
    plt.title(f"{title} — hard-pair confusions"); savefig(f"{outdir}/hard_pair_confusions.png")
    # 4 all confusion matrices
    nc = 4; nr = math.ceil(len(names) / nc); fig, axs = plt.subplots(nr, nc, figsize=(4.4 * nc, 4.1 * nr)); axs = np.array(axs).ravel()
    for a, nme in zip(axs, names):
        sns.heatmap(confusion_matrix(Y_TEST, PROBS[nme].argmax(1), labels=range(3)), annot=True, fmt="d", cmap="Blues", cbar=False, ax=a,
                    xticklabels=["N", "Pap", "Pseudo"], yticklabels=["N", "Pap", "Pseudo"])
        a.set_title(f"{nme}\nAcc {tab.loc[nme, 'Accuracy']:.2f}%", fontsize=9)
    for a in axs[len(names):]: a.axis("off")
    savefig(f"{outdir}/all_confusion_matrices.png")
    # 5 ROC per class, all models
    yb = label_binarize(Y_TEST, classes=range(3)); fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    for k, c in enumerate(CLASSES):
        for nme in names:
            f, t, _ = roc_curve(yb[:, k], PROBS[nme][:, k])
            axs[k].plot(f, t, lw=2.5 if nme in (PROPOSED, "B12 Full PolarFuse-Net") else 1,
                        label=f"{nme} ({roc_auc_score(yb[:, k], PROBS[nme][:, k]):.3f})")
        axs[k].plot([0, 1], [0, 1], "k--"); axs[k].set_title(f"ROC — {c}"); axs[k].legend(fontsize=6, loc="lower right"); axs[k].grid(alpha=.3)
    savefig(f"{outdir}/roc_all_models.png")
    # 6 accuracy vs params
    if "Params (M)" in tab:
        plt.figure(figsize=(8, 5.5)); t = tab.dropna(subset=["Params (M)"])
        plt.scatter(t["Params (M)"], t["Accuracy"], s=80)
        for nme, r in t.iterrows(): plt.annotate(nme, (r["Params (M)"], r["Accuracy"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
        plt.xlabel("Parameters (M)"); plt.ylabel("Test accuracy (%)"); plt.title(f"{title} — accuracy vs model size"); plt.grid(alpha=.3)
        savefig(f"{outdir}/accuracy_vs_params.png")
    # 7 table as image
    show = tab[MAIN + ["Cohen Kappa", "MCC", "Pap<->Pseudo confusions"]]
    fig, ax = plt.subplots(figsize=(17, 0.5 * len(show) + 1.3)); ax.axis("off")
    cells = [[f"{v:.2f}" if isinstance(v, float) and c not in ("Cohen Kappa", "MCC") else (f"{v:.4f}" if isinstance(v, float) else str(v))
              for c, v in r.items()] for _, r in show.iterrows()]
    tb = ax.table(cellText=cells, rowLabels=show.index, colLabels=[c.replace(" (", "\n(") for c in show.columns], loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.9)
    ax.set_title(f"{title} — test set (%; Kappa & MCC on 0–1)", fontweight="bold"); savefig(f"{outdir}/results_table.png")
    figs = ["results_table.png", "bar_main_metrics.png", "heatmap_all_metrics.png", "hard_pair_confusions.png",
            "all_confusion_matrices.png", "roc_all_models.png", "accuracy_vs_params.png"]
    return [f"{outdir}/{f}" for f in figs]

FIGS_A = study_figures(TAB_A, f"{RES}/A_comparative", "Comparative study") if TAB_A is not None else []
FIGS_B = study_figures(TAB_B, f"{RES}/B_ablation", "Ablation study") if TAB_B is not None else []

if TAB_A is not None:   # Grad-CAM of all models together
    montage([f"{RES}/A_comparative/{safe(n)}/gradcam.jpg" for n in TAB_A.index], f"{RES}/A_comparative/all_gradcam.jpg",
            "Grad-CAM - all models (same test images)", ncols=2, width=1500)
    FIGS_A.append(f"{RES}/A_comparative/all_gradcam.jpg")
if TAB_B is not None:
    montage([f"{RES}/B_ablation/{safe(n)}/tsne.png" for n in TAB_B.index] + [f"{RES}/A_comparative/{safe(PROPOSED)}/tsne.png"],
            f"{RES}/B_ablation/all_tsne.jpg", "t-SNE of embeddings - ablation variants", ncols=4, width=900)
    FIGS_B.append(f"{RES}/B_ablation/all_tsne.jpg")

# %% [markdown]
# ## Part C — 5-fold cross-validation: PolarFuse-Net vs best baseline

# %%
if CFG["run_cv"]:
    base_name = "DenseNet121"
    if TAB_A is not None:
        bl = TAB_A.drop(PROPOSED).sort_values(["F1 (macro)", "Accuracy"], ascending=False)
        base_name = bl.index[0]
    CV_MODELS = {PROPOSED: FULL_CFG, f"{base_name} (best baseline)": COMPARISON[base_name]}
    print("CV models:", list(CV_MODELS))
    skf = StratifiedKFold(CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    folds = list(skf.split(df, df.label))
    CV_ROWS, OOF = [], {n: np.zeros((len(df), 3)) for n in CV_MODELS}
    for f, (tr_i, te_i) in enumerate(folds):
        tr_all, te = df.iloc[tr_i], df.iloc[te_i]
        tr, va = train_test_split(tr_all, test_size=0.15, stratify=tr_all.label, random_state=CFG["seed"])
        for nme, cfg in CV_MODELS.items():
            print("\n" + "=" * 70 + f"\n[C] Fold {f+1}/{CFG['n_folds']} — {nme}\n" + "=" * 70)
            model, hist, info = train(cfg, tr, va, f"{nme}_fold{f+1}")
            y, prob = predict(model, make_loader(te, False), tta=True)
            OOF[nme][te_i] = prob
            m = overall_metrics(y, prob); m.update(info); m["Model"] = nme; m["Fold"] = f + 1; CV_ROWS.append(m)
            del model; free()
    cv = pd.DataFrame(CV_ROWS)
    pcols = [c for c in cv.columns if c not in RAW_COLS + ["Model", "Fold"]]
    cvp = cv.copy(); cvp[pcols] *= 100
    summ = []
    for nme, g in cvp.groupby("Model", sort=False):
        r = {"Model": nme}
        for c in pcols + ["MCC", "Cohen Kappa", "Pap<->Pseudo confusions"]:
            r[c] = f"{g[c].mean():.2f} ± {g[c].std():.2f}" if c in pcols else f"{g[c].mean():.4f} ± {g[c].std():.4f}"
        summ.append(r)
    CV_SUMMARY = pd.DataFrame(summ).set_index("Model")
    # paired t-test over folds
    names = list(CV_MODELS); a = cvp[cvp.Model == names[0]].sort_values("Fold"); b = cvp[cvp.Model == names[1]].sort_values("Fold")
    TT = pd.DataFrame([{"Metric": c, f"{names[0]} mean": a[c].mean(), f"{names[1]} mean": b[c].mean(),
                        "paired t-test p": ttest_rel(a[c], b[c]).pvalue} for c in ["Accuracy", "F1 (macro)", "AUC-ROC (macro)", "Cohen Kappa"]])
    OOF_TAB = to_pct_table([dict(overall_metrics(df.label.values, OOF[n]), Model=n) for n in CV_MODELS])
    print(CV_SUMMARY.to_string()); print(TT.to_string()); print(OOF_TAB.to_string())

    C = f"{RES}/C_cross_validation"
    for n in CV_MODELS:
        plot_cm(df.label.values, OOF[n], f"{n} — 5-fold OOF (1,369 images)", f"{C}/oof_cm_{safe(n)}.png")
        plot_roc_pr(df.label.values, OOF[n], f"{n} — 5-fold OOF", f"{C}/oof_roc_{safe(n)}.png")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for k, c in enumerate(["Accuracy", "F1 (macro)", "AUC-ROC (macro)"]):
        for n in CV_MODELS:
            g = cvp[cvp.Model == n].sort_values("Fold"); ax[k].plot(g.Fold, g[c], "o-", label=n)
        ax[k].set_title(f"{c} per fold"); ax[k].set_xlabel("fold"); ax[k].grid(alpha=.3); ax[k].legend(fontsize=8)
    savefig(f"{C}/per_fold.png")
    plt.figure(figsize=(8, 4.5)); sns.boxplot(data=cvp, x="Model", y="F1 (macro)"); sns.stripplot(data=cvp, x="Model", y="F1 (macro)", color="k")
    plt.title("Macro-F1 across 5 folds"); savefig(f"{C}/boxplot_f1.png")
    FIGS_C = [f"{C}/oof_cm_{safe(n)}.png" for n in CV_MODELS] + [f"{C}/oof_roc_{safe(n)}.png" for n in CV_MODELS] + [f"{C}/per_fold.png", f"{C}/boxplot_f1.png"]
else:
    FIGS_C = []

# %% ---------------- Excel, overview images, README, zip ----------------
with pd.ExcelWriter(f"{RES}/ALL_RESULTS_full_study.xlsx") as xw:
    if TAB_A is not None: TAB_A.to_excel(xw, sheet_name="A_comparative_test")
    if TAB_B is not None: TAB_B.to_excel(xw, sheet_name="B_ablation_test")
    if PC_ALL: pd.concat(PC_ALL).to_excel(xw, sheet_name="Per_class_test", index=False)
    if CFG["run_cv"]:
        CV_SUMMARY.to_excel(xw, sheet_name="C_CV_mean_std"); cvp.round(4).to_excel(xw, sheet_name="C_CV_per_fold", index=False)
        TT.to_excel(xw, sheet_name="C_paired_ttest", index=False); OOF_TAB.to_excel(xw, sheet_name="C_OOF_all_images")
    split_tbl.to_excel(xw, sheet_name="Data_split")
    pd.DataFrame([{"Model": k, **v} for k, v in COMPARISON.items()] + [{"Model": PROPOSED, **FULL_CFG}]).to_excel(xw, sheet_name="Model_configs", index=False)
    pd.DataFrame(list({k: str(v) for k, v in CFG.items()}.items()), columns=["param", "value"]).to_excel(xw, sheet_name="Config", index=False)
if TAB_A is not None: TAB_A.to_csv(f"{RES}/TABLE_A_comparative.csv")
if TAB_B is not None: TAB_B.to_csv(f"{RES}/TABLE_B_ablation.csv")
if CFG["run_cv"]: CV_SUMMARY.to_csv(f"{RES}/TABLE_C_cv_mean_std.csv"); TT.to_csv(f"{RES}/TABLE_C_paired_ttest.csv", index=False)

if FIGS_A: montage(FIGS_A, f"{RES}/_OVERVIEW_A_comparative.jpg", "Part A - comparative study (10 models + PolarFuse-Net)", 2, 1600)
if FIGS_B: montage(FIGS_B, f"{RES}/_OVERVIEW_B_ablation.jpg", "Part B - ablation study (12 variants)", 2, 1600)
if FIGS_C: montage(FIGS_C, f"{RES}/_OVERVIEW_C_cross_validation.jpg", "Part C - 5-fold cross-validation", 2, 1600)
montage([f"{RES}/data/cartesian_vs_polar.jpg", f"{RES}/summaries/_SUMMARY_{safe(PROPOSED)}.jpg"],
        f"{RES}/_OVERVIEW_proposed_model.jpg", "PolarFuse-Net - proposed model", 1, 1800)

with open(f"{RES}/README.txt", "w") as fh:
    fh.write("PolarFuse-Net — full study\n" + "=" * 70 + "\n")
    if TAB_A is not None: fh.write("\nPART A — COMPARATIVE (test)\n" + TAB_A[MAIN + ["Cohen Kappa", "MCC", "Pap<->Pseudo confusions", "McNemar p vs proposed"]].to_string() + "\n")
    if TAB_B is not None: fh.write("\nPART B — ABLATION (test)\n" + TAB_B[MAIN + ["Cohen Kappa", "Pap<->Pseudo confusions", "McNemar p vs proposed"]].to_string() + "\n")
    if CFG["run_cv"]: fh.write("\nPART C — 5-FOLD CV\n" + CV_SUMMARY.to_string() + "\n\n" + TT.to_string() + "\n")
    fh.write("\nFiles: _OVERVIEW_*.jpg (one picture per part), summaries/_SUMMARY_<model>.jpg (one picture per model),\n"
             "ALL_RESULTS_full_study.xlsx (every metric), TABLE_*.csv, A_comparative/, B_ablation/, C_cross_validation/\n")

zp = shutil.make_archive(f"{WORK}/results_full_study", "zip", RES)
print(f"\n✅ saved {zp}  ({os.path.getsize(zp)/1e6:.1f} MB) — Output tab → results_full_study.zip")
for f in ["_OVERVIEW_A_comparative.jpg", "_OVERVIEW_B_ablation.jpg"]:
    if os.path.exists(f"{RES}/{f}"):
        plt.figure(figsize=(20, 20)); plt.imshow(Image.open(f"{RES}/{f}")); plt.axis("off"); plt.show()
