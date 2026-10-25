# %% [markdown]
# # PolarFuse-Net: Cartesian–Polar Dual-View Network with Hierarchical Heads for Papilledema vs Pseudopapilledema
# Dataset: https://www.kaggle.com/datasets/shashwatwork/identification-of-pseudopapilledema (1,369 disc-centred fundus images)
#
# **Idea (novelty).** Clinically, true papilledema vs pseudopapilledema is decided by what happens *around the disc
# margin*: blurring of the margin, obscuration of vessels as they cross the margin, peripapillary halo / drusen.
# These signs are **radial and circumferential**. A normal CNN sees them only in Cartesian space.
#
# PolarFuse-Net adds three components on top of a DenseNet121 baseline:
# 1. **Dual-view input**: the Cartesian image + its **polar unwrapping around the disc centre**. In polar space the disc
#    margin becomes a horizontal band, so margin blurring / vessel obscuration becomes a simple row-wise pattern, and a
#    rotation of the eye becomes a circular shift (rotation robustness).
# 2. **Cross-View Attention Fusion (CVAF)**: tokens of each branch attend to the other branch, followed by a learned gate.
# 3. **Hierarchical clinical heads + supervised contrastive loss**: a flat 3-class head, plus a clinically-motivated
#    hierarchy (Normal vs *Elevated disc* → Papilledema vs Pseudopapilledema). A supervised contrastive loss pulls the
#    embeddings of the same class together and pushes Papilledema / Pseudopapilledema apart — this is the part aimed
#    at "not confusing" the two hard classes.
#
# **Experiments in this notebook**
# * Part A — **Ablation study** on the fixed 70/15/15 split (same split/seed as the baseline notebook, so numbers are comparable)
# * Part B — **5-fold stratified cross-validation** of the full model (mean ± std + out-of-fold confusion matrix over all 1,369 images)
# * **McNemar test** full model vs DenseNet121 baseline, Grad-CAM for both views, everything zipped to **results_polarfuse.zip**
#
# On Kaggle: Add Data → this dataset, **GPU T4/P100 on**. Full run ≈ 60–90 min on a T4.

# %%
import os, glob, random, time, copy, shutil, math
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
from scipy.stats import binomtest

# %% ---------------- Config ----------------
WORK = "/kaggle/working" if os.path.exists("/kaggle/working") else "./outputs"
CFG = dict(
    data_root   = "/kaggle/input",
    img_size    = 224,
    batch_size  = 24,
    epochs      = 30,
    lr_backbone = 1e-4,
    lr_head     = 1e-3,
    weight_decay= 1e-4,
    warmup_ep   = 2,
    patience    = 8,
    seed        = 42,
    num_workers = 2,
    backbone    = "densenet121",   # best single model in the baseline run
    embed_dim   = 256,
    supcon_w    = 0.3,             # weight of supervised contrastive loss
    hier_w      = 0.5,             # weight of hierarchical losses
    supcon_temp = 0.1,
    tta         = True,            # test-time augmentation (flips)
    n_bootstrap = 1000,
    run_ablation= True,            # Part A
    run_cv      = True,            # Part B
    n_folds     = 5,
    pretrained  = True,
    results_dir = f"{WORK}/results_polarfuse",
    weights_dir = f"{WORK}/weights_polarfuse",
)
RES = CFG["results_dir"]
if os.path.exists(RES): shutil.rmtree(RES)
for d in [RES, f"{RES}/ablation", f"{RES}/cv", f"{RES}/data", CFG["weights_dir"]]: os.makedirs(d, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
seed_everything(CFG["seed"])

def savefig(path):
    plt.tight_layout()
    kw = {"pil_kwargs": {"quality": 88}} if path.endswith(".jpg") else {}
    plt.savefig(path, dpi=150, bbox_inches="tight", **kw); plt.show(); plt.close("all")

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

# fixed split — identical to the baseline notebook (same seed + same calls)
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df.label, random_state=CFG["seed"])
val_df, test_df   = train_test_split(temp_df, test_size=0.50, stratify=temp_df.label, random_state=CFG["seed"])
print("fixed split:", len(train_df), len(val_df), len(test_df))

# %% ---------------- Polar transform (the second view) ----------------
S = CFG["img_size"]
def to_polar(img_np):
    """Disc-centred Cartesian image -> polar image (rows = angle 0..360°, cols = radius centre..edge)."""
    h, w = img_np.shape[:2]
    return cv2.warpPolar(img_np, (S, S), (w / 2, h / 2), min(h, w) / 2, cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR)

def from_polar(pol_np, size):
    """Inverse polar -> Cartesian (used to map polar Grad-CAM back onto the image)."""
    return cv2.warpPolar(pol_np, (size, size), (size / 2, size / 2), size / 2,
                         cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

# visual check of the two views
fig, ax = plt.subplots(3, 4, figsize=(12, 9))
for r, c in enumerate(CLASSES):
    for k, p in enumerate(df[df.label_name == c].sample(2, random_state=1).path):
        im = np.array(Image.open(p).convert("RGB").resize((S, S)))
        ax[r, 2 * k].imshow(im); ax[r, 2 * k].set_title(f"{c} — Cartesian", fontsize=9)
        ax[r, 2 * k + 1].imshow(to_polar(im)); ax[r, 2 * k + 1].set_title("Polar (y = angle, x = radius)", fontsize=9)
for a in ax.ravel(): a.axis("off")
savefig(f"{RES}/data/cartesian_vs_polar.jpg")

# %% ---------------- Dataset: augment in Cartesian, then derive the polar view ----------------
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
geo_aug = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
                     T.RandomAffine(0, translate=(0.04, 0.04), scale=(0.92, 1.08)),
                     T.ColorJitter(0.2, 0.2, 0.1)])
norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

class DualViewDS(Dataset):
    def __init__(self, d, train): self.p, self.y, self.train = d.path.values, d.label.values, train
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        img = Image.open(self.p[i]).convert("RGB").resize((S, S))
        if self.train: img = geo_aug(img)
        cart = np.array(img)
        pol = to_polar(cart)
        return norm(Image.fromarray(cart)), norm(Image.fromarray(pol)), int(self.y[i])

def make_loader(d, train):
    return DataLoader(DualViewDS(d, train), batch_size=CFG["batch_size"], shuffle=train, drop_last=train and len(d) > CFG["batch_size"],
                      num_workers=CFG["num_workers"], pin_memory=True)

# %% ---------------- Model ----------------
class CrossViewAttentionFusion(nn.Module):
    """Tokens of each view attend to the other view; a learned gate mixes the two enriched descriptors."""
    def __init__(self, dim, heads=4):
        super().__init__()
        self.c2p = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
        self.p2c = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.pos_c = nn.Parameter(torch.zeros(1, 49, dim)); self.pos_p = nn.Parameter(torch.zeros(1, 49, dim))
        self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.Sigmoid())
    def forward(self, tc, tp):                       # (B, N, D) each
        tc = tc + self.pos_c[:, :tc.shape[1]]; tp = tp + self.pos_p[:, :tp.shape[1]]
        c = self.n1(tc + self.c2p(tc, tp, tp)[0])   # Cartesian queries Polar
        p = self.n2(tp + self.p2c(tp, tc, tc)[0])   # Polar queries Cartesian
        c, p = c.mean(1), p.mean(1)
        g = self.gate(torch.cat([c, p], 1))
        return g * c + (1 - g) * p

class PolarFuseNet(nn.Module):
    """
    views  : 'cart' | 'polar' | 'dual'
    fusion : 'concat' | 'cvaf'      (dual only)
    hier   : use hierarchical heads (Normal vs Elevated → Pap vs Pseudo)
    """
    def __init__(self, views="dual", fusion="cvaf", hier=True, backbone="densenet121", dim=256, pretrained=True):
        super().__init__()
        self.views, self.fusion, self.hier = views, fusion, hier
        mk = lambda: timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        if views in ("cart", "dual"):  self.bc = mk()
        if views in ("polar", "dual"): self.bp = mk()
        ch = (self.bc if views != "polar" else self.bp).num_features
        self.proj_c = nn.Sequential(nn.Conv2d(ch, dim, 1), nn.BatchNorm2d(dim), nn.GELU()) if views != "polar" else None
        self.proj_p = nn.Sequential(nn.Conv2d(ch, dim, 1), nn.BatchNorm2d(dim), nn.GELU()) if views != "cart" else None
        if views == "dual" and fusion == "cvaf": self.fuse = CrossViewAttentionFusion(dim)
        fdim = 2 * dim if (views == "dual" and fusion == "concat") else dim
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(fdim, 3)
        if hier:
            self.head_bin = nn.Linear(fdim, 2)   # Normal vs Elevated disc
            self.head_sub = nn.Linear(fdim, 2)   # Papilledema vs Pseudopapilledema
        self.proj_con = nn.Sequential(nn.Linear(fdim, fdim), nn.ReLU(), nn.Linear(fdim, 128))  # for SupCon

    def _tokens(self, x, backbone, proj):
        f = proj(backbone(x))                        # (B, D, h, w)
        f = F.adaptive_avg_pool2d(f, 7)
        return f.flatten(2).transpose(1, 2)          # (B, 49, D)

    def forward(self, xc, xp):
        if self.views == "cart":  z = self._tokens(xc, self.bc, self.proj_c).mean(1)
        elif self.views == "polar": z = self._tokens(xp, self.bp, self.proj_p).mean(1)
        else:
            tc, tp = self._tokens(xc, self.bc, self.proj_c), self._tokens(xp, self.bp, self.proj_p)
            z = torch.cat([tc.mean(1), tp.mean(1)], 1) if self.fusion == "concat" else self.fuse(tc, tp)
        zd = self.drop(z)
        out = {"logits": self.head(zd), "emb": F.normalize(self.proj_con(z), dim=1)}
        if self.hier: out["bin"], out["sub"] = self.head_bin(zd), self.head_sub(zd)
        return out

def final_probs(out, hier):
    """Flat 3-class prob, averaged with the hierarchical prob when hierarchy is on."""
    p = torch.softmax(out["logits"].float(), 1)
    if hier:
        pb, ps = torch.softmax(out["bin"].float(), 1), torch.softmax(out["sub"].float(), 1)
        ph = torch.stack([pb[:, 0], pb[:, 1] * ps[:, 0], pb[:, 1] * ps[:, 1]], 1)
        p = 0.5 * (p + ph)
    return p

class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., 2020)."""
    def __init__(self, t=0.1): super().__init__(); self.t = t
    def forward(self, z, y):
        sim = z @ z.T / self.t
        mask_self = torch.eye(len(y), device=z.device, dtype=torch.bool)
        pos = (y[:, None] == y[None, :]) & ~mask_self
        sim = sim.masked_fill(mask_self, -1e9)
        logp = sim - torch.logsumexp(sim, 1, keepdim=True)
        n_pos = pos.sum(1)
        valid = n_pos > 0
        return -((logp * pos).sum(1)[valid] / n_pos[valid]).mean() if valid.any() else z.sum() * 0

# %% ---------------- Metrics ----------------
def overall_metrics(y, prob):
    pred = prob.argmax(1); yb = label_binarize(y, classes=range(3))
    cm = confusion_matrix(y, pred, labels=range(3))
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp; tn = cm.sum() - tp - fp - fn
    spec = tn / np.maximum(tn + fp, 1); npv = tn / np.maximum(tn + fn, 1)
    m = {"Accuracy": accuracy_score(y, pred), "Balanced Accuracy": balanced_accuracy_score(y, pred)}
    for avg in ["macro", "weighted"]:
        p, r, f, _ = precision_recall_fscore_support(y, pred, average=avg, zero_division=0)
        m[f"Precision ({avg})"], m[f"Recall ({avg})"], m[f"F1 ({avg})"] = p, r, f
    m["Specificity (macro)"] = spec.mean(); m["NPV (macro)"] = npv.mean()
    m["MCC"] = matthews_corrcoef(y, pred); m["Cohen Kappa"] = cohen_kappa_score(y, pred)
    m["AUC-ROC (macro)"] = roc_auc_score(y, prob, multi_class="ovr")
    m["AUC-PR (macro)"] = average_precision_score(yb, prob, average="macro")
    m["Log Loss"] = log_loss(y, np.clip(prob, 1e-7, 1), labels=range(3))
    # how often the two hard classes are confused with each other
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
                     "AUC-ROC": roc_auc_score(yb[:, k], prob[:, k]), "AUC-PR": average_precision_score(yb[:, k], prob[:, k])})
    return pd.DataFrame(rows)

def bootstrap_ci(y, prob, n, seed=0):
    rng = np.random.default_rng(seed); keys = ["Accuracy", "F1 (macro)", "AUC-ROC (macro)", "Cohen Kappa"]; v = {k: [] for k in keys}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 3: continue
        m = overall_metrics(y[i], prob[i]); [v[k].append(m[k]) for k in keys]
    return {k: (np.percentile(x, 2.5), np.percentile(x, 97.5)) for k, x in v.items()}

# %% ---------------- Train / predict ----------------
@torch.no_grad()
def predict(model, dl, hier, tta=False):
    model.eval(); P, Y = [], []
    for xc, xp, y in dl:
        xc, xp = xc.to(DEVICE), xp.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            p = final_probs(model(xc, xp), hier)
            if tta:
                # vertical flip in Cartesian (θ -> -θ) == reversing the angle axis of the polar image
                p = p + final_probs(model(xc.flip(2), xp.flip(2)), hier)
                p = p / 2
        P.append(p.float().cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Y), np.concatenate(P)

def train_one(variant, tr_df, va_df, tag):
    seed_everything(CFG["seed"])
    model = PolarFuseNet(variant["views"], variant.get("fusion", "cvaf"), variant["hier"], CFG["backbone"],
                         CFG["embed_dim"], CFG["pretrained"]).to(DEVICE)
    bb = [p for n, p in model.named_parameters() if n.startswith(("bc.", "bp."))]
    hd = [p for n, p in model.named_parameters() if not n.startswith(("bc.", "bp."))]
    opt = torch.optim.AdamW([{"params": bb, "lr": CFG["lr_backbone"]}, {"params": hd, "lr": CFG["lr_head"]}], weight_decay=CFG["weight_decay"])
    tr_dl, va_dl = make_loader(tr_df, True), make_loader(va_df, False)
    steps = CFG["epochs"] * len(tr_dl); warm = CFG["warmup_ep"] * len(tr_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    cnt = np.bincount(tr_df.label, minlength=3); w = torch.tensor(len(tr_df) / (3 * cnt), dtype=torch.float32, device=DEVICE)
    ce = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    ce_bin = nn.CrossEntropyLoss(); ce_sub = nn.CrossEntropyLoss(); supcon = SupConLoss(CFG["supcon_temp"])
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    hist, best, best_state, best_ep, wait, t0 = [], (-1, 1e9), None, 0, 0, time.time()
    for ep in range(1, CFG["epochs"] + 1):
        model.train(); tl, n = 0.0, 0; te = time.time()
        for xc, xp, y in tr_dl:
            xc, xp, y = xc.to(DEVICE), xp.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                out = model(xc, xp); loss = ce(out["logits"], y)
                if variant["hier"]:
                    loss = loss + CFG["hier_w"] * ce_bin(out["bin"], (y > 0).long())
                    ab = y > 0
                    if ab.any(): loss = loss + CFG["hier_w"] * ce_sub(out["sub"][ab], y[ab] - 1)
            if variant.get("supcon"): loss = loss + CFG["supcon_w"] * supcon(out["emb"].float(), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(y); n += len(y)
        yv, pv = predict(model, va_dl, variant["hier"]); vm = overall_metrics(yv, pv)
        hist.append(dict(epoch=ep, train_loss=tl / n, val_loss=vm["Log Loss"], val_acc=vm["Accuracy"], val_f1=vm["F1 (macro)"]))
        print(f"[{tag}] ep {ep:02d} loss {tl/n:.4f} | val acc {vm['Accuracy']:.4f} F1 {vm['F1 (macro)']:.4f} "
              f"loss {vm['Log Loss']:.4f} | {time.time()-te:.0f}s")
        score = (vm["F1 (macro)"], vm["Log Loss"])
        if score[0] > best[0] or (score[0] == best[0] and score[1] < best[1]):
            best, best_state, best_ep, wait = score, copy.deepcopy(model.state_dict()), ep, 0
        else:
            wait += 1
            if wait >= CFG["patience"]: print("early stop"); break
    model.load_state_dict(best_state)
    torch.save(best_state, f"{CFG['weights_dir']}/{tag}.pth")
    info = {"Params (M)": sum(p.numel() for p in model.parameters()) / 1e6, "Train time (min)": (time.time() - t0) / 60, "Best epoch": best_ep}
    return model, pd.DataFrame(hist), info

# %% ---------------- Plot helpers ----------------
def plot_cm(y, prob, title, out):
    cm = confusion_matrix(y, prob.argmax(1), labels=range(3)); cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=CLASSES, yticklabels=CLASSES, ax=ax[0])
    sns.heatmap(cmn, annot=True, fmt=".2%", cmap="Blues", cbar=False, vmin=0, vmax=1, xticklabels=CLASSES, yticklabels=CLASSES, ax=ax[1])
    ax[0].set_title(f"{title} — confusion matrix"); ax[1].set_title(f"{title} — normalized")
    for a in ax: a.set_xlabel("Predicted"); a.set_ylabel("True")
    savefig(out)

def plot_roc_pr(y, prob, title, out):
    yb = label_binarize(y, classes=range(3)); fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, c in enumerate(CLASSES):
        f, t, _ = roc_curve(yb[:, k], prob[:, k]); ax[0].plot(f, t, label=f"{c} (AUC={roc_auc_score(yb[:, k], prob[:, k]):.3f})")
        pr, rc, _ = precision_recall_curve(yb[:, k], prob[:, k]); ax[1].plot(rc, pr, label=f"{c} (AP={average_precision_score(yb[:, k], prob[:, k]):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set_title(f"{title} — ROC"); ax[1].set_title(f"{title} — Precision-Recall")
    ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    for a in ax: a.legend(fontsize=8); a.grid(alpha=.3)
    savefig(out)

def plot_hist(h, title, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(h.epoch, h.train_loss, "o-", ms=3, label="train (total loss)"); ax[0].plot(h.epoch, h.val_loss, "o-", ms=3, label="val (log loss)")
    ax[1].plot(h.epoch, h.val_acc, "o-", ms=3, label="val acc"); ax[1].plot(h.epoch, h.val_f1, "o-", ms=3, label="val macro-F1")
    ax[0].set_title(f"{title} — loss"); ax[1].set_title(f"{title} — validation")
    for a in ax: a.set_xlabel("epoch"); a.legend(); a.grid(alpha=.3)
    savefig(out)

def plot_tsne(model, d, hier, title, out):
    from sklearn.manifold import TSNE
    model.eval(); Z, Yl = [], []
    with torch.no_grad():
        for xc, xp, y in make_loader(d, False):
            Z.append(model(xc.to(DEVICE), xp.to(DEVICE))["emb"].float().cpu().numpy()); Yl.append(y.numpy())
    Z, Yl = np.concatenate(Z), np.concatenate(Yl)
    e = TSNE(2, perplexity=min(30, len(Z) - 1), random_state=0, init="pca").fit_transform(Z)
    plt.figure(figsize=(6, 5))
    for k, c in enumerate(CLASSES): plt.scatter(e[Yl == k, 0], e[Yl == k, 1], s=14, label=c, alpha=.8)
    plt.legend(); plt.title(f"{title} — t-SNE of embeddings (test)"); plt.xticks([]); plt.yticks([])
    savefig(out)

class GradCAM:
    def __init__(self, model, layer):
        self.model = model; layer.register_forward_hook(self._f)
    def _f(self, m, i, o):
        self.a = o.detach()
        if o.requires_grad: o.register_hook(lambda g: setattr(self, "g", g.detach()))
    def cam(self):
        c = torch.relu((self.g.mean((2, 3), keepdim=True) * self.a).sum(1))[0].cpu().numpy()
        return (c - c.min()) / (c.max() - c.min() + 1e-8)

def plot_gradcam_dual(model, hier, title, out, samples):
    """Grad-CAM of both branches; the polar CAM is warped back to Cartesian space."""
    model.eval()
    gc = GradCAM(model, model.bc); gp = GradCAM(model, model.bp)   # hook the full feature map of each backbone
    n = len(samples); fig, ax = plt.subplots(3, n, figsize=(2.6 * n, 8))
    for k, (_, r) in enumerate(samples.iterrows()):
        img = np.array(Image.open(r.path).convert("RGB").resize((S, S))); pol = to_polar(img)
        xc = norm(Image.fromarray(img))[None].to(DEVICE); xp = norm(Image.fromarray(pol))[None].to(DEVICE)
        with torch.enable_grad():
            o = model(xc, xp); p = final_probs(o, hier); cls = p.argmax(1).item()
            model.zero_grad(); o["logits"][0, cls].backward()
        cc = cv2.resize(gc.cam(), (S, S)); cp = from_polar(cv2.resize(gp.cam(), (S, S)).astype(np.float32), S)
        ok = CLASSES[cls] == r.label_name
        ax[0, k].imshow(img); ax[0, k].set_title(f"True: {r.label_name}", fontsize=8)
        ax[1, k].imshow(img); ax[1, k].imshow(cc, cmap="jet", alpha=.45); ax[1, k].set_title(f"Cartesian CAM | Pred: {CLASSES[cls]}", fontsize=7, color="green" if ok else "red")
        ax[2, k].imshow(img); ax[2, k].imshow(cp, cmap="jet", alpha=.45); ax[2, k].set_title("Polar CAM (mapped back)", fontsize=7)
        for a in ax[:, k]: a.axis("off")
    plt.suptitle(f"{title} — Grad-CAM of both views", fontweight="bold"); savefig(out)

def _font(s):
    try: return ImageFont.load_default(size=s)
    except TypeError: return ImageFont.load_default()

def montage(paths, out, title, ncols=2, width=1400, pad=30):
    ims = [Image.open(p).convert("RGB") for p in paths if os.path.exists(p)]
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

# %% [markdown]
# ## Part A — Ablation study (fixed split, same as the baseline notebook)

# %%
VARIANTS = {
    "A1_Cartesian_only (baseline)": dict(views="cart",  hier=False, supcon=False),
    "A2_Polar_only":                dict(views="polar", hier=False, supcon=False),
    "A3_Dual_concat":               dict(views="dual",  fusion="concat", hier=False, supcon=False),
    "A4_Dual_CVAF":                 dict(views="dual",  fusion="cvaf",   hier=False, supcon=False),
    "A5_Dual_CVAF_Hier":            dict(views="dual",  fusion="cvaf",   hier=True,  supcon=False),
    "A6_PolarFuse-Net (full)":      dict(views="dual",  fusion="cvaf",   hier=True,  supcon=True),
}
FULL = "A6_PolarFuse-Net (full)"; BASE = "A1_Cartesian_only (baseline)"
GC_SAMPLES = test_df.groupby("label_name").sample(2, random_state=CFG["seed"])
test_dl = make_loader(test_df, False)
ABL, ABL_PC, ABL_PROB, ABL_CI = [], [], {}, {}

if CFG["run_ablation"]:
    for name, v in VARIANTS.items():
        tag = name.split(" ")[0].replace("(", "").replace(")", "")
        d = f"{RES}/ablation/{tag}"; os.makedirs(d, exist_ok=True)
        print("\n" + "=" * 70 + f"\n{name}\n" + "=" * 70)
        model, hist, info = train_one(v, train_df, val_df, tag)
        y, prob = predict(model, test_dl, v["hier"], CFG["tta"])
        m = overall_metrics(y, prob); m.update(info); m["Variant"] = name; ABL.append(m)
        pc = per_class_metrics(y, prob); pc.insert(0, "Variant", name); ABL_PC.append(pc)
        ABL_PROB[name] = prob; ABL_CI[name] = bootstrap_ci(y, prob, CFG["n_bootstrap"])
        print(classification_report(y, prob.argmax(1), target_names=CLASSES, digits=4))
        pd.DataFrame({"path": test_df.path.values, "true": [CLASSES[i] for i in y], "pred": [CLASSES[i] for i in prob.argmax(1)],
                      **{f"prob_{c}": prob[:, k] for k, c in enumerate(CLASSES)}}).to_csv(f"{d}/predictions_test.csv", index=False)
        hist.to_csv(f"{d}/history.csv", index=False)
        plot_hist(hist, name, f"{d}/history.png"); plot_cm(y, prob, name, f"{d}/confusion_matrix.png"); plot_roc_pr(y, prob, name, f"{d}/roc_pr.png")
        plot_tsne(model, test_df, v["hier"], name, f"{d}/tsne.png")
        if v["views"] == "dual": plot_gradcam_dual(model, v["hier"], name, f"{d}/gradcam.jpg", GC_SAMPLES)
        montage([f"{d}/{f}" for f in ["confusion_matrix.png", "roc_pr.png", "history.png", "tsne.png", "gradcam.jpg"]],
                f"{RES}/_SUMMARY_{tag}.jpg", f"{name} - test results")
        del model; torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
    Y_TEST = y

    abl = pd.DataFrame(ABL).set_index("Variant")
    RAW = ["Log Loss", "MCC", "Cohen Kappa", "Pap<->Pseudo confusions", "Params (M)", "Train time (min)", "Best epoch"]
    abl_pct = abl.copy(); pc_cols = [c for c in abl.columns if c not in RAW]
    abl_pct[pc_cols] = (abl_pct[pc_cols] * 100).round(2); abl_pct[RAW] = abl_pct[RAW].round(4)
    for nme in abl_pct.index:
        ci = ABL_CI[nme]
        abl_pct.loc[nme, "Accuracy 95% CI"] = f"{ci['Accuracy'][0]*100:.2f}–{ci['Accuracy'][1]*100:.2f}"
        abl_pct.loc[nme, "F1 95% CI"] = f"{ci['F1 (macro)'][0]*100:.2f}–{ci['F1 (macro)'][1]*100:.2f}"
    print(abl_pct.to_string())

    # McNemar test: full model vs baseline (same test images)
    pb, pf = ABL_PROB[BASE].argmax(1) == Y_TEST, ABL_PROB[FULL].argmax(1) == Y_TEST
    b, c = int((pb & ~pf).sum()), int((~pb & pf).sum())
    p_mc = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    mcnemar = pd.DataFrame([{"Comparison": f"{FULL} vs {BASE}", "baseline right / full wrong": b,
                             "baseline wrong / full right": c, "exact McNemar p-value": p_mc}])
    print(mcnemar.to_string())

    # ablation figures
    show = ["Accuracy", "F1 (macro)", "Recall (macro)", "Specificity (macro)", "AUC-ROC (macro)"]
    ax = abl_pct[show].plot.bar(figsize=(13, 5), rot=15, width=.85)
    ax.set_ylim(max(0, abl_pct[show].values.min() - 5), 100.5); ax.set_ylabel("%"); ax.set_title("Ablation study — test set")
    ax.legend(ncol=3, fontsize=8, loc="lower right"); ax.grid(axis="y", alpha=.3); savefig(f"{RES}/ablation/ablation_bar.png")
    plt.figure(figsize=(8, 3.8)); sns.barplot(x=abl_pct["Pap<->Pseudo confusions"].values, y=abl_pct.index, color="#c0504d")
    plt.xlabel("# test images where Papilledema and Pseudopapilledema were confused"); plt.title("Hard-pair confusions (lower = better)")
    savefig(f"{RES}/ablation/hard_pair_confusions.png")
    n = len(ABL_PROB); fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
    for a, (nme, pr) in zip(axs, ABL_PROB.items()):
        sns.heatmap(confusion_matrix(Y_TEST, pr.argmax(1)), annot=True, fmt="d", cmap="Blues", cbar=False, ax=a, xticklabels=CLASSES, yticklabels=CLASSES)
        a.set_title(nme.split(" ")[0], fontsize=10); a.tick_params(labelsize=7)
    savefig(f"{RES}/ablation/all_confusion_matrices.png")

# %% [markdown]
# ## Part B — 5-fold stratified cross-validation of the full PolarFuse-Net

# %%
if CFG["run_cv"]:
    skf = StratifiedKFold(CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    OOF = np.zeros((len(df), 3)); CV_ROWS, CV_PC = [], []
    for f, (tr_i, te_i) in enumerate(skf.split(df, df.label)):
        tr_all, te = df.iloc[tr_i], df.iloc[te_i]
        tr, va = train_test_split(tr_all, test_size=0.15, stratify=tr_all.label, random_state=CFG["seed"])
        print("\n" + "=" * 70 + f"\nFold {f+1}/{CFG['n_folds']}  train {len(tr)}  val {len(va)}  test {len(te)}\n" + "=" * 70)
        model, hist, info = train_one(VARIANTS[FULL], tr, va, f"cv_fold{f+1}")
        y, prob = predict(model, make_loader(te, False), True, CFG["tta"])
        OOF[te_i] = prob
        m = overall_metrics(y, prob); m.update(info); m["Fold"] = f + 1; CV_ROWS.append(m)
        pc = per_class_metrics(y, prob); pc.insert(0, "Fold", f + 1); CV_PC.append(pc)
        hist.to_csv(f"{RES}/cv/history_fold{f+1}.csv", index=False)
        plot_cm(y, prob, f"Fold {f+1}", f"{RES}/cv/confusion_matrix_fold{f+1}.png")
        del model; torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    cv = pd.DataFrame(CV_ROWS).set_index("Fold")
    RAWc = ["Log Loss", "MCC", "Cohen Kappa", "Pap<->Pseudo confusions", "Params (M)", "Train time (min)", "Best epoch"]
    pct = [c for c in cv.columns if c not in RAWc]
    cv_show = cv.copy(); cv_show[pct] *= 100
    summary = pd.DataFrame({"mean": cv_show.mean(), "std": cv_show.std()}).round(4)
    summary["mean ± std"] = [f"{m:.2f} ± {s:.2f}" if k in pct else f"{m:.4f} ± {s:.4f}" for k, m, s in zip(summary.index, summary["mean"], summary["std"])]
    print(cv_show.round(2).to_string()); print(summary.to_string())

    yall = df.label.values
    oof_m = overall_metrics(yall, OOF)
    plot_cm(yall, OOF, "5-fold out-of-fold (all 1,369 images)", f"{RES}/cv/oof_confusion_matrix.png")
    plot_roc_pr(yall, OOF, "5-fold out-of-fold", f"{RES}/cv/oof_roc_pr.png")
    ax = cv_show[["Accuracy", "F1 (macro)", "AUC-ROC (macro)"]].plot(marker="o", figsize=(8, 4))
    ax.set_ylabel("%"); ax.set_title("Per-fold results"); ax.grid(alpha=.3); savefig(f"{RES}/cv/per_fold.png")
    pd.DataFrame({"path": df.path, "true": df.label_name, "pred": [CLASSES[i] for i in OOF.argmax(1)],
                  **{f"prob_{c}": OOF[:, k] for k, c in enumerate(CLASSES)}}).to_csv(f"{RES}/cv/oof_predictions.csv", index=False)

# %% ---------------- Save tables, overview image, zip ----------------
with pd.ExcelWriter(f"{RES}/ALL_RESULTS_PolarFuse.xlsx") as xw:
    if CFG["run_ablation"]:
        abl_pct.to_excel(xw, sheet_name="Ablation_test")
        pd.concat(ABL_PC).to_excel(xw, sheet_name="Ablation_per_class", index=False)
        mcnemar.to_excel(xw, sheet_name="McNemar", index=False)
    if CFG["run_cv"]:
        summary.to_excel(xw, sheet_name="CV_mean_std")
        cv_show.round(4).to_excel(xw, sheet_name="CV_per_fold")
        pd.concat(CV_PC).to_excel(xw, sheet_name="CV_per_class", index=False)
        pd.DataFrame([oof_m]).T.rename(columns={0: "OOF (all images)"}).to_excel(xw, sheet_name="CV_out_of_fold")
    pd.DataFrame(list({k: str(v) for k, v in CFG.items()}.items()), columns=["param", "value"]).to_excel(xw, sheet_name="Config", index=False)
if CFG["run_ablation"]:
    abl_pct.to_csv(f"{RES}/TABLE_ablation.csv"); mcnemar.to_csv(f"{RES}/McNemar.csv", index=False)
if CFG["run_cv"]:
    summary.to_csv(f"{RES}/TABLE_cv_mean_std.csv")

parts = []
if CFG["run_ablation"]:
    parts += [f"{RES}/ablation/ablation_bar.png", f"{RES}/ablation/hard_pair_confusions.png", f"{RES}/ablation/all_confusion_matrices.png",
              f"{RES}/ablation/A6_PolarFuse-Net/gradcam.jpg", f"{RES}/ablation/A6_PolarFuse-Net/tsne.png", f"{RES}/ablation/A1_Cartesian_only/tsne.png"]
if CFG["run_cv"]:
    parts += [f"{RES}/cv/oof_confusion_matrix.png", f"{RES}/cv/oof_roc_pr.png", f"{RES}/cv/per_fold.png"]
parts.insert(0, f"{RES}/data/cartesian_vs_polar.jpg")
montage(parts, f"{RES}/_OVERVIEW_PolarFuse.jpg", "PolarFuse-Net - ablation + 5-fold CV", ncols=2, width=1600)

with open(f"{RES}/README.txt", "w") as fh:
    fh.write("PolarFuse-Net results\n" + "=" * 60 + "\n")
    if CFG["run_ablation"]: fh.write("\nABLATION (fixed split, test set)\n" + abl_pct.to_string() + "\n\nMcNemar\n" + mcnemar.to_string() + "\n")
    if CFG["run_cv"]: fh.write("\n5-FOLD CV (mean ± std)\n" + summary["mean ± std"].to_string() + "\n\nOut-of-fold on all images\n" + pd.Series(oof_m).to_string() + "\n")
zp = shutil.make_archive(f"{WORK}/results_polarfuse", "zip", RES)
print(f"\n✅ saved {zp}  ({os.path.getsize(zp)/1e6:.1f} MB) — download it from the Output tab")
plt.figure(figsize=(20, 20)); plt.imshow(Image.open(f"{RES}/_OVERVIEW_PolarFuse.jpg")); plt.axis("off"); plt.show()
