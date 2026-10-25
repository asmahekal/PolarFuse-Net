# %% [markdown]
# # PolarFuse-Net v2 — Repeated 5-fold Cross-Validation (3 seeds × 5 folds)
# **Proposed model (chosen from the ablation study, variant B07):**
# Cartesian + **Polar** dual-view DenseNet121 → **Concat fusion** → **Hierarchical clinical heads**
# (Normal vs Elevated disc → Papilledema vs Pseudopapilledema) + **Supervised Contrastive loss** + flip-TTA.
#
# Compared against the two strongest models of the comparative study: **Swin-T** and **DenseNet121**.
#
# Protocol: repeated stratified 5-fold CV with 3 different seeds (different fold assignments) = **15 paired runs per model**.
# Every model sees exactly the same train / val / test images in each run → paired statistics:
# * paired t-test and Wilcoxon signed-rank over the 15 runs
# * **Nadeau–Bengio corrected resampled t-test** (the correct test for repeated CV)
# * McNemar test on the **out-of-fold predictions of all 1,369 images** (probabilities averaged over the 3 seeds)
#
# Everything → **`results_cv_v2.zip`**.
#
# ### Kaggle
# GPU T4/P100 + Internet ON → **Save Version → Save & Run All (Commit)** (≈ 4–6 h, runs in the background).
# A time guard (`max_hours`) stops training early and still writes all results + zip if Kaggle's 12 h limit gets close.

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
from scipy.stats import binomtest, ttest_rel, wilcoxon, t as t_dist

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
    seeds        = [42, 7, 2024],   # 3 repeats of 5-fold CV (different fold assignments)
    n_folds      = 5,
    num_workers  = 2,
    embed_dim    = 256,
    supcon_w     = 0.3,
    hier_w       = 0.5,
    supcon_temp  = 0.1,
    n_bootstrap  = 1000,
    pretrained   = True,
    max_hours    = 11.0,
    save_weights = False,       # 45 checkpoints would take ~3 GB; set True if you need them        # stop starting new runs after this (Kaggle limit is 12 h)
    results_dir  = f"{WORK}/results_cv_v2",
    weights_dir  = f"{WORK}/weights_cv_v2",
)
T_START = time.time()
RES = CFG["results_dir"]
if os.path.exists(RES): shutil.rmtree(RES)
for d in [RES, f"{RES}/figures", f"{RES}/per_run", f"{RES}/data", CFG["weights_dir"]]:
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
    if CFG.get("save_weights", False): torch.save(best_state, f"{CFG['weights_dir']}/{safe}.pth")
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


# %% ---------------- Models compared ----------------
PROPOSED = "PolarFuse-Net v2 (proposed)"
MODELS = {
    PROPOSED:      dict(kind="dual", views="dual", fusion="concat", hier=True, supcon=True, share=False,
                        backbone="densenet121", lr_bb=1e-4),
    "Swin-T":      dict(kind="base", backbone="swin_tiny_patch4_window7_224", lr_bb=5e-5),
    "DenseNet121": dict(kind="base", backbone="densenet121", lr_bb=1e-4),
}
NAMES = list(MODELS)

# %% [markdown]
# ## Repeated stratified 5-fold CV

# %%
ROWS, PC_ROWS = [], []
OOF = {s: {n: np.full((len(df), 3), np.nan) for n in NAMES} for s in CFG["seeds"]}
VIS_DONE = set()
stopped_early = False

for seed in CFG["seeds"]:
    skf = StratifiedKFold(CFG["n_folds"], shuffle=True, random_state=seed)
    for f, (tr_i, te_i) in enumerate(skf.split(df, df.label)):
        tr_all, te = df.iloc[tr_i], df.iloc[te_i]
        tr, va = train_test_split(tr_all, test_size=0.15, stratify=tr_all.label, random_state=seed)
        for nme in NAMES:
            if (time.time() - T_START) / 3600 > CFG["max_hours"]:
                print("⏰ time guard reached — stopping new runs, writing results"); stopped_early = True; break
            tag = f"{nme} | seed {seed} | fold {f+1}"
            print("\n" + "=" * 75 + f"\n{tag}   (elapsed {(time.time()-T_START)/3600:.2f} h)\n" + "=" * 75)
            CFG["seed"] = seed                      # train() seeds itself with CFG["seed"]
            model, hist, info = train(MODELS[nme], tr, va, f"{nme}_s{seed}_f{f+1}")
            y, prob = predict(model, make_loader(te, False), tta=True)
            OOF[seed][nme][te_i] = prob
            m = overall_metrics(y, prob); m.update(info); m.update(Model=nme, Seed=seed, Fold=f + 1, Run=f"s{seed}-f{f+1}")
            ROWS.append(m)
            pc = per_class_metrics(y, prob); pc.insert(0, "Run", f"s{seed}-f{f+1}"); pc.insert(0, "Model", nme); PC_ROWS.append(pc)
            hist.assign(Model=nme, Seed=seed, Fold=f + 1).to_csv(f"{RES}/per_run/history_{safe(nme)}_s{seed}_f{f+1}.csv", index=False)
            print(f"→ acc {m['Accuracy']*100:.2f}  F1 {m['F1 (macro)']*100:.2f}  AUC {m['AUC-ROC (macro)']*100:.2f}  Pap<->Pseudo {m['Pap<->Pseudo confusions']}")
            pd.DataFrame(ROWS).to_csv(f"{RES}/per_run/all_runs_live.csv", index=False)     # progress saved after every run
            if nme not in VIS_DONE:                 # Grad-CAM + t-SNE once per model (first fold of first seed)
                globals()["test_df"] = te
                globals()["GC_SAMPLES"] = te.groupby("label_name").sample(2, random_state=0)
                try: plot_gradcam(model, nme, f"{RES}/figures/gradcam_{safe(nme)}.jpg", False)
                except Exception as e: print("Grad-CAM skipped:", e)
                try: plot_tsne(model, nme, f"{RES}/figures/tsne_{safe(nme)}.png", False)
                except Exception as e: print("t-SNE skipped:", e)
                VIS_DONE.add(nme)
            del model; free()
        if stopped_early: break
    if stopped_early: break

# %% ---------------- Per-run table & mean ± std ----------------
runs = pd.DataFrame(ROWS)
complete = runs.groupby("Run").Model.nunique(); complete = complete[complete == len(NAMES)].index   # only fully paired runs
runs = runs[runs.Run.isin(complete)].copy()
print(f"\nPaired runs completed: {len(complete)}  (target {len(CFG['seeds']) * CFG['n_folds']})")
PCT = [c for c in runs.columns if c not in RAW_COLS + ["Model", "Seed", "Fold", "Run"]]
runs_pct = runs.copy(); runs_pct[PCT] *= 100

summ = []
for nme in NAMES:
    g = runs_pct[runs_pct.Model == nme]; r = {"Model": nme, "Runs": len(g)}
    for c in PCT + ["MCC", "Cohen Kappa", "Pap<->Pseudo confusions", "Params (M)", "Train time (min)", "Inference (ms/img)"]:
        r[c] = f"{g[c].mean():.2f} ± {g[c].std():.2f}" if c in PCT else f"{g[c].mean():.4f} ± {g[c].std():.4f}"
    summ.append(r)
SUMMARY = pd.DataFrame(summ).set_index("Model")
print(SUMMARY[["Runs", "Accuracy", "F1 (macro)", "AUC-ROC (macro)", "Cohen Kappa", "Pap<->Pseudo confusions"]].to_string())

# %% ---------------- Statistical tests: proposed vs each baseline ----------------
def corrected_t(diffs, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV."""
    d = np.asarray(diffs, float); k = len(d)
    if k < 2 or d.std(ddof=1) == 0: return np.nan
    var = d.var(ddof=1) * (1 / k + n_test / n_train)
    tstat = d.mean() / math.sqrt(var)
    return 2 * t_dist.sf(abs(tstat), k - 1)

n_te = len(df) / CFG["n_folds"]; n_tr = len(df) - n_te
STATS = []
for base in NAMES[1:]:
    for metric in ["Accuracy", "F1 (macro)", "AUC-ROC (macro)", "Cohen Kappa", "Pap<->Pseudo confusions"]:
        a = runs_pct[runs_pct.Model == PROPOSED].set_index("Run")[metric]
        b = runs_pct[runs_pct.Model == base].set_index("Run")[metric].reindex(a.index)
        d = a - b
        try: pw = wilcoxon(a, b).pvalue if (d != 0).any() else 1.0
        except ValueError: pw = np.nan
        STATS.append({"Comparison": f"{PROPOSED} vs {base}", "Metric": metric,
                      "Proposed mean": a.mean(), "Baseline mean": b.mean(), "Mean diff": d.mean(),
                      "Runs proposed better": int((d > 0).sum()) if metric != "Pap<->Pseudo confusions" else int((d < 0).sum()),
                      "Ties": int((d == 0).sum()), "Runs": len(d),
                      "paired t-test p": ttest_rel(a, b).pvalue if d.std() > 0 else np.nan,
                      "Wilcoxon p": pw, "Nadeau-Bengio corrected t p": corrected_t(d, n_tr, n_te)})
STATS = pd.DataFrame(STATS)
print(STATS.round(4).to_string())

# %% ---------------- Out-of-fold on all 1,369 images (probabilities averaged over seeds) ----------------
y_all = df.label.values
OOF_MEAN = {}
for nme in NAMES:
    stack = [OOF[s][nme] for s in CFG["seeds"] if not np.isnan(OOF[s][nme]).any()]
    if stack: OOF_MEAN[nme] = np.mean(stack, 0)
OOF_TAB = to_pct_table([dict(overall_metrics(y_all, OOF_MEAN[n]), Model=n) for n in OOF_MEAN]) if OOF_MEAN else pd.DataFrame()
MCN = []
for base in [n for n in NAMES[1:] if n in OOF_MEAN and PROPOSED in OOF_MEAN]:
    n01, n10, p = mcnemar(y_all, OOF_MEAN[PROPOSED], OOF_MEAN[base])
    MCN.append({"Comparison": f"{PROPOSED} vs {base}", "only proposed right": n01, "only baseline right": n10, "exact McNemar p": p})
MCN = pd.DataFrame(MCN)
OOF_PC = pd.concat([per_class_metrics(y_all, OOF_MEAN[n]).assign(Model=n) for n in OOF_MEAN]) if OOF_MEAN else pd.DataFrame()
if len(OOF_TAB): print(OOF_TAB[["Accuracy", "F1 (macro)", "AUC-ROC (macro)", "Cohen Kappa", "Pap<->Pseudo confusions"]].to_string())
print(MCN.to_string())

# %% ---------------- Figures ----------------
FG = f"{RES}/figures"; pal = dict(zip(NAMES, sns.color_palette("Set2", len(NAMES))))
# 1 box plots of the main metrics over 15 runs
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
for a, c in zip(ax, ["Accuracy", "F1 (macro)", "AUC-ROC (macro)", "Pap<->Pseudo confusions"]):
    sns.boxplot(data=runs_pct, x="Model", y=c, order=NAMES, palette=pal, ax=a)
    sns.stripplot(data=runs_pct, x="Model", y=c, order=NAMES, color="k", size=4, ax=a)
    a.set_title(f"{c} — {len(complete)} runs"); a.set_xlabel(""); a.tick_params(axis="x", rotation=15)
savefig(f"{FG}/boxplots.png")
# 2 paired lines per run (accuracy & F1)
fig, ax = plt.subplots(1, 2, figsize=(16, 4.5))
for a, c in zip(ax, ["Accuracy", "F1 (macro)"]):
    for nme in NAMES:
        g = runs_pct[runs_pct.Model == nme].sort_values(["Seed", "Fold"]); a.plot(g.Run, g[c], "o-", label=nme, color=pal[nme])
    a.set_title(f"{c} per run (paired)"); a.tick_params(axis="x", rotation=60); a.grid(alpha=.3); a.legend(fontsize=8)
savefig(f"{FG}/per_run_lines.png")
# 3 mean ± std bars
mean = runs_pct.groupby("Model")[["Accuracy", "Precision (macro)", "Recall/Sensitivity (macro)", "Specificity (macro)", "F1 (macro)", "AUC-ROC (macro)"]].agg(["mean", "std"]).reindex(NAMES)
cols = ["Accuracy", "Precision (macro)", "Recall/Sensitivity (macro)", "Specificity (macro)", "F1 (macro)", "AUC-ROC (macro)"]
fig, ax = plt.subplots(figsize=(13, 5)); w = 0.8 / len(NAMES)
x = np.arange(len(cols))
for i, nme in enumerate(NAMES):
    ax.bar(x + i * w, [mean.loc[nme, (c, "mean")] for c in cols], w, yerr=[mean.loc[nme, (c, "std")] for c in cols], capsize=3, label=nme, color=pal[nme])
ax.set_xticks(x + w * (len(NAMES) - 1) / 2); ax.set_xticklabels(cols, rotation=10)
ax.set_ylim(max(0, mean.xs("mean", axis=1, level=1).values.min() - 6), 100.5); ax.set_ylabel("%"); ax.legend(); ax.grid(axis="y", alpha=.3)
ax.set_title("Repeated 5-fold CV — mean ± std"); savefig(f"{FG}/mean_std_bars.png")
# 4 OOF confusion matrices + ROC
if OOF_MEAN:
    fig, ax = plt.subplots(1, len(OOF_MEAN), figsize=(5.2 * len(OOF_MEAN), 4.6))
    for a, nme in zip(np.atleast_1d(ax), OOF_MEAN):
        sns.heatmap(confusion_matrix(y_all, OOF_MEAN[nme].argmax(1)), annot=True, fmt="d", cmap="Blues", cbar=False, ax=a,
                    xticklabels=CLASSES, yticklabels=CLASSES)
        a.set_title(f"{nme}\nOOF acc {accuracy_score(y_all, OOF_MEAN[nme].argmax(1))*100:.2f}% (1,369 images)", fontsize=10)
        a.set_xlabel("Predicted"); a.set_ylabel("True"); a.tick_params(labelsize=8)
    savefig(f"{FG}/oof_confusion_matrices.png")
    yb = label_binarize(y_all, classes=range(3)); fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for k, c in enumerate(CLASSES):
        for nme in OOF_MEAN:
            fp, tp_, _ = roc_curve(yb[:, k], OOF_MEAN[nme][:, k])
            ax[k].plot(fp, tp_, lw=2.5 if nme == PROPOSED else 1.3, color=pal[nme], label=f"{nme} ({roc_auc_score(yb[:, k], OOF_MEAN[nme][:, k]):.4f})")
        ax[k].plot([0, 1], [0, 1], "k--"); ax[k].set_title(f"OOF ROC — {c}"); ax[k].legend(fontsize=8, loc="lower right"); ax[k].grid(alpha=.3)
    savefig(f"{FG}/oof_roc.png")
    for nme in OOF_MEAN: plot_per_class(per_class_metrics(y_all, OOF_MEAN[nme]), f"{nme} (OOF)", f"{FG}/oof_per_class_{safe(nme)}.png", False)
# 5 tables as images
def table_img(t, title, out, fs=9):
    fig, ax = plt.subplots(figsize=(min(26, 2.2 + 1.6 * len(t.columns)), 0.55 * len(t) + 1.4)); ax.axis("off")
    tb = ax.table(cellText=t.values, rowLabels=t.index, colLabels=[str(c).replace(" (", "\n(") for c in t.columns], loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(fs); tb.scale(1, 2); ax.set_title(title, fontweight="bold"); savefig(out)
table_img(SUMMARY[["Runs", "Accuracy", "Precision (macro)", "Recall/Sensitivity (macro)", "Specificity (macro)", "F1 (macro)",
                   "AUC-ROC (macro)", "Cohen Kappa", "MCC", "Pap<->Pseudo confusions"]],
          f"Repeated 5-fold CV ({len(complete)} paired runs) — mean ± std (%)", f"{FG}/table_cv_summary.png")
st = STATS[STATS.Metric.isin(["Accuracy", "F1 (macro)", "AUC-ROC (macro)"])].copy()
st = st.set_index(["Comparison", "Metric"])[["Proposed mean", "Baseline mean", "Runs proposed better", "Ties",
                                             "paired t-test p", "Wilcoxon p", "Nadeau-Bengio corrected t p"]].round(4)
st.index = [f"{a.split(' vs ')[1]} | {b}" for a, b in st.index]
table_img(st, "Proposed vs baselines — paired statistics", f"{FG}/table_stats.png")

# %% ---------------- Excel, overview, README, zip ----------------
with pd.ExcelWriter(f"{RES}/ALL_RESULTS_cv_v2.xlsx") as xw:
    SUMMARY.to_excel(xw, sheet_name="CV_mean_std")
    runs_pct.round(4).to_excel(xw, sheet_name="All_runs", index=False)
    STATS.round(6).to_excel(xw, sheet_name="Stats_paired", index=False)
    if len(OOF_TAB): OOF_TAB.to_excel(xw, sheet_name="OOF_all_images")
    if len(MCN): MCN.to_excel(xw, sheet_name="OOF_McNemar", index=False)
    if len(OOF_PC): OOF_PC.to_excel(xw, sheet_name="OOF_per_class", index=False)
    pd.concat(PC_ROWS).to_excel(xw, sheet_name="Per_class_per_run", index=False)
    pd.DataFrame([{"Model": k, **v} for k, v in MODELS.items()]).to_excel(xw, sheet_name="Model_configs", index=False)
    pd.DataFrame(list({k: str(v) for k, v in CFG.items()}.items()), columns=["param", "value"]).to_excel(xw, sheet_name="Config", index=False)
SUMMARY.to_csv(f"{RES}/TABLE_cv_mean_std.csv"); STATS.to_csv(f"{RES}/TABLE_stats.csv", index=False)
if len(OOF_TAB): OOF_TAB.to_csv(f"{RES}/TABLE_oof_all_images.csv")
for s in CFG["seeds"]:
    for nme in NAMES:
        if not np.isnan(OOF[s][nme]).any():
            pd.DataFrame({"path": df.path, "true": df.label_name, **{f"prob_{c}": OOF[s][nme][:, k] for k, c in enumerate(CLASSES)}}
                         ).to_csv(f"{RES}/per_run/oof_{safe(nme)}_seed{s}.csv", index=False)

figs = [f"{FG}/table_cv_summary.png", f"{FG}/table_stats.png", f"{FG}/boxplots.png", f"{FG}/mean_std_bars.png",
        f"{FG}/oof_confusion_matrices.png", f"{FG}/oof_roc.png", f"{FG}/per_run_lines.png",
        f"{FG}/oof_per_class_{safe(PROPOSED)}.png"] + [f"{FG}/gradcam_{safe(n)}.jpg" for n in NAMES] + [f"{FG}/tsne_{safe(n)}.png" for n in NAMES]
montage(figs, f"{RES}/_OVERVIEW_cv_v2.jpg", "PolarFuse-Net v2 - repeated 5-fold CV vs Swin-T and DenseNet121", 2, 1600)

with open(f"{RES}/README.txt", "w") as fh:
    fh.write("PolarFuse-Net v2 — repeated 5-fold CV\n" + "=" * 70 + "\n")
    fh.write(f"Paired runs: {len(complete)}   stopped by time guard: {stopped_early}   total time: {(time.time()-T_START)/3600:.2f} h\n\n")
    fh.write("MEAN ± STD\n" + SUMMARY.to_string() + "\n\nPAIRED STATISTICS\n" + STATS.round(4).to_string() + "\n")
    if len(OOF_TAB): fh.write("\nOUT-OF-FOLD (all 1,369 images, seed-averaged)\n" + OOF_TAB.to_string() + "\n\n" + MCN.to_string() + "\n")
zp = shutil.make_archive(f"{WORK}/results_cv_v2", "zip", RES)
print(f"\n✅ saved {zp}  ({os.path.getsize(zp)/1e6:.1f} MB) — Output tab → results_cv_v2.zip   total {(time.time()-T_START)/3600:.2f} h")
plt.figure(figsize=(20, 20)); plt.imshow(Image.open(f"{RES}/_OVERVIEW_cv_v2.jpg")); plt.axis("off"); plt.show()
