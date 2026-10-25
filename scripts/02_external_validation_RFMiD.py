# %% [markdown]
# # PolarFuse-Net — final experiments that address the study limitations
#
# | Part | Limitation addressed | What is done |
# |---|---|---|
# | 0 | data leakage | near-duplicate audit → duplicate-aware (group) folds (same as v3) |
# | **A** | comparison with prior work was not like-for-like | **re-implementation of the prior architectures** (Ahn 2019: ResNet-50, VGG19, Inception-V3; Kokulu 2023: MobileNetV2; Anitha 2026: EfficientNet-B2) inside the *same* leakage-free repeated CV as PolarFuse-Net, with paired statistics |
# | **B** | label noise | **confident-learning audit** on the out-of-fold predictions of all models + sensitivity analysis without the flagged images |
# | **C** | single centre, disc must be centred | **external validation** on RFMiD (India, 3 cameras): automatic optic-disc localisation → crop → papilledema/optic-disc-edema vs normal |
# | **D** | explainability only qualitative | **quantitative margin-focus score**: automatic disc-margin detection in polar space and the share of attribution on the margin band, PolarFuse-Net vs DenseNet121 |
#
# ### Kaggle setup
# * Add Data: **identification-of-pseudopapilledema** (required). For Part C (optional) add any of: **retinal-disease-classification** (andrewmvd, RFMiD 1.0), **RFMiD 2.0** (Zenodo 7505822), **JSIEC-1000** (Zenodo 3477553 / Kaggle 'fundusimage1000').
# * GPU T4/P100 + Internet ON → **Save Version → Save & Run All (Commit)**. Expected ≈ 6–8 h; a time guard writes everything before Kaggle's 12 h limit.
# * Output → **`results_final.zip`**.

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

from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, cohen_kappa_score, matthews_corrcoef, log_loss)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from scipy.stats import binomtest, ttest_rel, wilcoxon, t as t_dist

CODE_VERSION = "final-1.1-C (external validation only)"
print("CODE VERSION:", CODE_VERSION)

# %% ---------------- Config ----------------
WORK = "/kaggle/working" if os.path.exists("/kaggle/working") else "./outputs"
CFG = dict(
    data_root    = "/kaggle/input",
    img_size     = 224,
    batch_size   = 24,
    epochs       = 30,
    patience     = 8,
    warmup_ep    = 2,
    lr_head      = 1e-3,
    weight_decay = 1e-4,
    ema_decay    = 0.995,
    seed         = 42,
    seeds        = [42, 7, 2024],
    n_folds      = 5,
    num_workers  = 2,
    embed_dim    = 256,
    supcon_w     = 0.3,
    hier_w       = 0.5,
    supcon_temp  = 0.1,
    n_bootstrap  = 1000,
    pretrained   = True,
    audit_model  = "vit_small_patch14_dinov2.lvd142m",
    dup_hash_thr = 6,
    dup_cos_thr  = 0.97,
    run_A = False, run_B = False, run_C = True, run_D = False,
    crop_frac    = 0.60,        # Part C: crop side = crop_frac × field-of-view radius (disc ≈ 40 % of the crop, like the training data)
    max_hours    = 11.0,
    save_weights = False,
    results_dir  = f"{WORK}/results_external",
    weights_dir  = f"{WORK}/weights_final",
)
T_START = time.time()
RES = CFG["results_dir"]
if os.path.exists(RES): shutil.rmtree(RES)
for d in [RES, f"{RES}/figures", f"{RES}/per_run", f"{RES}/data", f"{RES}/audit", f"{RES}/A_prior_work", f"{RES}/B_label_noise",
          f"{RES}/C_external", f"{RES}/D_margin_focus", CFG["weights_dir"]]:
    os.makedirs(d, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "")

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
seed_everything(CFG["seed"])

def savefig(path, show=True):
    plt.tight_layout()
    kw = {"pil_kwargs": {"quality": 92}} if path.endswith(".jpg") else {}
    plt.savefig(path, dpi=200, bbox_inches="tight", **kw)
    if show: plt.show()
    plt.close("all")

def free():
    gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

def time_left_h(): return CFG["max_hours"] - (time.time() - T_START) / 3600
FAILED = []
def step(name, fn):
    """Run one part; on error log it and continue so that the zip is always written."""
    try: return fn()
    except Exception as e:
        import traceback; traceback.print_exc(); FAILED.append(f"{name}: {e!r}"); return None

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

_INV_MAPS = {}
def from_polar(pol_np, size):
    """Inverse polar -> Cartesian with an explicit remap.
    (cv2.warpPolar(..., WARP_INVERSE_MAP) leaves pixels outside the circle UNINITIALISED, which produced garbage
    values up to 1e24 — that is why the mapped-back CAMs looked empty or had coloured corners.)"""
    h, w = pol_np.shape[:2]
    if (size, h, w) not in _INV_MAPS:
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        dx, dy = xx - size / 2, yy - size / 2
        r = np.sqrt(dx ** 2 + dy ** 2) / (size / 2) * w                     # radius -> polar column
        th = (np.arctan2(dy, dx) % (2 * np.pi)) / (2 * np.pi) * h          # angle  -> polar row
        _INV_MAPS[(size, h, w)] = (r.astype(np.float32), th.astype(np.float32))
    mx, my = _INV_MAPS[(size, h, w)]
    src = np.concatenate([pol_np, pol_np[:1]], 0)          # close the 360° -> 0° seam so it interpolates instead of leaving a line
    return cv2.remap(np.ascontiguousarray(src), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

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
        mk = lambda: timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="",
                                      **({"img_size": S} if "dinov2" in backbone else {}))
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
        if f.ndim == 3:                                       # ViT / DINOv2: (B, prefix + N, C) -> (B, C, h, w)
            f = f[:, getattr(backbone, "num_prefix_tokens", 1):]
            s = int(round(math.sqrt(f.shape[1]))); f = f.transpose(1, 2).reshape(f.shape[0], -1, s, s)
        if f.requires_grad: f.retain_grad()
        self.feats[key] = f                                   # kept for Grad-CAM / HiResCAM
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
        self.net = timm.create_model(name, pretrained=pretrained, num_classes=3, **({"img_size": S} if "dinov2" in name else {}))
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
            if tta:
                # geometric TTA that is CONSISTENT between the two views (verified numerically):
                #   vertical flip   θ -> -θ      : reverse the angle axis of the polar image
                #   horizontal flip θ -> π - θ   : reverse + shift by half a turn
                #   180° rotation   θ -> θ + π   : shift by half a turn
                h = S // 2
                p = p + final_probs(model(xc.flip(2), xp.flip(2).roll(1, 2)), model.hier)
                p = p + final_probs(model(xc.flip(3), xp.flip(2).roll(h + 1, 2)), model.hier)
                p = p + final_probs(model(xc.flip(2).flip(3), xp.roll(h, 2)), model.hier)
                p = p / 4
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
    ema = copy.deepcopy(model).eval(); [p.requires_grad_(False) for p in ema.parameters()]
    def ema_update(d=CFG.get("ema_decay", 0.995)):
        with torch.no_grad():
            for (k, ve), vm_ in zip(ema.state_dict().items(), model.state_dict().values()):
                if ve.dtype.is_floating_point: ve.mul_(d).add_(vm_.detach(), alpha=1 - d)
                else: ve.copy_(vm_)
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
            scaler.step(opt); scaler.update(); sched.step(); ema_update()
            tl += loss.item() * len(y); n += len(y); corr += (out["logits"].argmax(1) == y).sum().item()
        yv, pv = predict(ema, va_dl); vm = overall_metrics(yv, pv)          # validate the EMA weights
        hist.append(dict(epoch=ep, train_loss=tl / n, train_acc=corr / n, val_loss=vm["Log Loss"], val_acc=vm["Accuracy"], val_f1=vm["F1 (macro)"]))
        print(f"[{tag}] ep {ep:02d} loss {tl/n:.4f} acc {corr/n:.4f} | val acc {vm['Accuracy']:.4f} F1 {vm['F1 (macro)']:.4f} "
              f"loss {vm['Log Loss']:.4f} | {time.time()-te:.0f}s")
        if vm["F1 (macro)"] > best[0] or (vm["F1 (macro)"] == best[0] and vm["Log Loss"] < best[1]):
            best, best_state, best_ep, wait = (vm["F1 (macro)"], vm["Log Loss"]), copy.deepcopy(ema.state_dict()), ep, 0
        else:
            wait += 1
            if wait >= CFG["patience"]: print("early stop"); break
    del ema; model.load_state_dict(best_state)
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




# %% ---------------- Extra metrics: calibration + selective prediction ----------------
def ece_score(y, prob, n_bins=15):
    conf = prob.max(1); acc = (prob.argmax(1) == y).astype(float); e = 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e

def selective(y, prob, coverages=(1.0, 0.95, 0.90)):
    conf = prob.max(1); order = np.argsort(-conf); correct = (prob.argmax(1) == y)[order]; out = {}
    for c in coverages:
        k = max(1, int(round(c * len(y)))); out[f"Acc @ {int(c*100)}% coverage"] = correct[:k].mean()
    risks = 1 - np.cumsum(correct) / np.arange(1, len(y) + 1); out["AURC"] = risks.mean()
    return out

def full_metrics(y, prob):
    m = overall_metrics(y, prob); m["ECE"] = ece_score(y, prob); m.update(selective(y, prob)); return m

RAW_COLS = RAW_COLS + ["ECE", "AURC"]

# %% [markdown]
# ## Part 0 — Benchmark audit: near-duplicate / same-eye images

# %%
def phash(path, size=32, keep=8):
    g = cv2.resize(cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2GRAY), (size, size)).astype(np.float32)
    d = cv2.dct(g)[:keep, :keep].flatten(); return d[1:] > np.median(d[1:])

hashes = np.stack([phash(p) for p in df.path])
H = (hashes[:, None, :] != hashes[None, :, :]).sum(-1)              # pairwise Hamming distance

audit_net = timm.create_model(CFG["audit_model"], pretrained=CFG["pretrained"], num_classes=0,
                              **({"img_size": S} if "dinov2" in CFG["audit_model"] else {})).to(DEVICE).eval()
E = []
with torch.no_grad():
    for i in range(0, len(df), 64):
        x = torch.stack([norm(Image.open(p).convert("RGB").resize((S, S))) for p in df.path.iloc[i:i + 64]]).to(DEVICE)
        E.append(F.normalize(audit_net(x).float(), dim=1).cpu())
E = torch.cat(E).numpy(); COS = E @ E.T
del audit_net; free()

iu = np.triu_indices(len(df), 1)
pairs = pd.DataFrame({"i": iu[0], "j": iu[1], "hamming": H[iu], "cosine": COS[iu]})
is_dup = (pairs.hamming <= CFG["dup_hash_thr"]) | ((pairs.cosine >= CFG["dup_cos_thr"]) & (pairs.hamming <= 20))
dups = pairs[is_dup].copy()
dups["label_i"] = df.label_name.values[dups.i]; dups["label_j"] = df.label_name.values[dups.j]
dups["path_i"] = df.path.values[dups.i]; dups["path_j"] = df.path.values[dups.j]
dups["same_label"] = dups.label_i == dups.label_j

parent = list(range(len(df)))
def find(a):
    while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
    return a
for a, b in zip(dups.i, dups.j): parent[find(a)] = find(b)
df["group"] = [find(k) for k in range(len(df))]
gsize = df.groupby("group").size()

AUDIT = pd.Series({
    "images": len(df),
    "near-duplicate pairs": len(dups),
    "pairs with the SAME label": int(dups.same_label.sum()),
    "pairs with DIFFERENT labels (label conflict)": int((~dups.same_label).sum()),
    "images involved in a duplicate group": int(gsize[gsize > 1].sum()),
    "duplicate groups": int((gsize > 1).sum()),
    "largest group": int(gsize.max()),
})
sens = pd.DataFrame([{"criterion": f"pHash ≤ {h}", "pairs": int((pairs.hamming <= h).sum())} for h in [2, 4, 6, 8, 10]] +
                    [{"criterion": f"cosine ≥ {c}", "pairs": int((pairs.cosine >= c).sum())} for c in [0.99, 0.98, 0.97, 0.95]])
print(AUDIT.to_string()); print(sens.to_string())
AUDIT.to_csv(f"{RES}/audit/audit_summary.csv"); sens.to_csv(f"{RES}/audit/threshold_sensitivity.csv", index=False)
dups.sort_values(["hamming", "cosine"], ascending=[True, False]).to_csv(f"{RES}/audit/near_duplicate_pairs.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].hist(pairs.hamming, bins=np.arange(0, 65), color="#4c72b0"); ax[0].axvline(CFG["dup_hash_thr"], color="r", ls="--"); ax[0].set_yscale("log")
ax[0].set_title("pHash Hamming distance, all pairs"); ax[0].set_xlabel("distance (64-bit)")
ax[1].hist(pairs.cosine, bins=100, range=(min(float(pairs.cosine.min()), 0.999) - 1e-3, 1.0), color="#55a868"); ax[1].axvline(CFG["dup_cos_thr"], color="r", ls="--"); ax[1].set_yscale("log")
ax[1].set_title("DINOv2 cosine similarity, all pairs"); ax[1].set_xlabel("cosine")
savefig(f"{RES}/audit/similarity_distributions.png")
top = dups.sort_values(["hamming", "cosine"], ascending=[True, False]).head(12)
if len(top):
    fig, ax = plt.subplots(2, len(top), figsize=(2.3 * len(top), 5.2)); ax = np.array(ax).reshape(2, len(top))
    for k, (_, r) in enumerate(top.iterrows()):
        for row, (p, l) in enumerate([(r.path_i, r.label_i), (r.path_j, r.label_j)]):
            ax[row, k].imshow(Image.open(p).convert("RGB")); ax[row, k].axis("off")
            ax[row, k].set_title(l, fontsize=7, color="black" if r.same_label else "red")
        ax[0, k].text(0.5, 1.25, f"d={r.hamming}  cos={r.cosine:.3f}", transform=ax[0, k].transAxes, ha="center", fontsize=7)
    plt.suptitle("Most similar image pairs (red = different labels)", fontweight="bold"); savefig(f"{RES}/audit/top_pairs.jpg")


# %% ---------------- shared helpers for the final experiments ----------------
PROPOSED = "PolarFuse-Net (proposed)"
DUAL_CFG = dict(kind="dual", views="dual", fusion="concat", hier=True, supcon=True, backbone="densenet121", lr_bb=1e-4)
MODELS = {
    "ResNet-50 [Ahn 2019; Al-Azzawi 2023]":  dict(kind="base", backbone="resnet50", lr_bb=1e-4),
    "VGG19 [Ahn 2019]":                      dict(kind="base", backbone="vgg19", lr_bb=5e-5),
    "Inception-V3 [Ahn 2019]":               dict(kind="base", backbone="inception_v3", lr_bb=1e-4),
    "MobileNetV2 [Kokulu & Göker 2023]":     dict(kind="base", backbone="mobilenetv2_100", lr_bb=2e-4),
    "EfficientNet-B2 [Anitha 2026]":         dict(kind="base", backbone="efficientnet_b2", lr_bb=1e-4),
    "DenseNet121 (baseline)":                dict(kind="base", backbone="densenet121", lr_bb=1e-4),
    PROPOSED:                                DUAL_CFG,
}
REPORTED = {"ResNet-50 [Ahn 2019; Al-Azzawi 2023]": "98.63 / 98", "VGG19 [Ahn 2019]": "96.80", "Inception-V3 [Ahn 2019]": "96.45",
            "MobileNetV2 [Kokulu & Göker 2023]": "96", "EfficientNet-B2 [Anitha 2026]": "98.55", "DenseNet121 (baseline)": "—", PROPOSED: "—"}
NAMES = list(MODELS)
short = {n: n.split(" [")[0].replace("PolarFuse-Net (proposed)", "PolarFuse-Net*") for n in NAMES}
pal = dict(zip(NAMES, sns.color_palette("tab10", len(NAMES))))

def hirescam(feat):
    return torch.relu((feat.grad * feat.detach()).sum(1))[0].float().cpu().numpy()
def up_c(c):
    c = np.clip(cv2.resize(c, (S, S), interpolation=cv2.INTER_CUBIC), 0, None); return c / c.max() if c.max() > 0 else c
def up_p(c):
    n = c.shape[0]; cp = np.vstack([c[-1:], c, c[:1]]); big = cv2.resize(cp, (S, int(round(S * (n + 2) / n))), interpolation=cv2.INTER_CUBIC)
    off = (big.shape[0] - S) // 2; big = np.clip(big[off:off + S], 0, None); return big / big.max() if big.max() > 0 else big

def corrected_t(d, n_train, n_test):
    d = np.asarray(d, float); k = len(d)
    if k < 2 or d.std(ddof=1) == 0: return np.nan
    return 2 * t_dist.sf(abs(d.mean() / math.sqrt(d.var(ddof=1) * (1 / k + n_test / n_train))), k - 1)

def table_img(t, title, out, fs=9, labels=None):
    fig, ax = plt.subplots(figsize=(min(28, 3 + 1.7 * len(t.columns)), 0.6 * len(t) + 1.5)); ax.axis("off")
    tb = ax.table(cellText=t.values, rowLabels=labels or list(t.index), colLabels=[str(c).replace(" (", "\n(").replace(" @ ", "\n@ ") for c in t.columns],
                  loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(fs); tb.scale(1, 2.1)
    ax.set_title(title, fontweight="bold"); savefig(out)

STATE = {}

# %% [markdown]
# ## Part A — prior-work architectures re-implemented in the same leakage-free repeated CV

# %%
def part_A():
    ROWS, OOF = [], {s: {n: np.full((len(df), 3), np.nan) for n in NAMES} for s in CFG["seeds"]}
    stop = False
    for seed in CFG["seeds"]:
        sgk = StratifiedGroupKFold(CFG["n_folds"], shuffle=True, random_state=seed)
        for f, (tr_i, te_i) in enumerate(sgk.split(df, df.label, df.group)):
            tr_all, te = df.iloc[tr_i], df.iloc[te_i]
            a_i, v_i = next(StratifiedGroupKFold(7, shuffle=True, random_state=seed).split(tr_all, tr_all.label, tr_all.group))
            tr, va = tr_all.iloc[a_i], tr_all.iloc[v_i]
            got = {}
            for nme, cfg in MODELS.items():
                if time_left_h() < 1.2:            # keep ~1 h for Parts B–D
                    print("⏰ time guard: stopping Part A"); stop = True; break
                print("\n" + "=" * 78 + f"\n[A] {nme} | seed {seed} | fold {f+1}  (elapsed {(time.time()-T_START)/3600:.2f} h)\n" + "=" * 78)
                CFG["seed"] = seed
                try:
                    model, hist, info = train(cfg, tr, va, f"{nme}_s{seed}_f{f+1}")
                except Exception as e:
                    print("!! failed:", e); FAILED.append(f"A {nme} s{seed} f{f+1}: {e!r}"); free(); continue
                _, prob = predict(model, make_loader(te, False), tta=True)
                got[nme] = (prob, info)
                if seed == CFG["seeds"][0] and f == 0 and nme in (PROPOSED, "DenseNet121 (baseline)"):
                    STATE.setdefault("cam_models", {})[nme] = model; STATE["cam_test"] = te
                else:
                    del model
                free()
            if stop: break
            if len(got) < len(NAMES): continue
            y = te.label.values
            for nme, (prob, info) in got.items():
                OOF[seed][nme][te_i] = prob
                m = full_metrics(y, prob); m.update(info); m.update(Model=nme, Seed=seed, Fold=f + 1, Run=f"s{seed}-f{f+1}"); ROWS.append(m)
            pd.DataFrame(ROWS).to_csv(f"{RES}/per_run/all_runs_live.csv", index=False)
        if stop: break
    for s in CFG["seeds"]:
        for n in NAMES:
            if not np.isnan(OOF[s][n]).any():
                pd.DataFrame({"path": df.path, "true": df.label_name, **{f"prob_{c}": OOF[s][n][:, k] for k, c in enumerate(CLASSES)}}
                             ).to_csv(f"{RES}/per_run/oof_{safe(short[n])}_seed{s}.csv", index=False)
    STATE["OOF"] = OOF
    runs = pd.DataFrame(ROWS)
    if not len(runs): return
    PCT = [c for c in runs.columns if c not in RAW_COLS + ["Model", "Seed", "Fold", "Run"]]
    rp = runs.copy(); rp[PCT] *= 100; STATE["runs_pct"] = rp
    summ = []
    for n in NAMES:
        g = rp[rp.Model == n]
        if not len(g): continue
        summ.append({"Model": n, "Reported in original paper (%)": REPORTED[n], "Runs": len(g),
                     **{c: f"{g[c].mean():.2f} ± {g[c].std():.2f}" for c in ["Accuracy", "Precision (macro)", "Recall/Sensitivity (macro)",
                                                                          "Specificity (macro)", "F1 (macro)", "AUC-ROC (macro)", "Acc @ 95% coverage"]},
                     "Cohen Kappa": f"{g['Cohen Kappa'].mean():.4f} ± {g['Cohen Kappa'].std():.4f}",
                     "ECE": f"{g['ECE'].mean():.4f} ± {g['ECE'].std():.4f}",
                     "Params (M)": f"{g['Params (M)'].mean():.1f}", "Inference (ms/img)": f"{g['Inference (ms/img)'].mean():.1f}"})
    SUM = pd.DataFrame(summ).set_index("Model"); STATE["SUM_A"] = SUM; print(SUM.to_string())
    n_te = len(df) / CFG["n_folds"]; n_tr = len(df) - n_te; ST = []
    for b in [n for n in NAMES if n != PROPOSED]:
        for met in ["Accuracy", "F1 (macro)", "AUC-ROC (macro)", "Cohen Kappa", "ECE"]:
            a = rp[rp.Model == PROPOSED].set_index("Run")[met]; bb = rp[rp.Model == b].set_index("Run")[met].reindex(a.index); d = a - bb
            if d.isna().any() or len(d) < 2: continue
            better = (d < 0) if met == "ECE" else (d > 0)
            try: pw = wilcoxon(a, bb).pvalue if (d != 0).any() else 1.0
            except ValueError: pw = np.nan
            ST.append({"Baseline": b, "Metric": met, "Proposed": a.mean(), "Baseline mean": bb.mean(), "Diff": d.mean(),
                       "Runs proposed better": f"{int(better.sum())}/{len(d)}", "paired t p": ttest_rel(a, bb).pvalue if d.std() > 0 else np.nan,
                       "Wilcoxon p": pw, "Nadeau-Bengio p": corrected_t(d, n_tr, n_te)})
    ST = pd.DataFrame(ST); STATE["ST_A"] = ST; print(ST.round(4).to_string())
    y_all = df.label.values; OM = {}
    for n in NAMES:
        stk = [OOF[s][n] for s in CFG["seeds"] if not np.isnan(OOF[s][n]).any()]
        if stk: OM[n] = np.mean(stk, 0)
    STATE["OOF_MEAN"] = OM
    if PROPOSED in OM:
        STATE["OOF_A"] = to_pct_table([dict(full_metrics(y_all, OM[n]), Model=n) for n in OM])
        STATE["MCN_A"] = pd.DataFrame([dict(zip(["only proposed right", "only baseline right", "exact McNemar p"],
                                                mcnemar(y_all, OM[PROPOSED], OM[b])), Baseline=b) for b in OM if b != PROPOSED])
        print(STATE["OOF_A"][["Accuracy", "F1 (macro)", "AUC-ROC (macro)", "ECE", "Acc @ 95% coverage"]].to_string()); print(STATE["MCN_A"].to_string())
    # figures
    rp2 = rp.assign(M=rp.Model.map(short)); order = [short[n] for n in NAMES if n in set(rp.Model)]
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.2))
    for a, c in zip(ax, ["Accuracy", "F1 (macro)", "ECE"]):
        sns.boxplot(data=rp2, x="M", y=c, order=order, palette=[pal[n] for n in NAMES if short[n] in order], ax=a)
        sns.stripplot(data=rp2, x="M", y=c, order=order, color="k", size=3, ax=a); a.set_xlabel(""); a.tick_params(axis="x", rotation=30)
        a.set_title(f"{c} — leakage-free repeated CV")
    savefig(f"{RES}/A_prior_work/boxplots.png")
    tab = SUM[["Reported in original paper (%)", "Accuracy", "F1 (macro)", "AUC-ROC (macro)", "Cohen Kappa", "ECE"]]
    table_img(tab, "Prior architectures re-implemented under the same leakage-free protocol (mean ± std over runs)",
              f"{RES}/A_prior_work/table_prior_work.png", labels=[short[i] for i in tab.index])
    if OM:
        fig, ax = plt.subplots(1, len(OM), figsize=(3.9 * len(OM), 4)); ax = np.atleast_1d(ax)
        for a, n in zip(ax, OM):
            sns.heatmap(confusion_matrix(y_all, OM[n].argmax(1)), annot=True, fmt="d", cmap="Blues", cbar=False, ax=a,
                        xticklabels=["N", "Pap", "Ps"], yticklabels=["N", "Pap", "Ps"])
            a.set_title(f"{short[n]}\nOOF {accuracy_score(y_all, OM[n].argmax(1))*100:.2f}%", fontsize=9)
        savefig(f"{RES}/A_prior_work/oof_confusion_matrices.png")

if CFG["run_A"]: step("Part A", part_A)

# %% [markdown]
# ## Part B — label-noise audit (confident learning on out-of-fold predictions)

# %%
def part_B():
    OOF = STATE.get("OOF")
    if not OOF: print("Part B needs Part A"); return
    y = df.label.values
    sets = {(n, s): OOF[s][n] for s in CFG["seeds"] for n in NAMES if not np.isnan(OOF[s][n]).any()}
    P = np.stack(list(sets.values())); pm = P.mean(0)
    thr = np.array([pm[y == k, k].mean() for k in range(3)])                       # per-class self-confidence thresholds
    above = pm >= thr[None, :]
    cl_flag = np.array([above[i].any() and pm[i].argmax() != y[i] and above[i, pm[i].argmax()] for i in range(len(y))])
    all_wrong = (P.argmax(2) != y[None, :]).all(0)
    conflict = set(dups[~dups.same_label].i) | set(dups[~dups.same_label].j) if "dups" in globals() else set()
    conflict_strict = set()
    if "dups" in globals():
        cc = dups[(~dups.same_label) & (dups.hamming <= 2)]; conflict_strict = set(cc.i) | set(cc.j)
    fl = pd.DataFrame({"path": df.path, "label": df.label_name, "consensus_pred": [CLASSES[k] for k in pm.argmax(1)],
                       "p(given label)": pm[np.arange(len(y)), y], "p(consensus)": pm.max(1),
                       "wrong in every model & seed": all_wrong, "confident-learning flag": cl_flag,
                       "exact duplicate with different label": [i in conflict_strict for i in range(len(y))]})
    flagged = fl[fl["wrong in every model & seed"] | fl["confident-learning flag"] | fl["exact duplicate with different label"]]
    flagged = flagged.sort_values("p(given label)"); flagged.to_csv(f"{RES}/B_label_noise/flagged_images.csv")
    print(f"Prediction sets used: {len(sets)}  flagged: {len(flagged)}"); print(flagged.to_string())
    STATE["flagged"] = flagged
    # sensitivity analysis — for each model, flags computed WITHOUT that model (no circularity)
    OM = STATE.get("OOF_MEAN", {}); rows = []
    for n in OM:
        others = np.stack([v for (m, s), v in sets.items() if m != n])
        w_others = (others.argmax(2) != y[None, :]).all(0)
        keep = ~w_others
        acc_all = (OM[n].argmax(1) == y).mean() * 100; acc_k = (OM[n].argmax(1)[keep] == y[keep]).mean() * 100
        f1_k = precision_recall_fscore_support(y[keep], OM[n].argmax(1)[keep], average="macro", zero_division=0)[2] * 100
        rows.append({"Model": n, "OOF acc (all 1,369)": round(acc_all, 2), "images excluded": int((~keep).sum()),
                     "OOF acc (suspected label errors excluded)": round(acc_k, 2), "OOF F1 (excluded)": round(f1_k, 2)})
    SENS = pd.DataFrame(rows).set_index("Model"); STATE["SENS"] = SENS; print(SENS.to_string())
    SENS.to_csv(f"{RES}/B_label_noise/sensitivity_analysis.csv")
    if len(flagged):
        k = min(len(flagged), 14); fig, ax = plt.subplots(1, k, figsize=(2.3 * k, 3.2)); ax = np.atleast_1d(ax)
        for a, (_, r) in zip(ax, flagged.head(k).iterrows()):
            a.imshow(Image.open(r.path).convert("RGB")); a.axis("off")
            a.set_title(f"label: {r.label}\nmodels: {r.consensus_pred}\np(label)={r['p(given label)']:.3f}", fontsize=7, color="red")
        plt.suptitle("Images whose label is contradicted by all models (candidate label errors — to be checked by an ophthalmologist)", fontweight="bold")
        savefig(f"{RES}/B_label_noise/flagged_images.jpg")

if CFG["run_B"]: step("Part B", part_B)

# %% [markdown]
# ## Part C — external validation on RFMiD (automatic disc localisation)

# %%
def _ds_root(p):
    parts = os.path.relpath(p, CFG["data_root"]).split(os.sep)
    return os.path.join(CFG["data_root"], *parts[:3]) if parts[0] == "datasets" else os.path.join(CFG["data_root"], parts[0])

def find_external():
    """External fundus datasets (full fundus photographs, NOT the Kim dataset):
       * RFMiD 1.0 / 2.0 (CSV with an 'ODE' column): ODE → elevated (1); Disease_Risk==0 or WNL==1 → normal (0);
         TD (tilted disc, a classical cause of pseudopapilledema) without ODE → 'pseudo-like' (-1)
       * JSIEC-1000 (folders '<n>.<name>'): 'Disc swelling and elevation' → 1; 'Normal' → 0; 'Congenital disc abnormality' → -1"""
    rows = []
    IMG = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    for csvp in glob.glob(os.path.join(CFG["data_root"], "**", "*.csv"), recursive=True):
        try: h = pd.read_csv(csvp, nrows=2)
        except Exception: continue
        if "ODE" not in h.columns or not any(c in h.columns for c in ["ID", "ImageID", "Id"]): continue
        lab = pd.read_csv(csvp); idc = [c for c in ["ID", "ImageID", "Id"] if c in lab.columns][0]
        low = os.path.basename(csvp).lower()
        key = "train" if "train" in low else ("valid" if ("valid" in low or "eval" in low) else ("test" if "test" in low else ""))
        imgs = [p for p in glob.glob(os.path.join(os.path.dirname(csvp), "**", "*"), recursive=True) if p.lower().endswith(IMG)]
        idx = {os.path.splitext(os.path.basename(p))[0]: p for p in imgs}
        if sum(str(i) in idx for i in lab[idc]) < 0.5 * len(lab):              # images stored elsewhere: match by split keyword
            imgs = [p for p in glob.glob(os.path.join(_ds_root(csvp), "**", "*"), recursive=True)
                    if p.lower().endswith(IMG) and (key in p.lower() or not key)]
            idx = {os.path.splitext(os.path.basename(p))[0]: p for p in imgs}
        src = "RFMiD:" + os.path.basename(_ds_root(csvp))
        for _, r in lab.iterrows():
            p = idx.get(str(r[idc]))
            if p is None: continue
            if r.get("ODE", 0) == 1: rows.append((p, 1, key, src))
            elif r.get("Disease_Risk", 1) == 0 or r.get("WNL", 0) == 1: rows.append((p, 0, key, src))
            elif r.get("TD", 0) == 1: rows.append((p, -1, key, src))
    import re
    for d in glob.glob(os.path.join(CFG["data_root"], "**", ""), recursive=True):
        name = os.path.basename(os.path.normpath(d)); m = re.match(r"^\d+(\.\d+)*\.(.+)$", name)
        if not m: continue
        nm = m.group(2).strip().lower(); lab_ = 1 if "disc swelling" in nm else (0 if nm == "normal" else (-1 if "congenital disc" in nm else None))
        if lab_ is None: continue
        for p in glob.glob(os.path.join(d, "*")):
            if p.lower().endswith(IMG): rows.append((p, lab_, "", "JSIEC-1000"))
    return pd.DataFrame(rows, columns=["path", "elevated", "split", "source"]).drop_duplicates("path")

def locate_disc(img):
    """Classical optic-disc localisation: bright region where vessels converge, inside the field of view."""
    h0, w0 = img.shape[:2]; sc = 800 / max(h0, w0); im = cv2.resize(img, (int(w0 * sc), int(h0 * sc)))
    r = im[..., 0].astype(np.float32); g = im[..., 1].astype(np.float32)
    fov = (r > 20).astype(np.uint8); n, lab, st, _ = cv2.connectedComponentsWithStats(fov)
    if n > 1: fov = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    ys, xs = np.nonzero(fov); R = 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min())
    inner = cv2.erode(fov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(R * 0.12) | 1, int(R * 0.12) | 1)))
    gc_ = cv2.createCLAHE(2.0, (8, 8)).apply(g.astype(np.uint8)).astype(np.float32)
    k = int(R * 0.15) | 1
    vessels = cv2.morphologyEx(gc_, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bright = cv2.GaussianBlur((0.5 * r + g), (0, 0), R / 20); vdens = cv2.GaussianBlur(vessels, (0, 0), R / 12)
    nz = lambda a: (a - a[inner > 0].min()) / (a[inner > 0].max() - a[inner > 0].min() + 1e-6)
    score = (0.6 * nz(bright) + 0.4 * nz(vdens)) * inner
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    return cx / sc, cy / sc, R / sc

def crop_disc(path):
    img = np.array(Image.open(path).convert("RGB")); cx, cy, R = locate_disc(img)
    half = CFG["crop_frac"] * R / 2
    x0, y0 = int(round(cx - half)), int(round(cy - half)); x1, y1 = int(round(cx + half)), int(round(cy + half))
    pad = int(half) + 2; imgp = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(imgp[y0 + pad:y1 + pad, x0 + pad:x1 + pad], (240, 240), interpolation=cv2.INTER_AREA)

def part_C():
    with open(f"{RES}/C_external/attached_inputs.txt", "w") as fh:          # diagnostic: what datasets are attached?
        for d in sorted(glob.glob(os.path.join(CFG["data_root"], "*", "*", "*")) + glob.glob(os.path.join(CFG["data_root"], "*"))):
            fh.write(d + "\n")
        for csvp in glob.glob(os.path.join(CFG["data_root"], "**", "*.csv"), recursive=True):
            try: fh.write(f"CSV {csvp}: {list(pd.read_csv(csvp, nrows=1).columns)[:12]}\n")
            except Exception as e: fh.write(f"CSV {csvp}: unreadable {e}\n")
    print(open(f"{RES}/C_external/attached_inputs.txt").read())
    ext = find_external()
    if not len(ext) or not ((ext.elevated == 1).any() and (ext.elevated == 0).any()):
        msg = "No external data found — add 'retinal-disease-classification' (andrewmvd, RFMiD) and/or JSIEC-1000 to run Part C"
        print(msg); FAILED.append("Part C: " + msg); return
    print(ext.groupby(["source", "elevated"]).size()); ext.groupby(["source", "elevated"]).size().to_csv(f"{RES}/C_external/external_counts.csv")
    cdir = f"{WORK}/rfmid_crops"; os.makedirs(cdir, exist_ok=True); paths = []
    for i, r in ext.reset_index(drop=True).iterrows():
        out = f"{cdir}/{i}_{r.elevated}.png"
        try: Image.fromarray(crop_disc(r.path)).save(out); paths.append(out)
        except Exception as e: print("crop failed:", r.path, e); paths.append(None)
    ext["crop"] = paths; ext = ext.dropna(subset=["crop"]).reset_index(drop=True)
    # visual QA of the automatic localisation
    qa = pd.concat([ext[ext.elevated == 1].head(20), ext[ext.elevated == 0].head(20), ext[ext.elevated == -1].head(8)])
    fig, ax = plt.subplots(6, 8, figsize=(16, 12.5))
    for a, (_, r) in zip(ax.ravel(), qa.iterrows()):
        a.imshow(Image.open(r.crop)); a.set_title({1: "ODE", 0: "normal", -1: "pseudo-like"}[r.elevated] + f" | {r.source[:14]}", fontsize=6); a.axis("off")
    for a in ax.ravel()[len(qa):]: a.axis("off")
    plt.suptitle("RFMiD — automatic optic-disc crops (first 24 ODE, first 24 normal) — visual check", fontweight="bold")
    savefig(f"{RES}/C_external/crop_quality_check.jpg")
    # train final models on ALL Kim images (group-aware 10 % validation for early stopping)
    a_i, v_i = next(StratifiedGroupKFold(10, shuffle=True, random_state=42).split(df, df.label, df.group))
    tr, va = df.iloc[a_i], df.iloc[v_i]
    ext_df = pd.DataFrame({"path": ext.crop, "label": np.where(ext.elevated == 1, 1, 0), "label_name": "x"})
    B = (ext.elevated >= 0).values                              # binary evaluation set (ODE vs normal)
    res, P = [], {}
    for nme in ["DenseNet121 (baseline)", PROPOSED]:
        if time_left_h() < 0.4: print("⏰ no time left for Part C"); break
        CFG["seed"] = 42; model, _, _ = train(MODELS[nme], tr, va, f"[C] {nme}")
        _, prob = predict(model, make_loader(ext_df, False), tta=True); P[nme] = prob
        if nme == PROPOSED: torch.save(model.state_dict(), f"{WORK}/polarfuse_final_all_data.pth")
        del model; free()
        P[nme + "__all"] = prob; prob = prob[B]
        yv = ext.elevated.values[B]; score = 1 - prob[:, 0]; pred = prob.argmax(1) != 0
        auc = roc_auc_score(yv, score); rng = np.random.default_rng(0); bs = []
        for _ in range(CFG["n_bootstrap"]):
            i = rng.integers(0, len(yv), len(yv))
            if len(np.unique(yv[i])) == 2: bs.append(roc_auc_score(yv[i], score[i]))
        tp = (pred & (yv == 1)).sum(); fn = (~pred & (yv == 1)).sum(); tn = (~pred & (yv == 0)).sum(); fp = (pred & (yv == 0)).sum()
        res.append({"Model": nme, "n ODE": int((yv == 1).sum()), "n normal": int((yv == 0).sum()),
                    "AUC (elevated disc)": round(auc, 4), "AUC 95% CI": f"{np.percentile(bs, 2.5):.3f}–{np.percentile(bs, 97.5):.3f}",
                    "Sensitivity": round(tp / max(tp + fn, 1), 4), "Specificity": round(tn / max(tn + fp, 1), 4),
                    "Accuracy": round((tp + tn) / len(yv), 4),
                    "ODE predicted as Papilledema": int(((prob.argmax(1) == 1) & (yv == 1)).sum()),
                    "ODE predicted as Pseudo": int(((prob.argmax(1) == 2) & (yv == 1)).sum()),
                    **{f"AUC {sname}": round(roc_auc_score(yv[ext.source.values[B] == sname], score[ext.source.values[B] == sname]), 4)
                       for sname in ext.source.unique() if len(np.unique(yv[ext.source.values[B] == sname])) == 2}})
        if (ext.elevated == -1).any():
            pa = P[nme + "__all"][(ext.elevated == -1).values].argmax(1)
            STATE.setdefault("PSEUDO_LIKE", []).append({"Model": nme, "n pseudo-like (tilted / congenital disc)": int(len(pa)),
                **{f"predicted {c}": int((pa == k).sum()) for k, c in enumerate(CLASSES)}})
    if not res: return
    EXT = pd.DataFrame(res).set_index("Model"); STATE["EXT"] = EXT; print(EXT.to_string()); EXT.to_csv(f"{RES}/C_external/external_results.csv")
    P = {k: v[B] for k, v in P.items() if not k.endswith("__all")}     # binary evaluation subset only
    if "PSEUDO_LIKE" in STATE: STATE["PSEUDO_LIKE"] = pd.DataFrame(STATE["PSEUDO_LIKE"]).set_index("Model"); print(STATE["PSEUDO_LIKE"].to_string())
    if len(P) == 2:
        yv = ext.elevated.values[B]; a_s, b_s = 1 - P[PROPOSED][:, 0], 1 - P["DenseNet121 (baseline)"][:, 0]; rng = np.random.default_rng(1); d = []
        for _ in range(CFG["n_bootstrap"]):
            i = rng.integers(0, len(yv), len(yv))
            if len(np.unique(yv[i])) == 2: d.append(roc_auc_score(yv[i], a_s[i]) - roc_auc_score(yv[i], b_s[i]))
        d = np.array(d); STATE["EXT_DIFF"] = {"AUC difference (proposed − baseline)": float(np.mean(d)),
                                              "95% CI": f"{np.percentile(d, 2.5):.4f} – {np.percentile(d, 97.5):.4f}",
                                              "bootstrap p (two-sided)": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}
        print(STATE["EXT_DIFF"])
    plt.figure(figsize=(6, 5.5))
    for nme, prob in P.items():
        yb_ = ext.elevated.values[B]; fpr, tpr, _ = roc_curve(yb_, 1 - prob[:, 0]); plt.plot(fpr, tpr, lw=2, label=f"{short[nme]} (AUC {roc_auc_score(yb_, 1 - prob[:, 0]):.3f})")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.grid(alpha=.3)
    plt.title("External validation (RFMiD): optic-disc edema vs normal"); savefig(f"{RES}/C_external/external_roc.png")
    ext[B].drop(columns=["crop"]).assign(**{f"p_{c}_{short[n]}": P[n][:, k] for n in P for k, c in enumerate(CLASSES)}).to_csv(
        f"{RES}/C_external/external_predictions.csv", index=False)

if CFG["run_C"]: step("Part C", part_C)

# %% [markdown]
# ## Part D — quantitative explainability: how much attribution falls on the disc margin?

# %%
def disc_margin_polar(img):
    """Disc boundary radius for every angle, found in polar space as the strongest bright-to-dark radial edge."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    pol = cv2.GaussianBlur(to_polar(g), (0, 0), sigmaX=4, sigmaY=6)
    d = np.diff(pol, axis=1); lo, hi = int(0.12 * S), int(0.85 * S)
    rb = lo + np.argmin(d[:, lo:hi], axis=1).astype(np.float32)
    rb_pad = np.concatenate([rb[-15:], rb, rb[:15]]); rb = np.array([np.median(rb_pad[i:i + 31]) for i in range(S)])
    return rb

def margin_enrichment(cam_pol, rb, w=0.08):
    cols = np.arange(S)[None, :]; band = np.abs(cols - rb[:, None]) <= w * S
    tot = cam_pol.sum()
    return np.nan if tot <= 0 else (cam_pol[band].sum() / tot) / band.mean()

def part_D():
    cm, te = STATE.get("cam_models", {}), STATE.get("cam_test")
    if PROPOSED not in cm or "DenseNet121 (baseline)" not in cm: print("Part D needs Part A models"); return
    mp, mb = cm[PROPOSED].eval(), cm["DenseNet121 (baseline)"].eval(); rows = []; examples = []
    for _, r in te.iterrows():
        img = np.array(Image.open(r.path).convert("RGB").resize((S, S))); pol = to_polar(img); rb = disc_margin_polar(img)
        xc = norm(Image.fromarray(img))[None].to(DEVICE); xp = norm(Image.fromarray(pol))[None].to(DEVICE)
        with torch.enable_grad():
            o = mp(xc, xp); p = final_probs(o, True); c = p.argmax(1).item(); mp.zero_grad(); torch.log(p[0, c] + 1e-8).backward()
        pol_cam = up_p(hirescam(mp.feats["polar"])); cart_cam = up_c(hirescam(mp.feats["cart"]))
        with torch.enable_grad():
            o2 = mb(xc, xp); p2 = final_probs(o2, False); c2 = p2.argmax(1).item(); mb.zero_grad(); torch.log(p2[0, c2] + 1e-8).backward()
        base_cam = up_c(_hires_base(mb))
        to_pol = lambda c_: to_polar(c_.astype(np.float32))
        comb = (to_pol(cart_cam) / (to_pol(cart_cam).max() + 1e-8) + pol_cam) / 2
        rows.append({"label": r.label_name, "PolarFuse-Net polar branch": margin_enrichment(pol_cam, rb),
                     "PolarFuse-Net Cartesian branch": margin_enrichment(to_pol(cart_cam), rb),
                     "PolarFuse-Net combined": margin_enrichment(comb, rb),
                     "DenseNet121 baseline": margin_enrichment(to_pol(base_cam), rb)})
        if sum(e[3] == r.label_name for e in examples) < 3: examples.append((img, rb, from_polar(pol_cam.astype(np.float32), S), r.label_name, base_cam))
    D = pd.DataFrame(rows); D.to_csv(f"{RES}/D_margin_focus/margin_enrichment_per_image.csv", index=False)
    cols = [c for c in D.columns if c != "label"]
    SUMD = pd.DataFrame({c: {"mean": D[c].mean(), "std": D[c].std(), "median": D[c].median()} for c in cols}).T
    for c in cols:
        if c != "DenseNet121 baseline":
            dd = D[[c, "DenseNet121 baseline"]].dropna(); SUMD.loc[c, "Wilcoxon p vs DenseNet121"] = wilcoxon(dd[c], dd["DenseNet121 baseline"]).pvalue
    SUMD.loc[:, "by class (mean)"] = [", ".join(f"{k}: {v:.2f}" for k, v in D.groupby("label")[c].mean().items()) for c in SUMD.index]
    STATE["SUMD"] = SUMD; print(SUMD.to_string()); SUMD.to_csv(f"{RES}/D_margin_focus/margin_enrichment_summary.csv")
    plt.figure(figsize=(9, 4.5)); m = D.melt(id_vars="label", value_vars=cols, var_name="attribution", value_name="margin enrichment")
    sns.boxplot(data=m, x="attribution", y="margin enrichment"); plt.axhline(1, color="r", ls="--", label="no preference for the margin")
    plt.xticks(rotation=12); plt.legend(); plt.title("Attribution concentration on the disc-margin band (>1 = enriched)")
    savefig(f"{RES}/D_margin_focus/margin_enrichment.png")
    if examples:
        n = len(examples); fig, ax = plt.subplots(3, n, figsize=(2.5 * n, 7.6)); ax = np.array(ax).reshape(3, n)
        for k, (img, rb, back, lab_, bcam) in enumerate(examples):
            th = np.arange(S) / S * 2 * np.pi; rr = rb / S * (S / 2)
            ax[0, k].imshow(img); ax[0, k].plot(S / 2 + rr * np.cos(th), S / 2 + rr * np.sin(th), "c-", lw=1); ax[0, k].set_title(lab_, fontsize=8)
            ax[1, k].imshow(img); ax[1, k].imshow(back, cmap="jet", alpha=.45, vmin=0, vmax=1)
            ax[2, k].imshow(img); ax[2, k].imshow(bcam, cmap="jet", alpha=.45, vmin=0, vmax=1)
            for i in range(3): ax[i, k].axis("off")
        ax[0, 0].set_title("detected margin\n" + examples[0][3], fontsize=8)
        fig.text(0.005, 0.83, "auto margin", rotation=90, fontsize=8); fig.text(0.005, 0.5, "PolarFuse polar", rotation=90, fontsize=8)
        fig.text(0.005, 0.17, "DenseNet121", rotation=90, fontsize=8)
        plt.suptitle("Automatic disc-margin detection and attribution maps", fontweight="bold"); savefig(f"{RES}/D_margin_focus/margin_examples.jpg")

def _hires_base(mb):
    f = mb.feats["cart"]
    if f.ndim == 4 and f.shape[-1] > f.shape[1] and f.shape[1] == f.shape[2]: return hirescam(f.permute(0, 3, 1, 2))
    return hirescam(f)

if CFG["run_D"]: step("Part D", part_D)

# %% ---------------- Save everything + zip ----------------
def save_all():
    with pd.ExcelWriter(f"{RES}/ALL_RESULTS_final.xlsx") as xw:
        AUDIT.to_frame("value").to_excel(xw, sheet_name="Audit")
        for key, sheet in [("SUM_A", "A_mean_std"), ("ST_A", "A_paired_stats"), ("OOF_A", "A_OOF_all_images"), ("MCN_A", "A_McNemar"),
                           ("SENS", "B_sensitivity"), ("flagged", "B_flagged_images"), ("EXT", "C_external"), ("PSEUDO_LIKE", "C_pseudo_like"), ("SUMD", "D_margin_focus")]:
            if key in STATE and STATE[key] is not None and len(STATE[key]):
                STATE[key].to_excel(xw, sheet_name=sheet, index=key not in ("ST_A", "MCN_A"))
        if "runs_pct" in STATE: STATE["runs_pct"].round(4).to_excel(xw, sheet_name="A_all_runs", index=False)
        if "EXT_DIFF" in STATE: pd.Series(STATE["EXT_DIFF"]).to_frame("value").to_excel(xw, sheet_name="C_AUC_difference")
        pd.DataFrame(list({k: str(v) for k, v in CFG.items()}.items()), columns=["param", "value"]).to_excel(xw, sheet_name="Config", index=False)
step("save excel", save_all)
figs = [f"{RES}/A_prior_work/table_prior_work.png", f"{RES}/A_prior_work/boxplots.png", f"{RES}/A_prior_work/oof_confusion_matrices.png",
        f"{RES}/B_label_noise/flagged_images.jpg", f"{RES}/C_external/external_roc.png", f"{RES}/C_external/crop_quality_check.jpg",
        f"{RES}/D_margin_focus/margin_enrichment.png", f"{RES}/D_margin_focus/margin_examples.jpg", f"{RES}/audit/top_pairs.jpg"]
step("overview", lambda: montage(figs, f"{RES}/_OVERVIEW_final.jpg", "PolarFuse-Net - final experiments", 2, 1600))
with open(f"{RES}/README.txt", "w") as fh:
    fh.write(f"CODE VERSION: {CODE_VERSION}\nTotal time {(time.time()-T_START)/3600:.2f} h\n\nAUDIT\n{AUDIT.to_string()}\n")
    for key in ["SUM_A", "ST_A", "OOF_A", "MCN_A", "SENS", "flagged", "EXT", "PSEUDO_LIKE", "SUMD"]:
        if key in STATE and STATE[key] is not None and len(STATE[key]):
            fh.write(f"\n{key}\n" + (STATE[key].round(4) if hasattr(STATE[key], "round") else STATE[key]).to_string() + "\n")
    if "EXT_DIFF" in STATE: fh.write(f"\nEXT_DIFF {STATE['EXT_DIFF']}\n")
    if FAILED: fh.write("\nFAILED STEPS\n" + "\n".join(FAILED) + "\n")
if FAILED: open(f"{RES}/FAILED_STEPS.txt", "w").write("\n".join(FAILED))
zp = shutil.make_archive(f"{WORK}/results_external", "zip", RES)
print(f"\n✅ saved {zp} ({os.path.getsize(zp)/1e6:.1f} MB) — download ONLY results_external.zip   total {(time.time()-T_START)/3600:.2f} h")
