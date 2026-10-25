# %% [markdown]
# # PolarFuse-Net v2 — paper figures (≈ 15–20 min on a T4)
# Trains the proposed model once on the fixed 70/15/15 split (seed 42, same split as all previous notebooks)
# and produces the figures for the paper:
#
# 1. Architecture diagram
# 2. Cartesian vs Polar view examples (per class)
# 3. **Fixed explainability**: HiResCAM for **both** branches — the polar CAM is shown in polar space *and* mapped back
#    onto the fundus image, plus a combined map. Fixed colour scale (0–1), so an empty map really looks empty.
# 4. Misclassified test images with their class probabilities
# 5. Hierarchical-head analysis: how well the second-stage head separates Papilledema from Pseudopapilledema
# 6. Confusion matrix, ROC/PR, per-class metrics, training curves, t-SNE
#
# Everything → **`results_paper_figures.zip`** (+ the trained weights `polarfuse_v2_best.pth` in /kaggle/working).
#
# Kaggle: GPU + Internet ON → Run All (short enough to run interactively, or Save & Run All).

# %%
import os, glob, random, time, copy, shutil, math, json, gc
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.patches import FancyBboxPatch
import cv2
from PIL import Image, ImageDraw, ImageFont

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, cohen_kappa_score, matthews_corrcoef, log_loss)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from scipy.stats import binomtest

# %% ---------------- Config ----------------
WORK = "/kaggle/working" if os.path.exists("/kaggle/working") else "./outputs"
CFG = dict(
    data_root    = "/kaggle/input",
    img_size     = 224,
    batch_size   = 24,
    epochs       = 40,
    patience     = 10,
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
    pretrained   = True,
    save_weights = False,
    n_cam_per_class = 4,        # correctly classified examples per class in the Grad-CAM figure
    results_dir  = f"{WORK}/results_paper_figures",
    weights_dir  = f"{WORK}",
)
RES = CFG["results_dir"]
if os.path.exists(RES): shutil.rmtree(RES)
for d in [RES, f"{RES}/data"]: os.makedirs(d, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CODE_VERSION = "figures-v2.1 (remap fix for polar CAM)"
print("CODE VERSION:", CODE_VERSION)
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



# %% [markdown]
# ## Train the proposed model (PolarFuse-Net v2) on the fixed split

# %%
PROPOSED = "PolarFuse-Net v2"
P_CFG = dict(kind="dual", views="dual", fusion="concat", hier=True, supcon=True, share=False, backbone="densenet121", lr_bb=1e-4)
# If you add the weights from the previous run as a Kaggle dataset (any folder under /kaggle/input containing
# polarfuse_v2_best.pth), they are loaded and training is skipped -> same model, same numbers as before.
_w = glob.glob(os.path.join(CFG["data_root"], "**", "polarfuse_v2_best.pth"), recursive=True)
if _w:
    print("Loading weights:", _w[0])
    model = build(P_CFG).to(DEVICE); model.load_state_dict(torch.load(_w[0], map_location=DEVICE))
    hist = pd.DataFrame(columns=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1"])
    info = {"Params (M)": sum(p.numel() for p in model.parameters()) / 1e6, "Train time (min)": np.nan,
            "Best epoch": np.nan, "Inference (ms/img)": inference_ms(model)}
else:
    model, hist, info = train(P_CFG, train_df, val_df, PROPOSED)
test_dl = make_loader(test_df, False)
Y, PROB = predict(model, test_dl, tta=True)
M = overall_metrics(Y, PROB); M.update(info)
print(classification_report(Y, PROB.argmax(1), target_names=CLASSES, digits=4))
print(pd.Series(M).to_string())
FD = RES
FAILED = []
wrong = test_df.iloc[:0]; diag = pd.DataFrame({"empty_cart": [], "empty_polar": []}); auc1 = auc2 = np.nan

# %% ---------------- Fig 1 — architecture diagram ----------------
try:
    def box(ax, x, y, w, h, text, fc, fs=9, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)
    def arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.4, color="#333"))

    fig, ax = plt.subplots(figsize=(15, 6.2)); ax.set_xlim(0, 15); ax.set_ylim(0, 6.2); ax.axis("off")
    ex = np.array(Image.open(test_df.path.iloc[0]).convert("RGB").resize((S, S)))
    ax.imshow(ex, extent=(0.1, 1.7, 3.6, 5.2)); ax.text(0.9, 5.35, "Input fundus\n(disc-centred)", ha="center", fontsize=9)
    ax.imshow(to_polar(ex), extent=(2.3, 3.9, 0.6, 2.2)); ax.text(3.1, 0.3, "Polar view\n(rows = angle, cols = radius)", ha="center", fontsize=8)
    box(ax, 2.3, 3.75, 1.6, 1.3, "Cartesian view\n224×224", "#dbe9f6")
    arrow(ax, 1.7, 4.4, 2.3, 4.4); arrow(ax, 0.9, 3.6, 2.3, 1.4)
    ax.text(1.0, 2.5, "polar\ntransform", fontsize=8, ha="center", style="italic")
    box(ax, 4.5, 3.75, 2.0, 1.3, "DenseNet121\n(ImageNet)", "#c6dbef", bold=True)
    box(ax, 4.5, 0.75, 2.0, 1.3, "DenseNet121\n(ImageNet)", "#fdd0a2", bold=True)
    arrow(ax, 3.9, 4.4, 4.5, 4.4); arrow(ax, 3.9, 1.4, 4.5, 1.4)
    box(ax, 7.0, 3.75, 1.7, 1.3, "1×1 Conv-BN-GELU\n→ 256-d\n+ GAP", "#e5f5e0")
    box(ax, 7.0, 0.75, 1.7, 1.3, "1×1 Conv-BN-GELU\n→ 256-d\n+ GAP", "#e5f5e0")
    arrow(ax, 6.5, 4.4, 7.0, 4.4); arrow(ax, 6.5, 1.4, 7.0, 1.4)
    box(ax, 9.2, 2.25, 1.4, 1.3, "Concat\n512-d", "#fff7bc", bold=True)
    arrow(ax, 8.7, 4.4, 9.2, 3.2); arrow(ax, 8.7, 1.4, 9.2, 2.6)
    box(ax, 11.2, 4.75, 3.5, 0.95, "Flat head: Normal / Pap / Pseudo", "#fcbba1")
    box(ax, 11.2, 3.45, 3.5, 0.95, "Stage-1 head: Normal vs Elevated disc", "#fcbba1")
    box(ax, 11.2, 2.15, 3.5, 0.95, "Stage-2 head: Papilledema vs Pseudo", "#fcbba1")
    box(ax, 11.2, 0.65, 3.5, 1.05, "Projection → SupCon loss\n(training only)", "#d9d9d9")
    for yy in [5.2, 3.9, 2.6, 1.2]: arrow(ax, 10.6, 2.9, 11.2, yy)
    ax.text(12.95, 6.0, "Final prob = ½ (flat + hierarchical)  +  flip-TTA", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("PolarFuse-Net v2 — Cartesian–Polar dual-view network with hierarchical clinical heads", fontsize=12, fontweight="bold")
    savefig(f"{FD}/fig1_architecture.png")
except Exception as e:
    import traceback; traceback.print_exc(); FAILED.append('Fig 1 — architecture diagram: ' + repr(e))

# %% ---------------- Fig 2 — Cartesian vs Polar examples ----------------
try:
    fig, ax = plt.subplots(3, 6, figsize=(17, 9))
    for r, c in enumerate(CLASSES):
        for k, p in enumerate(test_df[test_df.label_name == c].sample(3, random_state=3).path):
            im = np.array(Image.open(p).convert("RGB").resize((S, S)))
            ax[r, 2 * k].imshow(im); ax[r, 2 * k].set_title(f"{c}", fontsize=9)
            ax[r, 2 * k + 1].imshow(to_polar(im), extent=(0, 1, 360, 0), aspect="auto")
            ax[r, 2 * k + 1].set_title("Polar", fontsize=9); ax[r, 2 * k + 1].set_xlabel("radius (centre → edge)", fontsize=7)
            ax[r, 2 * k + 1].set_ylabel("angle (°)", fontsize=7); ax[r, 2 * k + 1].tick_params(labelsize=6)
            ax[r, 2 * k].axis("off")
    plt.suptitle("Cartesian vs polar view — in polar space the disc margin becomes a vertical band, so margin blurring "
                 "and vessel obscuration appear as column-wise patterns", fontsize=10)
    savefig(f"{FD}/fig2_cartesian_vs_polar.jpg")
except Exception as e:
    import traceback; traceback.print_exc(); FAILED.append('Fig 2 — Cartesian vs Polar examples: ' + repr(e))

# %% ---------------- Fig 3 — explainability (fixed) ----------------
try:
    # HiResCAM (Draelos & Carin, 2020): element-wise gradient × activation, summed over channels.
    # Unlike Grad-CAM it does not average the gradient spatially first, so positive and negative evidence
    # at different locations cannot cancel into an all-zero map. Target = log of the FINAL probability of the
    # predicted class (flat + hierarchical heads), i.e. exactly what the model outputs.
    def hirescam(feat):
        c = torch.relu((feat.grad * feat.detach()).sum(1))[0].float().cpu().numpy()
        return c

    def upsample_cart(c):
        c = cv2.resize(c, (S, S), interpolation=cv2.INTER_CUBIC); c = np.clip(c, 0, None)
        return c / c.max() if c.max() > 0 else c

    def upsample_polar(c):
        # angle axis (rows) is circular: pad with wrap before resizing so 0° and 360° join smoothly
        n = c.shape[0]; cp = np.vstack([c[-1:], c, c[:1]])
        big = cv2.resize(cp, (S, int(round(S * (n + 2) / n))), interpolation=cv2.INTER_CUBIC)
        off = (big.shape[0] - S) // 2; big = np.clip(big[off:off + S], 0, None)
        return big / big.max() if big.max() > 0 else big

    def explain(model, path):
        model.eval()
        img = np.array(Image.open(path).convert("RGB").resize((S, S))); pol = to_polar(img)
        xc = norm(Image.fromarray(img))[None].to(DEVICE); xp = norm(Image.fromarray(pol))[None].to(DEVICE)
        with torch.enable_grad():
            out = model(xc, xp); p = final_probs(out, True); cls = p.argmax(1).item()
            model.zero_grad(); torch.log(p[0, cls] + 1e-8).backward()
        cc_raw, cp_raw = hirescam(model.feats["cart"]), hirescam(model.feats["polar"])
        cc, cpol = upsample_cart(cc_raw), upsample_polar(cp_raw)
        cp_back = from_polar(cpol.astype(np.float32), S); cp_back = cp_back / cp_back.max() if cp_back.max() > 0 else cp_back
        comb = (cc + cp_back) / 2; comb = comb / comb.max() if comb.max() > 0 else comb
        return dict(img=img, pol=pol, cart=cc, polar=cpol, polar_back=cp_back, comb=comb, pred=cls,
                    prob=p[0].detach().float().cpu().numpy(), empty_cart=cc_raw.max() <= 0, empty_polar=cp_raw.max() <= 0)

    def cam_figure(rows_df, title, out):
        n = len(rows_df); fig, ax = plt.subplots(5, n, figsize=(2.5 * n, 13))
        ax = np.array(ax).reshape(5, n)
        labels = ["Input", "Cartesian branch", "Polar branch (polar space)", "Polar branch → mapped back", "Combined"]
        for k, (_, r) in enumerate(rows_df.iterrows()):
            e = explain(model, r.path); ok = CLASSES[e["pred"]] == r.label_name
            ax[0, k].imshow(e["img"]); ax[0, k].set_title(f"True: {r.label_name}\nPred: {CLASSES[e['pred']]} ({e['prob'].max()*100:.1f}%)",
                                                           fontsize=7.5, color="green" if ok else "red")
            ax[1, k].imshow(e["img"]); ax[1, k].imshow(e["cart"], cmap="jet", alpha=.45, vmin=0, vmax=1)
            ax[2, k].imshow(e["pol"]); ax[2, k].imshow(e["polar"], cmap="jet", alpha=.45, vmin=0, vmax=1)
            ax[3, k].imshow(e["img"]); ax[3, k].imshow(e["polar_back"], cmap="jet", alpha=.45, vmin=0, vmax=1)
            ax[4, k].imshow(e["img"]); ax[4, k].imshow(e["comb"], cmap="jet", alpha=.45, vmin=0, vmax=1)
            if e["empty_cart"]: ax[1, k].text(112, 112, "no positive\nevidence", ha="center", color="w", fontsize=7)
            if e["empty_polar"]:
                ax[2, k].text(112, 112, "no positive\nevidence", ha="center", color="w", fontsize=7)
                ax[3, k].text(112, 112, "no positive\nevidence", ha="center", color="w", fontsize=7)
            for i in range(5): ax[i, k].set_xticks([]); ax[i, k].set_yticks([])
        for i, l in enumerate(labels): ax[i, 0].set_ylabel(l, fontsize=8)
        plt.suptitle(title, fontweight="bold"); savefig(out)

    pred_test = PROB.argmax(1)
    tdf = test_df.assign(pred=pred_test, conf=PROB.max(1))
    correct = tdf[tdf.pred == tdf.label]
    cam_rows = pd.concat([correct[correct.label_name == c].sample(min(CFG["n_cam_per_class"], (correct.label_name == c).sum()), random_state=0)
                          for c in CLASSES])
    cam_figure(cam_rows, "PolarFuse-Net v2 — HiResCAM of both branches (correctly classified test images)", f"{FD}/fig3_explainability_correct.jpg")
    wrong = tdf[tdf.pred != tdf.label]
    if len(wrong):
        cam_figure(wrong.head(12), "PolarFuse-Net v2 — HiResCAM on misclassified test images", f"{FD}/fig4b_explainability_errors.jpg")

    # how often is each branch "empty"? (diagnostic for the paper / for us)
    diag = []
    for _, r in tdf.iterrows():
        e = explain(model, r.path); diag.append({"empty_cart": e["empty_cart"], "empty_polar": e["empty_polar"]})
    diag = pd.DataFrame(diag); print("Fraction of test images with an empty map:", diag.mean().round(3).to_dict())
    diag.mean().to_frame("fraction_empty").to_csv(f"{FD}/cam_empty_fraction.csv")
except Exception as e:
    import traceback; traceback.print_exc(); FAILED.append('Fig 3 — explainability (fixed): ' + repr(e))

# %% ---------------- Fig 4 — misclassified images with probabilities ----------------
try:
    if len(wrong):
        n = len(wrong); ncol = min(6, n); nrow = math.ceil(n / ncol)
        fig, ax = plt.subplots(2 * nrow, ncol, figsize=(2.7 * ncol, 5.2 * nrow), gridspec_kw={"height_ratios": [3, 1.3] * nrow})
        ax = np.array(ax).reshape(2 * nrow, ncol)
        for a in ax.ravel(): a.axis("off")
        for k, (i, r) in enumerate(wrong.iterrows()):
            rr, cc = 2 * (k // ncol), k % ncol; pr = PROB[test_df.index.get_loc(i)]
            ax[rr, cc].imshow(Image.open(r.path).convert("RGB").resize((S, S))); ax[rr, cc].axis("off")
            ax[rr, cc].set_title(f"True: {r.label_name}\nPred: {CLASSES[r.pred]}", fontsize=8, color="red")
            b = ax[rr + 1, cc]; b.axis("on"); b.barh(["N", "Pap", "Pseudo"], pr * 100, color=["#4c72b0", "#dd8452", "#55a868"])
            b.set_xlim(0, 100); b.tick_params(labelsize=7); b.invert_yaxis()
        plt.suptitle(f"Misclassified test images ({n} of {len(test_df)}) with predicted probabilities (%)", fontweight="bold")
        savefig(f"{FD}/fig4_misclassified.jpg")
    wrong.assign(**{f"prob_{c}": PROB[[test_df.index.get_loc(i) for i in wrong.index], k] for k, c in enumerate(CLASSES)}
                 ).to_csv(f"{FD}/misclassified.csv", index=False)
except Exception as e:
    import traceback; traceback.print_exc(); FAILED.append('Fig 4 — misclassified images with probabilities: ' + repr(e))

# %% ---------------- Fig 5 — hierarchical head analysis ----------------
try:
    @torch.no_grad()
    def head_outputs(model, dl):
        model.eval(); B, Sb, Y_ = [], [], []
        for xc, xp, y in dl:
            o = model(xc.to(DEVICE), xp.to(DEVICE))
            B.append(torch.softmax(o["bin"].float(), 1)[:, 1].cpu().numpy()); Sb.append(torch.softmax(o["sub"].float(), 1)[:, 1].cpu().numpy()); Y_.append(y.numpy())
        return np.concatenate(B), np.concatenate(Sb), np.concatenate(Y_)
    p_elev, p_pseudo, yh = head_outputs(model, test_dl)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for k, c in enumerate(CLASSES): ax[0].hist(p_elev[yh == k], bins=25, alpha=.6, label=c, range=(0, 1))
    ax[0].set_title("Stage 1: P(elevated disc)"); ax[0].set_xlabel("probability"); ax[0].legend()
    ab = yh > 0
    for k in [1, 2]: ax[1].hist(p_pseudo[yh == k], bins=25, alpha=.6, label=CLASSES[k], range=(0, 1), color=["", "#dd8452", "#55a868"][k])
    ax[1].set_title("Stage 2 (elevated discs only): P(pseudopapilledema)"); ax[1].set_xlabel("probability"); ax[1].legend()
    auc2 = roc_auc_score((yh[ab] == 2).astype(int), p_pseudo[ab]); f2, t2, _ = roc_curve((yh[ab] == 2).astype(int), p_pseudo[ab])
    auc1 = roc_auc_score((yh > 0).astype(int), p_elev); f1_, t1_, _ = roc_curve((yh > 0).astype(int), p_elev)
    ax[2].plot(f1_, t1_, label=f"Stage 1 Normal vs Elevated (AUC {auc1:.4f})"); ax[2].plot(f2, t2, label=f"Stage 2 Pap vs Pseudo (AUC {auc2:.4f})")
    ax[2].plot([0, 1], [0, 1], "k--"); ax[2].legend(fontsize=8); ax[2].set_title("Hierarchical heads — ROC"); ax[2].set_xlabel("FPR"); ax[2].set_ylabel("TPR")
    savefig(f"{FD}/fig5_hierarchical_heads.png")
except Exception as e:
    import traceback; traceback.print_exc(); FAILED.append('Fig 5 — hierarchical head analysis: ' + repr(e))

# %% ---------------- Fig 6 — standard result figures ----------------
try:
    plot_cm(Y, PROB, PROPOSED, f"{FD}/fig6a_confusion_matrix.png")
    plot_roc_pr(Y, PROB, PROPOSED, f"{FD}/fig6b_roc_pr.png")
    pc = per_class_metrics(Y, PROB); plot_per_class(pc, PROPOSED, f"{FD}/fig6c_per_class.png")
    if len(hist): plot_hist(hist, PROPOSED, f"{FD}/fig6d_training_curves.png")
    plot_tsne(model, PROPOSED, f"{FD}/fig6e_tsne.png")
except Exception as e:
    import traceback; traceback.print_exc(); FAILED.append('Fig 6 — standard result figures: ' + repr(e))

# %% ---------------- Save, overview, zip ----------------
ci = bootstrap_ci(Y, PROB, CFG["n_bootstrap"])
mt = pd.Series(M).to_frame("value")
for k, (lo, hi) in ci.items(): mt.loc[f"{k} 95% CI"] = f"{lo:.4f} – {hi:.4f}"
mt.loc["Stage-1 AUC (Normal vs Elevated)"] = auc1; mt.loc["Stage-2 AUC (Pap vs Pseudo)"] = auc2
mt.to_csv(f"{FD}/metrics_test.csv"); pc.to_csv(f"{FD}/per_class_test.csv", index=False); hist.to_csv(f"{FD}/history.csv", index=False)
pd.DataFrame({"path": test_df.path.values, "true": test_df.label_name.values, "pred": [CLASSES[i] for i in PROB.argmax(1)],
              **{f"prob_{c}": PROB[:, k] for k, c in enumerate(CLASSES)}}).to_csv(f"{FD}/predictions_test.csv", index=False)
figs = sorted(glob.glob(f"{FD}/fig*.png") + glob.glob(f"{FD}/fig*.jpg"))
montage(figs, f"{FD}/_OVERVIEW_paper_figures.jpg", "PolarFuse-Net v2 - paper figures", 2, 1600)
with open(f"{FD}/README.txt", "w") as fh:
    fh.write(f"CODE VERSION: {CODE_VERSION}\n"); fh.write("PolarFuse-Net v2 — paper figures (fixed split, seed 42)\n" + "=" * 60 + "\n\n" + mt.to_string() + "\n\n")
    fh.write("Empty-map fraction (HiResCAM):\n" + diag.mean().to_string() + "\n\nFigures:\n" + "\n".join(os.path.basename(f) for f in figs) + "\n")
if FAILED:
    open(f"{FD}/FAILED_STEPS.txt", "w").write("\n".join(FAILED)); print("Some figures failed:", FAILED)
zp = shutil.make_archive(f"{WORK}/results_paper_figures", "zip", RES)
torch.save(model.state_dict(), f"{WORK}/polarfuse_v2_best.pth")   # weights saved last, separately from the zip
print(f"\n✅ saved {zp} ({os.path.getsize(zp)/1e6:.1f} MB)")
print("➡️  Download ONLY results_paper_figures.zip from the Output tab (the .pth file is the model weights, not the results).")
