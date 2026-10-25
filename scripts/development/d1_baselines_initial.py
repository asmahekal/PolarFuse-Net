# %% [markdown]
# # Identification of Pseudopapilledema — Normal vs Papilledema vs Pseudopapilledema
# Dataset: https://www.kaggle.com/datasets/shashwatwork/identification-of-pseudopapilledema
# (original source: Kim U., OSF 2018 — 1,369 fundus images: 779 Normal, 295 Papilledema, 295 Pseudopapilledema)
#
# What this notebook does:
# 1. Stratified train / val / test split (70 / 15 / 15)
# 2. Fine-tunes ImageNet-pretrained CNNs (EfficientNet-B0, ResNet50, DenseNet121, ConvNeXt-Tiny) + soft-voting ensemble
# 3. Computes ALL performance metrics (overall + per class) on train, val AND test for every model,
#    with 95% bootstrap confidence intervals on the test set
# 4. Saves every figure per model, one summary image per model (all its figures in one picture),
#    and combined images that put all models side by side
# 5. Packs everything into **results.zip** (in /kaggle/working → Output tab → download)
#
# On Kaggle: Add Data → this dataset, and turn on **GPU (T4 / P100)** in the notebook settings.

# %%
import os, glob, random, time, copy, shutil, json, math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score, cohen_kappa_score,
                             matthews_corrcoef, log_loss)
from sklearn.preprocessing import label_binarize

# %% ---------------- Config ----------------
WORK = "/kaggle/working" if os.path.exists("/kaggle/working") else "./outputs"
CFG = dict(
    data_root    = "/kaggle/input",   # the code searches here for the 3 class folders
    img_size     = 224,
    batch_size   = 32,
    epochs       = 25,
    lr           = 3e-4,
    weight_decay = 1e-4,
    patience     = 6,                 # early stopping on val macro-F1
    seed         = 42,
    num_workers  = 2,
    n_bootstrap  = 1000,              # for 95% confidence intervals on the test set
    models       = ["efficientnet_b0", "resnet50", "densenet121", "convnext_tiny"],
    results_dir  = f"{WORK}/results", # everything that goes into results.zip
    weights_dir  = f"{WORK}/weights", # model weights (kept OUT of the zip — too big)
    zip_weights  = False,             # set True if you also want the .pth files inside the zip
)
RES = CFG["results_dir"]
if os.path.exists(RES): shutil.rmtree(RES)
for d in [RES, f"{RES}/models", f"{RES}/combined", f"{RES}/data", CFG["weights_dir"]]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(CFG["seed"])

def savefig(path):
    plt.tight_layout()
    kw = {"pil_kwargs": {"quality": 88}} if path.endswith(".jpg") else {}
    plt.savefig(path, dpi=150, bbox_inches="tight", **kw); plt.show(); plt.close("all")

# %% ---------------- Find the images ----------------
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def class_of(path):
    parts = [p.lower() for p in path.split(os.sep)]
    for p in reversed(parts[:-1]):          # nearest parent folder wins
        if p.startswith("pseudo"):  return "Pseudopapilledema"
        if p.startswith("papill"):  return "Papilledema"
        if p.startswith("normal") or p.startswith("control"): return "Normal"
    return None

files = [f for f in glob.glob(os.path.join(CFG["data_root"], "**", "*"), recursive=True)
         if f.lower().endswith(IMG_EXT)]
df = pd.DataFrame({"path": files})
df["label_name"] = df["path"].map(class_of)
df = df.dropna().drop_duplicates("path").reset_index(drop=True)

CLASSES = ["Normal", "Papilledema", "Pseudopapilledema"]
cls2idx = {c: i for i, c in enumerate(CLASSES)}
df["label"] = df["label_name"].map(cls2idx)
print("Total images:", len(df))
print(df["label_name"].value_counts())

# %% Class distribution + sample images
fig, ax = plt.subplots(figsize=(5, 3.5))
sns.countplot(x="label_name", data=df, order=CLASSES, ax=ax)
for c in ax.containers: ax.bar_label(c)
ax.set_title("Class distribution"); ax.set_xlabel("")
savefig(f"{RES}/data/class_distribution.png")

fig, axes = plt.subplots(3, 5, figsize=(13, 8))
for r, c in enumerate(CLASSES):
    sample = df[df.label_name == c].sample(5, random_state=CFG["seed"])
    for k, p in enumerate(sample.path):
        axes[r, k].imshow(Image.open(p).convert("RGB")); axes[r, k].axis("off")
    axes[r, 0].set_title(c, loc="left")
savefig(f"{RES}/data/sample_images.jpg")

# %% ---------------- Split 70 / 15 / 15 (stratified) ----------------
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df.label, random_state=CFG["seed"])
val_df, test_df   = train_test_split(temp_df, test_size=0.50, stratify=temp_df.label, random_state=CFG["seed"])
split_tbl = pd.DataFrame({n: d.label_name.value_counts().reindex(CLASSES) for n, d in
                          [("train", train_df), ("val", val_df), ("test", test_df)]})
split_tbl.loc["Total"] = split_tbl.sum()
print(split_tbl)
split_tbl.to_csv(f"{RES}/data/split_counts.csv")
for n, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
    d[["path", "label_name"]].to_csv(f"{RES}/data/{n}_files.csv", index=False)

# %% ---------------- Dataset & augmentation ----------------
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
train_tf = T.Compose([
    T.Resize((CFG["img_size"], CFG["img_size"])),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20),
    T.RandomAffine(0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.ToTensor(), T.Normalize(MEAN, STD),
])
eval_tf = T.Compose([T.Resize((CFG["img_size"], CFG["img_size"])), T.ToTensor(), T.Normalize(MEAN, STD)])

class FundusDS(Dataset):
    def __init__(self, d, tf): self.p, self.y, self.tf = d.path.values, d.label.values, tf
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB")), int(self.y[i])

def loader(d, tf, shuffle):
    return DataLoader(FundusDS(d, tf), batch_size=CFG["batch_size"], shuffle=shuffle,
                      num_workers=CFG["num_workers"], pin_memory=True)
train_dl = loader(train_df, train_tf, True)
EVAL_DL = {"train": loader(train_df, eval_tf, False),   # train set without augmentation, for metrics
           "val":   loader(val_df,   eval_tf, False),
           "test":  loader(test_df,  eval_tf, False)}
SPLIT_DF = {"train": train_df, "val": val_df, "test": test_df}

counts = np.bincount(train_df.label, minlength=len(CLASSES))
class_w = torch.tensor(len(train_df) / (len(CLASSES) * counts), dtype=torch.float32).to(DEVICE)
print("Class weights:", class_w.cpu().numpy().round(3))

# %% ---------------- ALL performance metrics ----------------
def overall_metrics(y, prob):
    """Every overall metric for one set of predictions."""
    pred = prob.argmax(1)
    yb = label_binarize(y, classes=range(len(CLASSES)))
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp; tn = cm.sum() - tp - fp - fn
    spec = tn / np.maximum(tn + fp, 1); npv = tn / np.maximum(tn + fn, 1)
    w = np.bincount(y, minlength=len(CLASSES)) / len(y)
    m = {}
    m["Accuracy"]             = accuracy_score(y, pred)
    m["Balanced Accuracy"]    = balanced_accuracy_score(y, pred)
    m["Error Rate"]           = 1 - m["Accuracy"]
    for avg in ["macro", "weighted", "micro"]:
        p, r, f, _ = precision_recall_fscore_support(y, pred, average=avg, zero_division=0)
        m[f"Precision ({avg})"] = p; m[f"Recall/Sensitivity ({avg})"] = r; m[f"F1 ({avg})"] = f
    m["Specificity (macro)"]  = spec.mean()
    m["Specificity (weighted)"] = (spec * w).sum()
    m["NPV (macro)"]          = npv.mean()
    m["MCC"]                  = matthews_corrcoef(y, pred)
    m["Cohen Kappa"]          = cohen_kappa_score(y, pred)
    m["AUC-ROC (macro OvR)"]  = roc_auc_score(y, prob, multi_class="ovr", average="macro")
    m["AUC-ROC (weighted OvR)"] = roc_auc_score(y, prob, multi_class="ovr", average="weighted")
    m["AUC-ROC (macro OvO)"]  = roc_auc_score(y, prob, multi_class="ovo", average="macro")
    m["AUC-PR (macro)"]       = average_precision_score(yb, prob, average="macro")
    m["Log Loss"]             = log_loss(y, np.clip(prob, 1e-7, 1), labels=range(len(CLASSES)))
    return m

def per_class_metrics(y, prob):
    pred = prob.argmax(1)
    yb = label_binarize(y, classes=range(len(CLASSES)))
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    rows = []
    for k, c in enumerate(CLASSES):
        TP = cm[k, k]; FP = cm[:, k].sum() - TP; FN = cm[k].sum() - TP; TN = cm.sum() - TP - FP - FN
        prec = TP / max(TP + FP, 1); rec = TP / max(TP + FN, 1)
        rows.append({"Class": c, "Support": int(cm[k].sum()), "TP": TP, "FP": FP, "FN": FN, "TN": TN,
                     "Accuracy (one-vs-rest)": (TP + TN) / cm.sum(),
                     "Precision (PPV)": prec, "Recall/Sensitivity (TPR)": rec,
                     "Specificity (TNR)": TN / max(TN + FP, 1), "NPV": TN / max(TN + FN, 1),
                     "F1": 2 * prec * rec / max(prec + rec, 1e-12),
                     "FPR": FP / max(FP + TN, 1), "FNR": FN / max(FN + TP, 1),
                     "AUC-ROC": roc_auc_score(yb[:, k], prob[:, k]),
                     "AUC-PR": average_precision_score(yb[:, k], prob[:, k])})
    return pd.DataFrame(rows)

def bootstrap_ci(y, prob, n=1000, seed=0):
    """95% CI for the main metrics (test set)."""
    rng = np.random.default_rng(seed); keys = ["Accuracy", "F1 (macro)", "AUC-ROC (macro OvR)", "Cohen Kappa", "MCC"]
    vals = {k: [] for k in keys}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < len(CLASSES): continue
        m = overall_metrics(y[idx], prob[idx])
        for k in keys: vals[k].append(m[k])
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in vals.items()}

# %% ---------------- Train / predict ----------------
@torch.no_grad()
def predict(model, dl):
    model.eval(); P, Y = [], []
    for x, y in dl:
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            out = model(x.to(DEVICE))
        P.append(torch.softmax(out.float(), 1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Y), np.concatenate(P)

def train_model(name):
    seed_everything(CFG["seed"])
    model = timm.create_model(name, pretrained=True, num_classes=len(CLASSES)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
    crit = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    hist, best_f1, best_state, best_ep, wait = [], -1, None, 0, 0
    t_start = time.time()

    for ep in range(1, CFG["epochs"] + 1):
        model.train(); t0, run_loss, correct, n = time.time(), 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                out = model(x); loss = crit(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run_loss += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); n += len(y)
        sched.step()
        yv, pv = predict(model, EVAL_DL["val"]); vm = overall_metrics(yv, pv)
        hist.append(dict(epoch=ep, train_loss=run_loss / n, train_acc=correct / n, val_loss=vm["Log Loss"],
                         val_acc=vm["Accuracy"], val_f1=vm["F1 (macro)"], lr=opt.param_groups[0]["lr"],
                         epoch_time_s=time.time() - t0))
        print(f"[{name}] ep {ep:02d} | train loss {run_loss/n:.4f} acc {correct/n:.4f} | "
              f"val loss {vm['Log Loss']:.4f} acc {vm['Accuracy']:.4f} F1 {vm['F1 (macro)']:.4f} | {time.time()-t0:.0f}s")
        if vm["F1 (macro)"] > best_f1:
            best_f1, best_state, best_ep, wait = vm["F1 (macro)"], copy.deepcopy(model.state_dict()), ep, 0
        else:
            wait += 1
            if wait >= CFG["patience"]: print("Early stopping"); break

    model.load_state_dict(best_state)
    torch.save(best_state, f"{CFG['weights_dir']}/{name}_best.pth")
    info = {"Params (M)": sum(p.numel() for p in model.parameters()) / 1e6,
            "Train time (min)": (time.time() - t_start) / 60, "Best epoch": best_ep, "Epochs run": len(hist)}
    return model, pd.DataFrame(hist), info

def inference_time_ms(model, n=50):
    model.eval(); x = torch.randn(1, 3, CFG["img_size"], CFG["img_size"], device=DEVICE)
    with torch.no_grad():
        for _ in range(5): model(x)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t = time.time()
        for _ in range(n): model(x)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
    return (time.time() - t) / n * 1000

# %% ---------------- Figures for one model ----------------
def plot_history(h, name, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(h.epoch, h.train_loss, "o-", label="train"); ax[0].plot(h.epoch, h.val_loss, "o-", label="val")
    ax[0].set_title(f"{name} — Loss")
    ax[1].plot(h.epoch, h.train_acc, "o-", label="train"); ax[1].plot(h.epoch, h.val_acc, "o-", label="val")
    ax[1].set_title(f"{name} — Accuracy")
    for a in ax: a.set_xlabel("Epoch"); a.legend(); a.grid(alpha=.3)
    savefig(out)

def plot_cm(y, prob, name, split, out):
    cm = confusion_matrix(y, prob.argmax(1), labels=range(len(CLASSES)))
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax[0], cbar=False)
    sns.heatmap(cmn, annot=True, fmt=".2%", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax[1], cbar=False, vmin=0, vmax=1)
    ax[0].set_title(f"{name} — Confusion matrix ({split})"); ax[1].set_title(f"{name} — Normalized ({split})")
    for a in ax: a.set_xlabel("Predicted"); a.set_ylabel("True")
    savefig(out)

def plot_roc_pr(y, prob, name, out):
    yb = label_binarize(y, classes=range(len(CLASSES)))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(yb[:, k], prob[:, k])
        ax[0].plot(fpr, tpr, label=f"{c} (AUC={roc_auc_score(yb[:, k], prob[:, k]):.3f})")
        pr, rc, _ = precision_recall_curve(yb[:, k], prob[:, k])
        ax[1].plot(rc, pr, label=f"{c} (AP={average_precision_score(yb[:, k], prob[:, k]):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set_xlabel("False Positive Rate"); ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title(f"{name} — ROC (test)")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title(f"{name} — Precision-Recall (test)")
    for a in ax: a.legend(loc="lower right" if a is ax[0] else "lower left"); a.grid(alpha=.3)
    savefig(out)

def plot_per_class_bars(pc, name, out):
    cols = ["Precision (PPV)", "Recall/Sensitivity (TPR)", "Specificity (TNR)", "NPV", "F1", "AUC-ROC"]
    ax = (pc.set_index("Class")[cols] * 100).plot.bar(figsize=(11, 4), rot=0, width=.8)
    ax.set_ylim(0, 105); ax.set_ylabel("%"); ax.set_title(f"{name} — Per-class metrics (test)")
    ax.legend(ncol=3, fontsize=8, loc="lower right"); ax.grid(axis="y", alpha=.3)
    savefig(out)

def plot_metric_table(m, pc, ci, info, name, out):
    """A picture of the metrics table, so the summary image also shows the numbers."""
    keys = ["Accuracy", "Balanced Accuracy", "Precision (macro)", "Recall/Sensitivity (macro)", "F1 (macro)",
            "F1 (weighted)", "Specificity (macro)", "NPV (macro)", "MCC", "Cohen Kappa",
            "AUC-ROC (macro OvR)", "AUC-PR (macro)", "Log Loss"]
    rows = []
    for k in keys:
        raw = k in ("Log Loss", "Cohen Kappa", "MCC")
        v = f"{m[k]:.4f}" if raw else f"{m[k]*100:.2f}%"
        if ci and k in ci:
            v += f"  [{ci[k][0]:.3f}–{ci[k][1]:.3f}]" if raw else f"  [{ci[k][0]*100:.1f}–{ci[k][1]*100:.1f}]"
        rows.append([k, v])
    for k, v in (info or {}).items(): rows.append([k, f"{v:.2f}" if isinstance(v, float) else str(v)])
    fig, ax = plt.subplots(figsize=(6.5, 0.36 * len(rows) + 0.8)); ax.axis("off")
    t = ax.table(cellText=rows, colLabels=["Metric (test)", "Value  [95% CI]"], loc="center", cellLoc="left", colWidths=[.55, .45])
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.3)
    ax.set_title(f"{name} — Test metrics", fontweight="bold")
    savefig(out)

class GradCAM:
    def __init__(self, model):
        self.model = model
        layer = [m for m in model.modules() if isinstance(m, nn.Conv2d)][-1]
        layer.register_forward_hook(self._fwd)
    def _fwd(self, m, i, o):
        self.a = o.detach()
        if o.requires_grad: o.register_hook(lambda g: setattr(self, "g", g.detach()))
    def __call__(self, x):
        with torch.enable_grad():
            out = self.model(x); cls = out.argmax(1).item()
            self.model.zero_grad(); out[0, cls].backward()
        w = self.g.mean((2, 3), keepdim=True)
        cam = torch.relu((w * self.a).sum(1)).squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8), cls

GRADCAM_SAMPLES = test_df.groupby("label_name").sample(2, random_state=CFG["seed"])  # same images for every model

def plot_gradcam(model, name, out):
    model.eval(); cam_fn = GradCAM(model)
    n = len(GRADCAM_SAMPLES)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6))
    for k, (_, row) in enumerate(GRADCAM_SAMPLES.iterrows()):
        img = Image.open(row.path).convert("RGB").resize((CFG["img_size"], CFG["img_size"]))
        x = eval_tf(img).unsqueeze(0).to(DEVICE).requires_grad_(True)
        cam, pred = cam_fn(x)
        cam = np.array(Image.fromarray(np.uint8(cam * 255)).resize(img.size)) / 255.0
        ok = CLASSES[pred] == row.label_name
        axes[0, k].imshow(img); axes[0, k].set_title(f"True: {row.label_name}", fontsize=9)
        axes[1, k].imshow(img); axes[1, k].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, k].set_title(f"Pred: {CLASSES[pred]}", fontsize=9, color="green" if ok else "red")
        axes[0, k].axis("off"); axes[1, k].axis("off")
    plt.suptitle(f"{name} — Grad-CAM (test images)", fontweight="bold")
    savefig(out)

# %% ---------------- Montage: put many PNGs into ONE picture ----------------
def _font(size):
    try: return ImageFont.load_default(size=size)
    except TypeError: return ImageFont.load_default()

def montage(paths, out, title, ncols=2, width=1400, pad=30):
    ims = [Image.open(p).convert("RGB") for p in paths if os.path.exists(p)]
    ims = [im.resize((width, int(im.height * width / im.width))) for im in ims]
    rows = [ims[i:i + ncols] for i in range(0, len(ims), ncols)]
    heights = [max(im.height for im in r) for r in rows]
    head = 110
    W = ncols * width + (ncols + 1) * pad; H = head + sum(heights) + (len(rows) + 1) * pad
    canvas = Image.new("RGB", (W, H), "white"); draw = ImageDraw.Draw(canvas)
    draw.text((pad, 30), title, fill="black", font=_font(56))
    y = head + pad
    for r, h in zip(rows, heights):
        x = pad
        for im in r: canvas.paste(im, (x, y)); x += width + pad
        y += h + pad
    canvas.save(out, quality=88, optimize=True)
    return out

# %% ---------------- Run all models ----------------
ALL_OVERALL, ALL_PERCLASS, ALL_HIST, INFO, CI, PROBS, REPORTS = [], [], [], {}, {}, {}, []
Y = {}

def evaluate_and_save(name, probs_by_split, hist=None, info=None, model=None):
    d = f"{RES}/models/{name}"; os.makedirs(d, exist_ok=True)
    for split, prob in probs_by_split.items():
        y = Y[split]
        m = overall_metrics(y, prob); m.update(Model=name, Split=split); ALL_OVERALL.append(m)
        pc = per_class_metrics(y, prob); pc.insert(0, "Split", split); pc.insert(0, "Model", name); ALL_PERCLASS.append(pc)
        rep = classification_report(y, prob.argmax(1), target_names=CLASSES, digits=4)
        REPORTS.append(f"===== {name} | {split} =====\n{rep}\n")
        pd.DataFrame({"path": SPLIT_DF[split].path.values, "true": [CLASSES[i] for i in y],
                      "pred": [CLASSES[i] for i in prob.argmax(1)],
                      **{f"prob_{c}": prob[:, k] for k, c in enumerate(CLASSES)}}).to_csv(f"{d}/predictions_{split}.csv", index=False)
        plot_cm(y, prob, name, split, f"{d}/confusion_matrix_{split}.png")
    yt, pt = Y["test"], probs_by_split["test"]
    ci = bootstrap_ci(yt, pt, CFG["n_bootstrap"], CFG["seed"]); CI[name] = ci
    pc_test = per_class_metrics(yt, pt)
    print(classification_report(yt, pt.argmax(1), target_names=CLASSES, digits=4))
    if hist is not None: plot_history(hist, name, f"{d}/training_history.png")
    plot_roc_pr(yt, pt, name, f"{d}/roc_pr_curves.png")
    plot_per_class_bars(pc_test, name, f"{d}/per_class_metrics.png")
    plot_metric_table(overall_metrics(yt, pt), pc_test, ci, info, name, f"{d}/metrics_table.png")
    if model is not None: plot_gradcam(model, name, f"{d}/gradcam.jpg")
    # ---- ONE picture with every figure of this model ----
    order = ["metrics_table.png", "training_history.png", "confusion_matrix_test.png", "roc_pr_curves.png",
             "per_class_metrics.png", "gradcam.jpg", "confusion_matrix_train.png", "confusion_matrix_val.png"]
    montage([f"{d}/{o}" for o in order], f"{RES}/_SUMMARY_{name}.jpg", f"{name} - all results")

for split in ["train", "val", "test"]:
    Y[split] = SPLIT_DF[split].label.values

for name in CFG["models"]:
    print("\n" + "=" * 70 + f"\n{name}\n" + "=" * 70)
    model, hist, info = train_model(name)
    info["Inference (ms/img)"] = inference_time_ms(model)
    INFO[name] = info
    hist.insert(0, "Model", name); ALL_HIST.append(hist)
    probs = {s: predict(model, EVAL_DL[s])[1] for s in ["train", "val", "test"]}
    PROBS[name] = probs
    evaluate_and_save(name, probs, hist, info, model)
    del model
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

# %% ---------------- Soft-voting ensemble ----------------
ens = "Ensemble_soft_voting"
PROBS[ens] = {s: np.mean([PROBS[m][s] for m in CFG["models"]], axis=0) for s in ["train", "val", "test"]}
evaluate_and_save(ens, PROBS[ens])
MODELS_ALL = CFG["models"] + [ens]

# %% ---------------- Metrics tables (CSV + Excel) ----------------
overall = pd.DataFrame(ALL_OVERALL)
overall = overall[["Model", "Split"] + [c for c in overall.columns if c not in ("Model", "Split")]]
perclass = pd.concat(ALL_PERCLASS, ignore_index=True)
history = pd.concat(ALL_HIST, ignore_index=True)
info_df = pd.DataFrame(INFO).T.rename_axis("Model").reset_index()

test_tbl = overall[overall.Split == "test"].drop(columns="Split").set_index("Model")
RAW = ["Log Loss", "Cohen Kappa", "MCC"]   # these stay on their own scale (not %)
pct_cols = [c for c in test_tbl.columns if c not in RAW]
test_pct = test_tbl.copy(); test_pct[pct_cols] = (test_pct[pct_cols] * 100).round(2); test_pct[RAW] = test_pct[RAW].round(4)
ci_df = pd.DataFrame({m: {f"{k} 95% CI": (f"{v[0]:.4f} – {v[1]:.4f}" if k in RAW else f"{v[0]*100:.2f} – {v[1]*100:.2f}")
                          for k, v in CI[m].items()} for m in MODELS_ALL}).T
paper_tbl = test_pct[["Accuracy", "Precision (macro)", "Recall/Sensitivity (macro)", "Specificity (macro)",
                      "F1 (macro)", "AUC-ROC (macro OvR)", "Cohen Kappa", "MCC"]].join(ci_df)
paper_tbl = paper_tbl.join(info_df.set_index("Model"))

print("\n===== TEST SET — main table =====")
print(paper_tbl.to_string())

paper_tbl.to_csv(f"{RES}/TABLE_test_results_for_paper.csv")
overall.to_csv(f"{RES}/metrics_overall_all_splits.csv", index=False)
perclass.to_csv(f"{RES}/metrics_per_class_all_splits.csv", index=False)
history.to_csv(f"{RES}/training_history_all_models.csv", index=False)
with open(f"{RES}/classification_reports.txt", "w") as f: f.writelines(REPORTS)

with pd.ExcelWriter(f"{RES}/ALL_RESULTS.xlsx") as xw:
    paper_tbl.to_excel(xw, sheet_name="Test_main_table")
    test_pct.to_excel(xw, sheet_name="Test_all_metrics_%")
    perclass[perclass.Split == "test"].to_excel(xw, sheet_name="Test_per_class", index=False)
    overall.to_excel(xw, sheet_name="All_splits_overall", index=False)
    perclass.to_excel(xw, sheet_name="All_splits_per_class", index=False)
    history.to_excel(xw, sheet_name="Training_history", index=False)
    info_df.to_excel(xw, sheet_name="Model_info", index=False)
    split_tbl.to_excel(xw, sheet_name="Data_split")
    pd.DataFrame(list({k: str(v) for k, v in CFG.items()}.items()), columns=["param", "value"]).to_excel(xw, sheet_name="Config", index=False)

# %% ---------------- Combined figures: all models together ----------------
C = f"{RES}/combined"

# 1) comparison bars of the main metrics
main = ["Accuracy", "Precision (macro)", "Recall/Sensitivity (macro)", "Specificity (macro)", "F1 (macro)", "AUC-ROC (macro OvR)"]
ax = test_pct[main].plot.bar(figsize=(13, 5), rot=15, width=.85)
ax.set_ylim(max(0, test_pct[main].values.min() - 10), 100.5); ax.set_ylabel("%")
ax.set_title("All models — test-set metrics"); ax.legend(ncol=3, fontsize=8, loc="lower right"); ax.grid(axis="y", alpha=.3)
savefig(f"{C}/comparison_bar.png")

# 2) heatmap of every metric × every model
hm = test_tbl.drop(columns=["Log Loss", "Error Rate"]) * 100   # Kappa & MCC shown ×100 here
plt.figure(figsize=(max(10, 1.0 * hm.shape[1]), 0.6 * hm.shape[0] + 2))
sns.heatmap(hm, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "%"})
plt.title("All metrics × all models (test, %)"); plt.xticks(rotation=45, ha="right")
savefig(f"{C}/metrics_heatmap.png")

# 3) all confusion matrices side by side
n = len(MODELS_ALL); fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4))
for a, m in zip(axes, MODELS_ALL):
    cm = confusion_matrix(Y["test"], PROBS[m]["test"].argmax(1))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a, xticklabels=CLASSES, yticklabels=CLASSES)
    a.set_title(f"{m}\nAcc={accuracy_score(Y['test'], PROBS[m]['test'].argmax(1))*100:.2f}%", fontsize=10)
    a.set_xlabel("Predicted"); a.set_ylabel("True"); a.tick_params(labelsize=7)
savefig(f"{C}/all_confusion_matrices.png")

# 4) all ROC curves (macro) on one plot + per-model ROC panels
yb = label_binarize(Y["test"], classes=range(len(CLASSES)))
fig, axes = plt.subplots(1, len(CLASSES), figsize=(5.2 * len(CLASSES), 4.5))
for k, c in enumerate(CLASSES):
    for m in MODELS_ALL:
        fpr, tpr, _ = roc_curve(yb[:, k], PROBS[m]["test"][:, k])
        axes[k].plot(fpr, tpr, label=f"{m} ({roc_auc_score(yb[:, k], PROBS[m]['test'][:, k]):.3f})")
    axes[k].plot([0, 1], [0, 1], "k--"); axes[k].set_title(f"ROC — {c}"); axes[k].legend(fontsize=7, loc="lower right")
    axes[k].set_xlabel("FPR"); axes[k].set_ylabel("TPR"); axes[k].grid(alpha=.3)
savefig(f"{C}/all_roc_curves.png")

# 5) training curves of all models
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for m, h in history.groupby("Model"):
    ax[0].plot(h.epoch, h.val_loss, "o-", ms=3, label=m); ax[1].plot(h.epoch, h.val_acc, "o-", ms=3, label=m)
ax[0].set_title("Validation loss — all models"); ax[1].set_title("Validation accuracy — all models")
for a in ax: a.set_xlabel("Epoch"); a.legend(fontsize=8); a.grid(alpha=.3)
savefig(f"{C}/all_training_curves.png")

# 6) per-class F1 of all models
pcf = perclass[perclass.Split == "test"].pivot(index="Model", columns="Class", values="F1").reindex(MODELS_ALL) * 100
ax = pcf.plot.bar(figsize=(11, 4), rot=15); ax.set_ylabel("F1 %"); ax.set_ylim(max(0, pcf.values.min() - 10), 100.5)
ax.set_title("Per-class F1 — all models (test)"); ax.grid(axis="y", alpha=.3)
savefig(f"{C}/all_per_class_f1.png")

# 7) Grad-CAM of all models stacked
montage([f"{RES}/models/{m}/gradcam.jpg" for m in CFG["models"]], f"{C}/all_gradcam.jpg",
        "Grad-CAM - all models (same test images)", ncols=1, width=1800)

# 8) main results table as an image
fig, ax = plt.subplots(figsize=(15, 0.5 * len(paper_tbl) + 1.2)); ax.axis("off")
show = paper_tbl[["Accuracy", "Precision (macro)", "Recall/Sensitivity (macro)", "Specificity (macro)",
                  "F1 (macro)", "AUC-ROC (macro OvR)", "Cohen Kappa", "MCC"]]
t = ax.table(cellText=show.values, rowLabels=show.index, colLabels=[c.replace(" (macro", "\n(macro").replace("Recall/", "Recall/\n") for c in show.columns],
             loc="center", cellLoc="center"); t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 2)
ax.set_title("Test-set results (% — Kappa & MCC on 0–1 scale)", fontweight="bold"); savefig(f"{C}/results_table.png")

# ===== THE BIG PICTURE: every combined figure in one image =====
montage([f"{C}/results_table.png", f"{C}/comparison_bar.png", f"{C}/metrics_heatmap.png",
         f"{C}/all_per_class_f1.png", f"{C}/all_confusion_matrices.png", f"{C}/all_roc_curves.png",
         f"{C}/all_training_curves.png", f"{C}/all_gradcam.jpg"],
        f"{RES}/_ALL_MODELS_OVERVIEW.jpg", "Pseudopapilledema classification - all models", ncols=2, width=1600)


# %% ---------------- README + ZIP ----------------
best = test_tbl.drop(ens).sort_values(["F1 (macro)", "Accuracy"], ascending=False).index[0]
with open(f"{RES}/README.txt", "w") as f:
    f.write("Identification of Pseudopapilledema — results\n" + "=" * 60 + "\n\n")
    f.write(f"Images: {len(df)}  |  split train/val/test = {len(train_df)}/{len(val_df)}/{len(test_df)}\n")
    f.write(f"Device: {DEVICE}  |  image size: {CFG['img_size']}  |  epochs (max): {CFG['epochs']}\n")
    f.write(f"Best single model (test macro-F1): {best}\n\n")
    f.write("TEST SET — main table\n" + paper_tbl.to_string() + "\n\n")
    f.write("Files:\n"
            "  _ALL_MODELS_OVERVIEW.jpg      -> one image with all combined figures\n"
            "  _SUMMARY_<model>.jpg          -> one image with all figures of that model\n"
            "  ALL_RESULTS.xlsx              -> every metric, every model, every split (+ per class, history, config)\n"
            "  TABLE_test_results_for_paper.csv\n"
            "  metrics_overall_all_splits.csv / metrics_per_class_all_splits.csv\n"
            "  classification_reports.txt\n"
            "  models/<model>/               -> separate figures + predictions CSV per split\n"
            "  combined/                     -> figures comparing all models\n"
            "  data/                         -> class distribution, samples, split lists\n")

if CFG["zip_weights"]:
    shutil.copytree(CFG["weights_dir"], f"{RES}/weights", dirs_exist_ok=True)
zip_path = shutil.make_archive(f"{WORK}/results", "zip", RES)
print(f"\n✅ All results saved to: {zip_path}  ({os.path.getsize(zip_path)/1e6:.1f} MB)")
print("Download it from the Output tab (right panel → /kaggle/working/results.zip)")

# show the overview picture in the notebook
plt.figure(figsize=(20, 20)); plt.imshow(Image.open(f"{RES}/_ALL_MODELS_OVERVIEW.jpg")); plt.axis("off"); plt.show()
