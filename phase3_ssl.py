"""MorphoGround - PHASE 3: self-supervised embryo representations.
1) Domain-adaptive MAE pretraining on the SSL-allowed images from Phase 1 (never the gold test set),
   starting from ImageNet-MAE and from FEMI.
2) Frozen-representation evaluation of every encoder: linear probe, kNN, few-shot label efficiency,
   t-SNE + silhouette, and augmentation consistency.
Self-contained (no other files needed): paste into ONE Jupyter cell, or run  python phase3_ssl.py
v2: fixes the crash of the MAE evaluation under transformers 5.x ('ViTMAEModel' object has no attribute 'encoder'); MAE
tokens are read without masking or shuffling in any version; HF_TOKEN setting for the gated FEMI model; the adapted encoder
is retrained only when the SSL image set changes (fingerprint stored next to it)."""
from pathlib import Path

# ====================================== USER SETTINGS =======================================
PROJECT_DIR = Path.home() / "Embryo_Updated Project"   # same folder as Phase 1
PHASE1_DIR  = PROJECT_DIR / "phase1_outputs"           # manifest.csv + class_weights.json from Phase 1
CACHE_DIR   = PROJECT_DIR / "cache"                     # resized images, features, adapted models (reused)
IMG_SIZE    = 224
NUM_WORKERS = 4
QUICK_TEST  = False   # True = fast smoke test (1 fold, 2 epochs) to check that everything runs

# ---- PHASE 3 settings
HF_TOKEN    = None   # FEMI is gated: open huggingface.co/ihlab/FEMI, press "Agree and access", wait for approval,
                     # then paste a READ token from huggingface.co/settings/tokens here, e.g. "hf_xxx" (stored for later phases)
MAE_INITS = {"mae_in1k": "facebook/vit-mae-base", "femi": "ihlab/FEMI"}   # MAE pretraining continues from these
MAE_EPOCHS, MAE_BATCH, MAE_BASE_LR, MASK_RATIO = 100, 64, 1.5e-4, .75
COMPARE   = ["resnet50", "vit_b16", "dinov2_b14", "mae_in1k", "femi"]    # + the *_dapt encoders trained here
FEWSHOT, FEWSHOT_FOLDS = [.05, .1, .25, .5, 1.], [0, 1, 2]
OUT       = PROJECT_DIR / "phase3_outputs"
MAE_BACKEND = "auto"   # auto: Hugging Face first, timm fallback for ImageNet-MAE if it fails | "hf" | "timm"
RETRAIN_MAE = False    # False: reuse an adapted model already in the cache (a rerun only redoes what is missing)
TIMM_MAE    = "vit_base_patch16_224.mae"   # the same ImageNet-MAE encoder, served by timm (safetensors)
# ============================================================================================

import json, math, random, sys, time, warnings
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
TASKS = ("EXP", "ICM", "TE")
CLASS_NAMES = {"EXP": ["1", "2", "3", "4", "5", "6"], "ICM": ["A", "B", "C"], "TE": ["A", "B", "C"]}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16 if DEVICE == "cpu" or torch.cuda.is_bf16_supported() else torch.float16
IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
BACKBONES = {  # name: (library, model id or local folder)
    "resnet50":   ("timm", "resnet50.a1_in1k"),                           # ImageNet supervised CNN
    "vit_b16":    ("timm", "vit_base_patch16_224.augreg_in21k_ft_in1k"),  # ImageNet supervised ViT
    "dinov2_b14": ("timm", "vit_base_patch14_dinov2.lvd142m"),            # self-supervised (natural images)
    "mae_in1k":   ("hf",   "facebook/vit-mae-base"),                      # MAE (natural images)
    "femi":       ("hf",   "ihlab/FEMI"),  # MAE on 18M embryo images; gated: accept terms on HF + huggingface-cli login
}


def log(msg): print(f"{time.strftime('%H:%M:%S')} | {msg}", flush=True)


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def autocast(): return torch.autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=DEVICE == "cuda")


def make_scaler():
    on = DEVICE == "cuda" and AMP_DTYPE == torch.float16
    return torch.amp.GradScaler("cuda", enabled=on) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=on)


def cosine_sched(opt, steps, warm):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))


# ---------------------------------------------------------------- data from Phase 1
def load_manifest():
    """Readable images from Phase 1: folds 0-4 = CV pool, -1 = locked gold test, -2 excluded, -3 unlabelled."""
    m = pd.read_csv(PHASE1_DIR / "manifest.csv", low_memory=False)
    return m[m["readable"].astype(bool) & m["path"].notna()].reset_index(drop=True)


def targets(m):
    Y = m[["EXP_ord", "ICM_ord", "TE_ord"]].fillna(-1).astype(int).to_numpy()
    return Y, [int(Y[:, 0].max()) + 1, 3, 3]


def phase_status(folder, name):
    """First verdict line of an earlier phase (e.g. 'PASS WITH WARNINGS'), or None if it was not run."""
    f = PROJECT_DIR / folder / name
    if not f.exists(): return None
    line = next((l for l in f.read_text().splitlines() if "VERDICT:" in l), "")
    return line.split("VERDICT:")[1].split("(")[0].strip() if line else None


def phase1_class_weights():
    cw = json.loads((PHASE1_DIR / "class_weights.json").read_text())
    key = lambda d: next(k for k in d if k.startswith("effective_number"))  # noqa: E731
    return {int(f[4:]): [cw[f][t][key(cw[f][t])] for t in TASKS] for f in cw if f.startswith("fold")}


def image_cache(m):
    """uint8 (N, IMG_SIZE, IMG_SIZE, 3) array aligned with the manifest rows; built once, reused by every phase."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f, kf, keys = CACHE_DIR / f"images_{IMG_SIZE}.npy", CACHE_DIR / f"images_{IMG_SIZE}_keys.json", m["image_key"].astype(str).tolist()
    if f.exists() and kf.exists() and json.loads(kf.read_text()) == keys: return np.load(f, mmap_mode="r")
    log(f"Caching {len(m)} images at {IMG_SIZE}px -> {f}")
    arr = np.lib.format.open_memmap(f, mode="w+", dtype=np.uint8, shape=(len(m), IMG_SIZE, IMG_SIZE, 3))
    for i, p in enumerate(m["path"]):
        with Image.open(p) as im: arr[i] = np.asarray(im.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))
    arr.flush(); del arr; kf.write_text(json.dumps(keys))
    return np.load(f, mmap_mode="r")


class RandRot90:
    def __call__(self, x): return torch.rot90(x, random.randint(0, 3), (1, 2))


TRAIN_AUG = T.Compose([T.RandomResizedCrop(IMG_SIZE, scale=(.8, 1), ratio=(.9, 1.1), antialias=True),
                       T.RandomHorizontalFlip(), T.RandomVerticalFlip(), RandRot90(), T.ColorJitter(.2, .2)])


class EmbryoDS(torch.utils.data.Dataset):
    """Cached image -> augmented (optional) and normalised tensor, plus the 3 targets (-1 = ignore)."""
    def __init__(self, imgs, idx, Y, mean, std, aug=None):
        self.imgs, self.idx, self.Y, self.aug = imgs, np.asarray(idx), Y, aug
        self.mean, self.std = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.imgs[self.idx[i]])).permute(2, 0, 1).float() / 255
        x = self.aug(x) if self.aug is not None else x
        y = torch.as_tensor(self.Y[self.idx[i]]) if self.Y is not None else torch.tensor(-1)
        return (x - self.mean) / self.std, y


def loader(ds, bs, shuffle, workers=NUM_WORKERS):
    ctx = "fork" if workers > 0 and sys.platform.startswith("linux") else None   # works when pasted into a notebook cell
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, multiprocessing_context=ctx,
                                       pin_memory=DEVICE == "cuda", drop_last=shuffle and len(ds) > bs)


# ---------------------------------------------------------------- backbones and models
def hf_stats(model_id):
    try:
        from transformers import AutoImageProcessor
        p = AutoImageProcessor.from_pretrained(model_id); return tuple(p.image_mean), tuple(p.image_std)
    except Exception: return IMNET  # noqa: BLE001


def find_blocks(net):
    """Transformer blocks of a Hugging Face ViT/ViT-MAE, robust to attribute renames across transformers versions
    (4.x: net.encoder.layer; some 5.x models: net.layers / net.encoder.layers)."""
    for path in ("encoder.layer", "encoder.layers", "layers", "layer", "blocks", "vit.encoder.layer"):
        obj = net
        for a in path.split("."): obj = getattr(obj, a, None)
        if isinstance(obj, nn.ModuleList) and len(obj): return list(obj)
    lists = [mod for mod in net.modules() if isinstance(mod, nn.ModuleList) and len(mod)]
    if not lists: raise AttributeError(f"no transformer blocks found in {type(net).__name__}")
    return list(max(lists, key=len))


def no_mask(net):
    """Switch MAE random masking off everywhere it may be stored (config and sub-module copies of it)."""
    for mod in net.modules():
        cfg = getattr(mod, "config", None)
        if cfg is not None and hasattr(cfg, "mask_ratio"): cfg.mask_ratio = 0.0
        if isinstance(getattr(mod, "mask_ratio", None), float): mod.mask_ratio = 0.0
    return net


def hf_tokens(net, x):
    """HF ViT-MAE forward without masking and without token shuffling -> patch tokens in spatial order (B, N, D)."""
    n = (x.shape[-1] // net.config.patch_size) ** 2
    try: o = net(pixel_values=x, noise=torch.arange(n, device=x.device, dtype=torch.float32).expand(len(x), -1))  # identity order
    except TypeError: o = net(pixel_values=x)
    h = o.last_hidden_state[:, 1:]
    if h.shape[1] != n: raise RuntimeError(f"MAE still masks patches ({h.shape[1]}/{n} kept) - transformers version not supported")
    ids = getattr(o, "ids_restore", None)
    if ids is not None and not torch.equal(ids[0], torch.arange(n, device=ids.device)):
        h = torch.gather(h, 1, ids.unsqueeze(-1).expand(-1, -1, h.shape[-1]))                        # undo shuffling
    return h


def hf_login():
    """Use HF_TOKEN (setting or environment) for gated models such as FEMI. huggingface_hub stores it, so later
    phases get access too. Returns a short status line for the verdict."""
    import os
    tok = HF_TOKEN or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok: return "no HF_TOKEN set (public models only; FEMI needs one)"
    os.environ["HF_TOKEN"] = tok
    try:
        from huggingface_hub import login, whoami
        login(token=tok, add_to_git_credential=False); return f"logged in to Hugging Face as {whoami(token=tok).get('name', '?')}"
    except Exception as e: return f"token set, but login check failed ({type(e).__name__}: {str(e)[:60]})"  # noqa: BLE001


class Encoder(nn.Module):
    """Any backbone -> pooled feature (B, dim). timm: model's own pooling; HF ViT-MAE: mean of patch tokens, no masking."""
    def __init__(self, name):
        super().__init__()
        self.lib, mid = BACKBONES[name]
        if self.lib in ("timm", "timm_tokens"):
            import timm
            from timm.data import resolve_data_config
            arch, _, w = mid.partition("|")                     # "arch|local_weights.pth" for encoders adapted in Phase 3
            self.net = timm.create_model(arch, pretrained=not w, num_classes=0, **({"img_size": IMG_SIZE} if "dinov2" in arch else {}))
            if w: self.net.load_state_dict(torch.load(w, map_location="cpu"))
            cfg = resolve_data_config({}, model=self.net)
            self.mean, self.std, self.size, self.dim = tuple(cfg["mean"]), tuple(cfg["std"]), IMG_SIZE, self.net.num_features
            self.blocks = list(self.net.blocks) if hasattr(self.net, "blocks") else list(self.net.layer4)
        else:
            from transformers import ViTMAEModel, logging as hf_logging
            hf_logging.set_verbosity_error()
            self.net = no_mask(ViTMAEModel.from_pretrained(mid))                                  # keep every patch
            (self.mean, self.std), self.size = hf_stats(mid), self.net.config.image_size
            self.dim, self.blocks = self.net.config.hidden_size, find_blocks(self.net)          # v2: transformers 4.x and 5.x

    def set_mode(self, mode, n_blocks=2):
        """frozen: nothing trains | partial: last n_blocks + final norm | full: everything."""
        for p in self.parameters(): p.requires_grad = mode == "full"
        if mode == "partial":
            norms = [getattr(self.net, a) for a in ("norm", "fc_norm", "layernorm") if isinstance(getattr(self.net, a, None), nn.Module)]
            for mod in self.blocks[-n_blocks:] + norms:
                for p in mod.parameters(): p.requires_grad = True

    def forward(self, x):
        if x.shape[-1] != self.size: x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        if self.lib == "timm": return self.net(x)
        if self.lib == "timm_tokens": return self.net.forward_features(x)[:, self.net.num_prefix_tokens:].mean(1)  # MAE: mean of patches
        return hf_tokens(self.net, x).mean(1)


class Grader(nn.Module):
    """Encoder + one linear head per Gardner criterion (EXP, ICM, TE)."""
    def __init__(self, enc, n_cls):
        super().__init__(); self.enc = enc
        self.heads = nn.ModuleList(nn.Sequential(nn.Dropout(.2), nn.Linear(enc.dim, k)) for k in n_cls)

    def forward(self, x):
        f = self.enc(x); return [h(f) for h in self.heads]


@torch.no_grad()
def extract_features(name, m, imgs, force=False):
    """Frozen pooled features for every manifest row (cached in CACHE_DIR)."""
    f, kf, keys = CACHE_DIR / f"feat_{name}.npy", CACHE_DIR / f"feat_{name}_keys.json", m["image_key"].astype(str).tolist()
    if f.exists() and kf.exists() and not force and json.loads(kf.read_text()) == keys:   # v2: rows must match the manifest,
        return np.load(f)                                                                  # not only their number
    enc = Encoder(name).to(DEVICE).eval(); out = []
    for x, _ in loader(EmbryoDS(imgs, np.arange(len(m)), None, enc.mean, enc.std), 64, False):
        with autocast(): out.append(enc(x.to(DEVICE)).float().cpu().numpy())
    Fe = np.concatenate(out); np.save(f, Fe); kf.write_text(json.dumps(keys)); del enc; torch.cuda.empty_cache()
    return Fe


# ---------------------------------------------------------------- metrics
def metrics(y, prob):
    """y: true labels (-1 ignored); prob: (n, K). Accuracy, macro/weighted F1, QWK, MAE, per-class recall."""
    k = y >= 0; y, K = y[k], prob.shape[1]; p = prob[k].argmax(1)
    if len(y) == 0: return {"n": 0}
    out = {"n": int(len(y)), "acc": accuracy_score(y, p),
           "macro_f1": f1_score(y, p, average="macro", zero_division=0),
           "weighted_f1": f1_score(y, p, average="weighted", zero_division=0),
           "qwk": cohen_kappa_score(y, p, weights="quadratic") if len(set(y) | set(p)) > 1 else 0.0,
           "mae": float(np.abs(y - p).mean())}
    rec = recall_score(y, p, labels=list(range(K)), average=None, zero_division=0).astype(float)
    rec[np.bincount(y, minlength=K)[:K] == 0] = np.nan          # class absent from y -> undefined, not 0
    out.update({f"recall_{c}": r for c, r in enumerate(rec)})
    return out


def sel_score(Yv, probs):
    """Model-selection score on validation data: mean over tasks of (macro-F1 + QWK) / 2."""
    ms = [metrics(Yv[:, t], probs[t]) for t in range(len(TASKS))]
    return float(np.mean([(d["macro_f1"] + d["qwk"]) / 2 for d in ms if d["n"]]))


def majority_baseline(m, Y, n_cls):
    """Val macro-F1 / QWK of always predicting the training majority class (the floor every model must beat)."""
    res = {}
    for t, K in enumerate(n_cls):
        s = []
        for f in sorted(x for x in m["fold"].unique() if x >= 0):
            tr, va = Y[((m["fold"] >= 0) & (m["fold"] != f)).to_numpy(), t], Y[(m["fold"] == f).to_numpy(), t]
            P = np.eye(K)[np.full(len(va), np.bincount(tr[tr >= 0], minlength=K).argmax())]
            s.append(metrics(va, P))
        res[TASKS[t]] = {k: float(np.mean([d[k] for d in s])) for k in ("macro_f1", "qwk")}
    return res


# ---------------------------------------------------------------- frozen-feature probes (linear / kNN)
def _fit_predict(Xtr, ytr, Xs, K, kind):
    if kind == "knn":   # cosine kNN, k=20, temperature-weighted votes (DINO protocol)
        norm = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)  # noqa: E731
        A, outs = norm(Xtr), []
        for X in Xs:
            S = norm(X) @ A.T; k = min(20, len(A)); idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
            w = np.exp(np.take_along_axis(S, idx, 1) / .07); P = np.zeros((len(X), K))
            np.add.at(P, (np.repeat(np.arange(len(X)), k), ytr[idx].ravel()), w.ravel()); outs.append(P / P.sum(1, keepdims=True))
        return outs
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=.5, max_iter=3000, class_weight="balanced")).fit(Xtr, ytr)
    outs = []
    for X in Xs:
        P = np.zeros((len(X), K)); P[:, clf.classes_] = clf.predict_proba(X); outs.append(P)
    return outs


def probe_cv(Fe, m, Y, n_cls, kind="linear", frac=1.0, folds=None, seed=0):
    """Probe on each CV fold (train = other folds, optionally only `frac` of the labels) -> metric rows,
    out-of-fold val probabilities and locked-test probabilities averaged over the fold models."""
    folds = folds if folds is not None else sorted(int(f) for f in m["fold"].unique() if f >= 0)
    te, rng, rows = np.where(m["fold"] == -1)[0], np.random.default_rng(seed), []
    oof, tp = [np.full((len(m), K), np.nan) for K in n_cls], [np.zeros((len(te), K)) for K in n_cls]
    for f in folds:
        tr, va = np.where((m["fold"] >= 0) & (m["fold"] != f))[0], np.where(m["fold"] == f)[0]
        for t, K in enumerate(n_cls):
            trt = tr[Y[tr, t] >= 0]
            if frac < 1:
                trt = np.concatenate([rng.choice(trt[Y[trt, t] == c], max(1, round(frac * (Y[trt, t] == c).sum())), replace=False)
                                      for c in np.unique(Y[trt, t])])
            Pva, Pte = _fit_predict(Fe[trt], Y[trt, t], [Fe[va], Fe[te]], K, kind)
            oof[t][va], tp[t] = Pva, tp[t] + Pte / len(folds)
            rows += [{"fold": str(f), "task": TASKS[t], "split": "val", **metrics(Y[va, t], Pva)},
                     {"fold": str(f), "task": TASKS[t], "split": "test", **metrics(Y[te, t], Pte)}]
    rows += [{"fold": "ens", "task": TASKS[t], "split": "test", **metrics(Y[te, t], tp[t])} for t in range(len(n_cls))]
    return pd.DataFrame(rows), oof, tp


# ---------------------------------------------------------------- fine-tuning
@torch.no_grad()
def predict(model, dl):
    model.eval(); outs = [[] for _ in TASKS]
    for x, _ in dl:
        with autocast(): o = model(x.to(DEVICE, non_blocking=True))
        for t in range(len(TASKS)): outs[t].append(o[t].float().softmax(1).cpu().numpy())
    return [np.concatenate(o) for o in outs]


def finetune(name, mode, fold, m, imgs, Y, n_cls, cw, epochs, bs=32, patience=7):
    """Train encoder (partial/full) + heads on the other folds, keep the epoch with the best validation score.
    Returns (val probs, locked-test probs, learning curve). The test set is only predicted, never used to choose."""
    enc = Encoder(name); enc.set_mode(mode); model = Grader(enc, n_cls).to(DEVICE)
    tr = np.where((m["fold"] >= 0) & (m["fold"] != fold))[0]
    va, te = np.where(m["fold"] == fold)[0], np.where(m["fold"] == -1)[0]
    dl_tr = loader(EmbryoDS(imgs, tr, Y, enc.mean, enc.std, TRAIN_AUG), bs, True)
    dl_va, dl_te = (loader(EmbryoDS(imgs, i, Y, enc.mean, enc.std), 64, False) for i in (va, te))
    lr_bb = (1e-4 if "resnet" in name else 2e-5) if mode == "full" else 1e-4
    opt = torch.optim.AdamW([{"params": [p for p in enc.parameters() if p.requires_grad], "lr": lr_bb},
                             {"params": model.heads.parameters(), "lr": 1e-3}], weight_decay=.05)
    sched, scaler = cosine_sched(opt, epochs * len(dl_tr), len(dl_tr)), make_scaler()
    crit = [nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE), ignore_index=-1, label_smoothing=.05)
            for w in cw]
    best, bad, state, curve = -1.0, 0, None, []
    for ep in range(epochs):
        model.train(); tot, nb = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE)
            with autocast(): outs = model(x)
            losses = [c(o.float(), y[:, t]) for t, (c, o) in enumerate(zip(crit, outs)) if (y[:, t] >= 0).any()]
            if not losses: continue
            loss = sum(losses); opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot, nb = tot + loss.item(), nb + 1
        s = sel_score(Y[va], predict(model, dl_va))
        curve.append({"epoch": ep + 1, "train_loss": tot / max(nb, 1), "val_score": s})
        if s > best: best, bad, state = s, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: bad += 1
        if bad >= patience or not math.isfinite(tot): break
    model.load_state_dict(state)
    Pva, Pte = predict(model, dl_va), predict(model, dl_te)
    del model, enc; torch.cuda.empty_cache()
    return Pva, Pte, curve


# ---------------------------------------------------------------- plots
def save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def grouped_bars(ax, df, x, hue, val, err=None, hline=None, title="", ylabel=""):
    xs, hs = list(dict.fromkeys(df[x])), list(dict.fromkeys(df[hue])); w = .8 / max(len(hs), 1)
    for j, h in enumerate(hs):
        d = df[df[hue] == h].set_index(x).reindex(xs)
        ax.bar(np.arange(len(xs)) + (j - (len(hs) - 1) / 2) * w, d[val], w, label=str(h),
               yerr=None if err is None else d[err].fillna(0), capsize=2)
    ax.set_xticks(range(len(xs)), xs, rotation=30, ha="right", fontsize=8); ax.set_title(title, fontsize=9); ax.set_ylabel(ylabel)
    if hline is not None: ax.axhline(hline, ls="--", c="gray", lw=1)


def heat(ax, M, xl, yl, title, fmt="{:d}"):
    M = np.asarray(M, dtype=float) if fmt != "{:d}" else np.asarray(M); ax.imshow(M, cmap="Blues"); ax.set_title(title, fontsize=9)
    ax.set_xticks(range(len(xl)), xl, fontsize=8); ax.set_yticks(range(len(yl)), yl, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "-" if np.isnan(M[i, j]) else fmt.format(M[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > .6 * np.nanmax(M) else "black")


# ---------------------------------------------------------------- verdict (terminal indicator)
def verdict(phase, checks, key_lines, out_dir):
    """Print + save a PASS / WARN / FAIL checklist. checks = [(status, name, detail), ...]"""
    st = Counter(s for s, _, _ in checks)
    overall = "FAIL" if st["FAIL"] else "PASS WITH WARNINGS" if st["WARN"] else "PASS"
    nxt = {"FAIL": f"Fix the FAIL items before Phase {phase + 1}.",
           "PASS WITH WARNINGS": f"OK to continue to Phase {phase + 1}; WARN items are worth a look.",
           "PASS": f"Ready for Phase {phase + 1}."}[overall]
    bar = "=" * 100
    L = [bar, f" PHASE {phase} VERDICT:  {overall}    (PASS {st['PASS']}, WARN {st['WARN']}, FAIL {st['FAIL']})", bar]
    L += [f" [{s}] {n + ' ':.<56} {d}" for s, n, d in checks] + ["-" * 100, " Key numbers"] + [f"  {k}" for k in key_lines]
    L += ["-" * 100, f" Next: {nxt}", f" Results folder: {Path(out_dir).resolve()}",
          " Copy everything between the ==== lines and send it back for review.", bar]
    txt = "\n".join(L); (Path(out_dir) / f"PHASE{phase}_VERDICT.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt, flush=True)
    return overall


def run_safely(phase, main):
    try: return main()
    except Exception as err:
        import builtins, traceback
        print("\n" + "=" * 100 + f"\n PHASE {phase} VERDICT:  DID NOT RUN\n {type(err).__name__}: {err}\n"
              " Copy this block (and the traceback below) and send it back.\n" + "=" * 100, flush=True)
        if hasattr(builtins, "get_ipython"):          # Jupyter: plain traceback (IPython's formatter can crash on long cells)
            traceback.print_exc(); return None
        raise


# ###################################### PHASE 3 ######################################
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
if QUICK_TEST: MAE_EPOCHS, FEWSHOT_FOLDS = 2, [0]
MAE_AUG = T.Compose([T.RandomResizedCrop(IMG_SIZE, scale=(.5, 1), ratio=(.9, 1.1), antialias=True),
                     T.RandomHorizontalFlip(), T.RandomVerticalFlip(), RandRot90()])
PERTURB = {"flip+rot90": T.Compose([T.RandomHorizontalFlip(1.), lambda x: torch.rot90(x, 1, (1, 2))]),
           "brightness/contrast": T.ColorJitter(.3, .3), "blur": T.GaussianBlur(7, 2.),
           "crop 70-85%": T.RandomResizedCrop(IMG_SIZE, scale=(.7, .85), antialias=True)}
SUPERVISED, SSL_MODELS = {"resnet50", "vit_b16"}, {"dinov2_b14", "mae_in1k", "femi", "mae_in1k_dapt", "femi_dapt"}


@torch.no_grad()
def recon_figure(model, imgs, idx, mean, std, size, path, title, n=6):
    """Original / masked input / MAE reconstruction for n SSL images."""
    model.eval(); sel = idx[np.linspace(0, len(idx) - 1, n).astype(int)]
    ds = EmbryoDS(imgs, sel, None, mean, std); x = torch.stack([ds[i][0] for i in range(n)]).to(DEVICE)
    if x.shape[-1] != size: x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    torch.manual_seed(0); o = model(pixel_values=x); pred, p = o.logits.float(), model.config.patch_size
    if model.config.norm_pix_loss:
        tgt = model.patchify(x); pred = pred * (tgt.var(-1, keepdim=True) + 1e-6) ** .5 + tgt.mean(-1, keepdim=True)
    rec, mk = model.unpatchify(pred), model.unpatchify(o.mask.float().unsqueeze(-1).repeat(1, 1, p * p * 3))
    mu, sd = (torch.tensor(v, device=DEVICE).view(1, 3, 1, 1) for v in (mean, std))
    show = lambda t: (t * sd + mu).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()  # noqa: E731
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6.4))
    for r, (lab, arr) in enumerate([("original", show(x)), (f"masked input ({MASK_RATIO:.0%})", show(x * (1 - mk))),
                                    ("reconstruction", show(x * (1 - mk) + rec * mk))]):
        for c in range(n): axes[r, c].imshow(arr[c]); axes[r, c].axis("off")
        axes[r, 0].set_title(lab, fontsize=9, loc="left")
    fig.suptitle(f"MAE after domain adaptation - start: {title}"); save(fig, path)


import hashlib, importlib, re, traceback, types  # noqa: E402


def record_error(where, e):
    """Save the full traceback to errors.log and return a plain-language reason."""
    with open(OUT / "errors.log", "a") as fh: fh.write(f"\n===== {where} =====\n{traceback.format_exc()}")
    s = f"{type(e).__name__}: {e}"
    if re.search(r"torch\.load|upgrade torch|v2\.6|CVE-2025", s, re.I):
        msg = "transformers needs PyTorch >= 2.6 to read this .bin checkpoint (fix: pip install -U torch torchvision)"
    elif re.search(r"gated|401|403|restricted|awaiting|authorized list", s, re.I):
        msg = ("no access to this gated model: open huggingface.co/ihlab/FEMI, press 'Agree and access repository', wait for "
               "approval, then set HF_TOKEN at the top of Phase 2/3 (or run huggingface-cli login)")
    elif re.search(r"ConnectionError|Timeout|resolve host|offline|SSLError|proxy|MaxRetry", s, re.I):
        msg = "cannot reach huggingface.co (network / proxy)"
    elif re.search(r"out of memory", s, re.I): msg = "GPU out of memory (lower MAE_BATCH)"
    elif isinstance(e, ImportError): msg = f"missing package ({e})"
    else: msg = s[:200]
    log(f"PROBLEM in {where}: {msg}   [full traceback: {OUT / 'errors.log'}]")
    return msg


def hf_check(model_id):
    """None if Hugging Face transformers can load this ViT-MAE here, otherwise the reason (installs transformers if missing)."""
    for attempt in (1, 2):
        try:
            from transformers import ViTMAEForPreTraining, logging as hf_logging
            hf_logging.set_verbosity_error(); ViTMAEForPreTraining.from_pretrained(model_id); return None
        except ImportError as e:
            if attempt == 2: return record_error(f"importing transformers for {model_id}", e)
            import subprocess
            log("transformers is not importable in this Python - installing it here"); importlib.invalidate_caches()
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers"], check=False); importlib.invalidate_caches()
        except Exception as e:  # noqa: BLE001
            return record_error(f"loading {model_id} with transformers", e)


def versions():
    out = [f"python {sys.version.split()[0]}", f"torch {torch.__version__}"]
    for pkg in ("transformers", "timm"):
        try: out.append(f"{pkg} {importlib.import_module(pkg).__version__}")
        except Exception: out.append(f"{pkg} not installed")  # noqa: BLE001
    return ", ".join(out)


def sincos_2d(dim, g):
    """Fixed 2-D sine-cosine position embedding (1, 1 + g*g, dim); first row = class token (MAE decoder)."""
    y, x = torch.meshgrid(torch.arange(g, dtype=torch.float32), torch.arange(g, dtype=torch.float32), indexing="ij")
    om = 1.0 / 10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))
    pe = torch.cat([f(c.reshape(-1, 1) * om) for c in (x, y) for f in (torch.sin, torch.cos)], 1)
    return torch.cat([torch.zeros(1, dim), pe], 0).unsqueeze(0)


class TimmMAE(nn.Module):
    """Minimal MAE (He et al. 2022) on timm's ImageNet-MAE encoder - no transformers needed. Same interface as the
    Hugging Face model: model(pixel_values=x).loss / .logits / .mask, .config, .patchify, .unpatchify, .save_pretrained."""
    def __init__(self, arch=TIMM_MAE, dec_dim=512, dec_depth=4):
        super().__init__()
        import timm
        from timm.models.vision_transformer import Block
        self.enc = timm.create_model(arch, pretrained=True, num_classes=0); e = self.enc
        p, N, D = e.patch_embed.patch_size[0], e.patch_embed.num_patches, e.embed_dim
        self.arch, self.config = arch, types.SimpleNamespace(image_size=IMG_SIZE, patch_size=p, mask_ratio=MASK_RATIO, norm_pix_loss=True)
        self.dec_embed, self.mask_token = nn.Linear(D, dec_dim), nn.Parameter(torch.randn(1, 1, dec_dim) * .02)
        self.register_buffer("dec_pos", sincos_2d(dec_dim, math.isqrt(N)), persistent=False)
        self.dec = nn.Sequential(*[Block(dec_dim, 16, 4.0) for _ in range(dec_depth)])
        self.dec_norm, self.dec_pred = nn.LayerNorm(dec_dim), nn.Linear(dec_dim, p * p * 3)

    def patchify(self, x):
        p = self.config.patch_size; B, C, H, W = x.shape; g = H // p
        return x.reshape(B, C, g, p, g, p).permute(0, 2, 4, 3, 5, 1).reshape(B, g * g, p * p * C)

    def unpatchify(self, t):
        p = self.config.patch_size; B, N, _ = t.shape; g = math.isqrt(N)
        return t.reshape(B, g, g, p, p, 3).permute(0, 5, 1, 3, 2, 4).reshape(B, 3, g * p, g * p)

    def forward(self, pixel_values):
        e, x = self.enc, pixel_values; B = len(x)
        t = e.patch_embed(x) + e.pos_embed[:, e.num_prefix_tokens:]
        N, D = t.shape[1], t.shape[2]; keep = max(1, int(N * (1 - self.config.mask_ratio)))
        ids_shuffle = torch.rand(B, N, device=x.device).argsort(1); ids_restore = ids_shuffle.argsort(1)
        t = torch.gather(t, 1, ids_shuffle[:, :keep, None].expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device); mask[:, :keep] = 0; mask = torch.gather(mask, 1, ids_restore)   # 1 = hidden
        t = e.norm(e.blocks(torch.cat([(e.cls_token + e.pos_embed[:, :1]).expand(B, -1, -1), t], 1)))
        d = self.dec_embed(t); dd = d.shape[-1]
        d_ = torch.cat([d[:, 1:], self.mask_token.expand(B, N - keep, -1)], 1)
        d = torch.cat([d[:, :1], torch.gather(d_, 1, ids_restore[..., None].expand(-1, -1, dd))], 1) + self.dec_pos
        pred = self.dec_pred(self.dec_norm(self.dec(d)))[:, 1:]
        tgt = self.patchify(x)
        if self.config.norm_pix_loss: tgt = (tgt - tgt.mean(-1, keepdim=True)) / (tgt.var(-1, keepdim=True) + 1e-6) ** .5
        loss = (((pred.float() - tgt.float()) ** 2).mean(-1) * mask).sum() / mask.sum()
        return types.SimpleNamespace(loss=loss, logits=pred, mask=mask)

    def save_pretrained(self, out):
        out = Path(out); out.mkdir(parents=True, exist_ok=True); torch.save(self.enc.state_dict(), out / "encoder.pth")
        (out / "backbone.json").write_text(json.dumps({"lib": "timm_tokens", "arch": self.arch, "weights": "encoder.pth"}))


def register(name, out):
    """Make an adapted encoder saved in the cache usable by Encoder (and by Phase 5)."""
    j = json.loads((out / "backbone.json").read_text())
    BACKBONES[name] = ("timm_tokens", f"{j['arch']}|{out / j['weights']}") if j["lib"] == "timm_tokens" else ("hf", str(out))


def train_mae(init, hub_id, backend, m, imgs):
    """Continue MAE pretraining on the SSL-allowed images; saves the adapted encoder to CACHE_DIR/<init>_dapt.
    backend 'hf' = Hugging Face ViT-MAE (pretrained decoder); 'timm' = timm encoder + new light decoder (warmed up first)."""
    if backend == "hf":
        from transformers import ViTMAEForPreTraining, logging as hf_logging
        hf_logging.set_verbosity_error(); model = ViTMAEForPreTraining.from_pretrained(hub_id); mean, std = hf_stats(hub_id)
    else:
        from timm.data import resolve_data_config
        model = TimmMAE(); cfg = resolve_data_config({}, model=model.enc); mean, std = tuple(cfg["mean"]), tuple(cfg["std"])
    model.config.mask_ratio = MASK_RATIO; model.to(DEVICE)
    size, ssl = model.config.image_size, np.where(m["ssl_allowed"].astype(bool))[0]
    warm = max(3, MAE_EPOCHS // 20) if backend == "timm" else 0                 # epochs with the encoder frozen
    dl = loader(EmbryoDS(imgs, ssl, None, mean, std, MAE_AUG), MAE_BATCH, True)
    opt = torch.optim.AdamW(model.parameters(), lr=MAE_BASE_LR * MAE_BATCH / 256, betas=(.9, .95), weight_decay=.05)
    sched, scaler, hist = cosine_sched(opt, MAE_EPOCHS * len(dl), len(dl) * max(1, MAE_EPOCHS // 20)), make_scaler(), []
    for ep in range(MAE_EPOCHS):
        model.train(); tot = 0.0
        if backend == "timm":
            for p_ in model.enc.parameters(): p_.requires_grad = ep >= warm
        for x, _ in dl:
            x = x.to(DEVICE, non_blocking=True)
            if x.shape[-1] != size: x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            with autocast(): loss = model(pixel_values=x).loss
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        hist.append(tot / len(dl))
        if ep % 10 == 0 or ep == MAE_EPOCHS - 1: log(f"   MAE {init}: epoch {ep + 1}/{MAE_EPOCHS}  loss {hist[-1]:.4f}")
        if not math.isfinite(hist[-1]): break
    out = CACHE_DIR / f"{init}_dapt"; model.save_pretrained(out)
    if backend == "hf":
        (out / "backbone.json").write_text(json.dumps({"lib": "hf"}))
        try:
            from transformers import AutoImageProcessor
            AutoImageProcessor.from_pretrained(hub_id).save_pretrained(out)
        except Exception: pass  # noqa: BLE001
    (out / "loss_history.json").write_text(json.dumps(hist))
    try: recon_figure(model, imgs, ssl, mean, std, size, OUT / f"mae_reconstruction_{init}.png", f"{init} ({backend})")
    except Exception as e: record_error(f"reconstruction figure for {init}", e)  # noqa: BLE001  a figure never discards the model
    model.cpu(); del model, opt; torch.cuda.empty_cache()
    return hist


@torch.no_grad()
def aug_consistency(name, imgs, idx):
    """Centred cosine similarity between features of an image and a perturbed copy (1 = invariant)."""
    enc, res = Encoder(name).to(DEVICE).eval(), {}

    def feats(aug):
        out = []
        for x, _ in loader(EmbryoDS(imgs, idx, None, enc.mean, enc.std, aug), 64, False, workers=0):
            with autocast(): out.append(enc(x.to(DEVICE)).float())
        return torch.cat(out)
    base = feats(None); mu = base.mean(0, keepdim=True); b = F.normalize(base - mu, dim=1)
    for k, aug in PERTURB.items():
        torch.manual_seed(0); random.seed(0); res[k] = float((b * F.normalize(feats(aug) - mu, dim=1)).sum(1).mean())
    del enc; torch.cuda.empty_cache()
    return res


def evaluate(name, m, imgs, Y, n_cls, lab, cidx):
    Fe = extract_features(name, m, imgs, force=name.endswith("_dapt") or name == "mae_in1k")   # MAE back-end may change
    lin, knn = probe_cv(Fe, m, Y, n_cls)[0], probe_cv(Fe, m, Y, n_cls, "knn")[0]
    few = []
    for frac in FEWSHOT:
        d = probe_cv(Fe, m, Y, n_cls, frac=frac, folds=FEWSHOT_FOLDS)[0]; d = d[d["split"] == "val"]
        few.append({"model": name, "frac": frac, "macro_f1": d["macro_f1"].mean(), "qwk": d["qwk"].mean()})
    Z = PCA(50, random_state=0).fit_transform(StandardScaler().fit_transform(Fe[lab]))
    sil = silhouette_score(Z, Y[lab, 0], metric="cosine", sample_size=min(2000, len(lab)), random_state=0)
    return lin, knn, few, TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(Z), sil, aug_consistency(name, imgs, cidx)


def figures(hists, P, FS, tsne, cons, Y, lab, maj):
    if hists:
        fig, ax = plt.subplots(figsize=(7, 4))
        for k, h in hists.items(): ax.plot(range(1, len(h) + 1), h, label=f"from {k}")
        ax.set(xlabel="epoch", ylabel="MAE reconstruction loss", title="Domain-adaptive MAE pretraining"); ax.legend()
        save(fig, OUT / "mae_loss_curves.png")
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    for r, met in enumerate(("macro_f1", "qwk")):
        for c, t in enumerate(TASKS):
            grouped_bars(axes[r, c], P[P["task"] == t], "model", "probe", f"{met}_mean", f"{met}_std", maj[t][met],
                         f"{t} - frozen features, validation {met} (dashed = majority class)", met)
    axes[0, 0].legend(fontsize=8); save(fig, OUT / "representation_probes.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, met in zip(axes, ("macro_f1", "qwk")):
        for mod, d in FS.groupby("model", sort=False):
            ax.plot(d["frac"], d[met], "o-", label=mod, lw=2.5 if mod.endswith("_dapt") else 1.2)
        ax.set(xscale="log", xlabel="fraction of training labels", ylabel=f"val {met} (mean over tasks)",
               title=f"Label efficiency - linear probe {met}"); ax.legend(fontsize=7)
    save(fig, OUT / "fewshot_curves.png")
    n = len(tsne); cols = min(4, n); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.8 * rows), squeeze=False)
    for ax, (mod, Z) in zip(axes.ravel(), tsne.items()):
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=Y[lab, 0] + 1, cmap="viridis", s=4); ax.set_title(mod, fontsize=9); ax.axis("off")
    for ax in axes.ravel()[n:]: ax.axis("off")
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="EXP grade", shrink=.6); fig.savefig(OUT / "tsne_EXP.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    D = pd.DataFrame([{"model": k, "perturbation": p, "cos": v} for k, d in cons.items() for p, v in d.items()])
    fig, ax = plt.subplots(figsize=(12, 4))
    grouped_bars(ax, D, "perturbation", "model", "cos", None, None, "Augmentation consistency (centred cosine, higher = more invariant)", "cosine")
    ax.legend(fontsize=7, ncol=4); save(fig, OUT / "aug_consistency.png")


def main():
    t0 = time.time(); seed_all(); OUT.mkdir(parents=True, exist_ok=True)
    m = load_manifest(); Y, n_cls = targets(m); imgs = image_cache(m)
    ssl = m["ssl_allowed"].astype(bool); ssl_test = int((ssl & (m["fold"] == -1)).sum())
    if ssl_test: raise RuntimeError(f"{ssl_test} gold-test images are marked ssl_allowed - rerun Phase 1")
    (OUT / "errors.log").unlink(missing_ok=True); env = versions(); log(f"Environment: {env}"); hf_status = hf_login(); log(hf_status)
    pool = hashlib.md5("|".join(sorted(m.loc[ssl, "image_key"].astype(str))).encode()).hexdigest()   # SSL image set fingerprint
    hists, skipped, backend, why = {}, {}, {}, {}
    for init, hub in MAE_INITS.items():                                        # 1) which library can load each MAE here?
        why[init] = None if MAE_BACKEND == "timm" else hf_check(hub)
        if MAE_BACKEND != "timm" and why[init] is None: backend[init] = "hf"
        elif init == "mae_in1k" and MAE_BACKEND in ("auto", "timm"): backend[init] = "timm"; BACKBONES["mae_in1k"] = ("timm_tokens", TIMM_MAE)
        else: backend[init] = None
        log(f"MAE backend for {init}: {backend[init] or 'unavailable'}" + (f" (Hugging Face: {why[init]})" if why[init] else ""))
    for init, hub in MAE_INITS.items():                                        # 2) domain-adaptive MAE pretraining
        name, out = f"{init}_dapt", CACHE_DIR / f"{init}_dapt"
        if backend[init] is None: skipped[name] = why[init] or "FEMI is only available through Hugging Face transformers"; continue
        hf_file = out / "loss_history.json"
        pf = out / "ssl_pool.txt"; same_pool = not pf.exists() or pf.read_text().strip() == pool   # legacy caches: no fingerprint
        if not RETRAIN_MAE and same_pool and (out / "backbone.json").exists() and hf_file.exists() and len(json.loads(hf_file.read_text())) == MAE_EPOCHS:
            hists[init] = json.loads(hf_file.read_text()); register(name, out); log(f"== reusing adapted encoder {name} from the cache"); continue
        log(f"== MAE domain adaptation from {init} ({backend[init]}) on {int(ssl.sum())} SSL-allowed images")
        if not same_pool: log(f"   SSL image set changed since {name} was trained (Phase 1 rerun) - retraining")
        try: hists[init] = train_mae(init, hub, backend[init], m, imgs); register(name, out); pf.write_text(pool)
        except Exception as e: skipped[name] = record_error(f"MAE adaptation from {init}", e)  # noqa: BLE001
    (CACHE_DIR / "phase3_backbones.json").write_text(json.dumps({k: list(v) for k, v in BACKBONES.items() if k.startswith(("mae_in1k", "femi"))}))

    lab = np.where((m["fold"] >= -1) & (Y[:, 0] >= 0))[0]
    cidx = np.random.default_rng(0).choice(np.where(ssl)[0], min(256, int(ssl.sum())), replace=False)
    P, FS, tsne, cons, rows = [], [], {}, {}, []
    for name in COMPARE + [f"{k}_dapt" for k in MAE_INITS if f"{k}_dapt" in BACKBONES]:
        if name in MAE_INITS and backend.get(name) is None:                    # e.g. FEMI without access: explained once
            skipped[name] = why[name] or "not available"; continue
        log(f"== evaluating {name}")
        try: lin, knn, few, tsne[name], sil, cons[name] = evaluate(name, m, imgs, Y, n_cls, lab, cidx)
        except Exception as e:  # noqa: BLE001
            skipped[name] = record_error(f"evaluating {name}", e); continue
        for probe, D in (("linear", lin), ("kNN", knn)):
            g = D[(D["split"] == "val") & (D["fold"] != "ens")].groupby("task", sort=False)[["macro_f1", "qwk"]].agg(["mean", "std"])
            g.columns = [f"{a}_{b}" for a, b in g.columns]; P.append(g.reset_index().assign(model=name, probe=probe))
        FS += few; ens = lin[lin["fold"] == "ens"]
        rows.append({"model": name, "lin_f1": P[-2]["macro_f1_mean"].mean(), "lin_qwk": P[-2]["qwk_mean"].mean(),
                     "test_qwk": ens["qwk"].mean(), "knn_f1": P[-1]["macro_f1_mean"].mean(),
                     "few10_f1": next(f["macro_f1"] for f in few if f["frac"] == .1), "few100_f1": few[-1]["macro_f1"],
                     "consistency": float(np.mean(list(cons[name].values()))), "silhouette_EXP": sil})
    if not rows: raise RuntimeError(f"No encoder could be evaluated: {skipped}")
    P, FS, S = pd.concat(P, ignore_index=True), pd.DataFrame(FS), pd.DataFrame(rows).set_index("model")
    S["score"] = (S["lin_f1"] + S["lin_qwk"]) / 2; S = S.sort_values("score", ascending=False)
    for n, d in (("summary", S), ("probe_details", P), ("fewshot", FS)): d.to_csv(OUT / f"{n}.csv")
    if hists: pd.DataFrame({k: pd.Series(v) for k, v in hists.items()}).to_csv(OUT / "mae_loss_history.csv", index_label="epoch")
    maj = majority_baseline(m, Y, n_cls)
    figures(hists, P, FS, tsne, cons, Y, lab, maj)

    # ---------------- verdict
    s2 = phase_status("phase2_outputs", "PHASE2_VERDICT.txt")
    checks = [("PASS" if s2 and "FAIL" not in s2 else "FAIL" if s2 else "WARN", "Phase 2 verdict", s2 or "PHASE2_VERDICT.txt not found"),
              ("PASS", "SSL pool excludes the locked gold test set", f"{int(ssl.sum())} SSL images, 0 test images"),
              ("INFO", "Environment", env), ("INFO", "Hugging Face access", hf_status)]
    for init in MAE_INITS:
        b = backend.get(init)
        checks.append(("PASS" if b == "hf" else "PASS" if b == "timm" else "WARN", f"MAE backend for {init}",
                       "Hugging Face transformers" if b == "hf" else f"timm fallback (Hugging Face: {why[init]})" if b == "timm"
                       else (why[init] or "not available")))
    for init in MAE_INITS:
        h = hists.get(init)
        if h is None: checks.append(("WARN", f"MAE adaptation from {init}", skipped.get(f"{init}_dapt", "not run")))
        elif not all(map(math.isfinite, h)): checks.append(("FAIL", f"MAE adaptation from {init}", "loss became NaN/inf"))
        else: checks.append(("PASS" if h[-1] < h[0] else "WARN", f"MAE adaptation from {init} converged",
                             f"loss {h[0]:.4f} -> {h[-1]:.4f} ({h[-1] / h[0] - 1:+.1%}) in {len(h)} epochs"))
        if f"{init}_dapt" in S.index and init in S.index:
            d = S.loc[f"{init}_dapt", "score"] - S.loc[init, "score"]
            checks.append(("PASS" if d > 0 else "WARN", f"Domain adaptation improves {init} (probe score)",
                           f"{S.loc[init, 'score']:.3f} -> {S.loc[f'{init}_dapt', 'score']:.3f} ({d:+.3f})"))
    if (OUT / "errors.log").exists(): checks.append(("INFO", "Full error details", str(OUT / "errors.log")))
    chance = np.mean([maj[t]["macro_f1"] for t in TASKS]) + .05
    weak = S.index[S["knn_f1"] <= chance].tolist()
    ssl_best = S[S.index.isin(SSL_MODELS)]["score"]; sup_best = S[S.index.isin(SUPERVISED)]["score"]
    checks += [
        ("PASS" if not [k for k in skipped if k in COMPARE] else "WARN", "All encoders evaluated",
         "yes" if not skipped else "; ".join(f"{k}: {v}" for k, v in skipped.items())[:150]),
        ("PASS" if not weak else "WARN", "kNN on frozen features beats majority class", ", ".join(weak) or "all encoders"),
        ("INFO", "RQ1: best SSL vs best supervised representation",
         f"{ssl_best.idxmax()} {ssl_best.max():.3f} vs {sup_best.idxmax()} {sup_best.max():.3f}" if len(ssl_best) and len(sup_best) else "n/a"),
        ("INFO", "Label efficiency of best encoder (10% vs 100% labels)",
         f"{S.index[0]}: macro-F1 {S.iloc[0]['few10_f1']:.3f} vs {S.iloc[0]['few100_f1']:.3f} "
         f"({S.iloc[0]['few10_f1'] / max(S.iloc[0]['few100_f1'], 1e-9):.0%})")]
    keys = ["Encoder ranking (frozen features; val = mean over EXP/ICM/TE and folds):",
            "  model              probe F1  probe QWK  test QWK  kNN F1  10%-labels F1  consistency  silhouette(EXP)"]
    keys += [f"  {k:<18} {r.lin_f1:.3f}     {r.lin_qwk:.3f}      {r.test_qwk:.3f}     {r.knn_f1:.3f}   {r.few10_f1:.3f}          "
             f"{r.consistency:.3f}        {r.silhouette_EXP:.3f}" for k, r in S.iterrows()]
    keys += [f"MAE: epochs {MAE_EPOCHS}, batch {MAE_BATCH}, mask {MASK_RATIO}, SSL images {int(ssl.sum())} | few-shot folds {FEWSHOT_FOLDS} | "
             f"device {DEVICE}, quick_test {QUICK_TEST}, runtime {(time.time() - t0) / 60:.1f} min"]
    verdict(3, checks, keys, OUT)
    return S


if __name__ == "__main__":
    run_safely(3, main)
