"""MorphoGround - PHASE 2: baseline ladder on the Phase 1 folds.
Each backbone: frozen (linear probe) -> partial fine-tuning (last 2 blocks) -> full fine-tuning, 3 Gardner heads.
Model selection uses validation folds only; the locked gold test set is predicted by every fold model and averaged.
v2: rerun after Phase 1 v2 (folds now grouped by patient/cycle); Hugging Face MAE/FEMI encoders load with transformers 4.x and
5.x (block lookup no longer assumes net.encoder.layer); HF_TOKEN setting for the gated FEMI model.
Self-contained (no other files needed): paste into ONE Jupyter cell, or run  python phase2_baselines.py"""
from pathlib import Path

# ====================================== USER SETTINGS =======================================
PROJECT_DIR = Path.home() / "Embryo_Updated Project"   # same folder as Phase 1
PHASE1_DIR  = PROJECT_DIR / "phase1_outputs"           # manifest.csv + class_weights.json from Phase 1
CACHE_DIR   = PROJECT_DIR / "cache"                     # resized images, features, adapted models (reused)
IMG_SIZE    = 224
NUM_WORKERS = 4
QUICK_TEST  = False   # True = fast smoke test (1 fold, 2 epochs) to check that everything runs

# ---- PHASE 2 settings
HF_TOKEN    = None   # FEMI is gated: open huggingface.co/ihlab/FEMI, press "Agree and access", wait for approval,
                     # then paste a READ token from huggingface.co/settings/tokens here, e.g. "hf_xxx" (stored for later phases)
RUN_BACKBONES = ["resnet50", "vit_b16", "dinov2_b14", "femi"]  # femi is skipped (WARN) without Hugging Face access
FT_MODES      = ["partial", "full"]                            # 'frozen' (linear probe) always runs
FT_FOLDS      = [0, 1, 2, 3, 4]
EPOCHS, BATCH, PATIENCE = 25, 32, 7
KROMP_REF     = {"EXP": .77, "ICM": .51, "TE": .51}  # low end of Kromp et al. 2023 DL baselines (weighted F1, gold test)
OUT           = PROJECT_DIR / "phase2_outputs"
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
        if self.lib == "timm":
            import timm
            from timm.data import resolve_data_config
            self.net = timm.create_model(mid, pretrained=True, num_classes=0, **({"img_size": IMG_SIZE} if "dinov2" in mid else {}))
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
        return self.net(x) if self.lib == "timm" else hf_tokens(self.net, x).mean(1)


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


# ###################################### PHASE 2 ######################################
if QUICK_TEST: FT_FOLDS, EPOCHS = [0], 2


def run_backbone(name, m, imgs, Y, n_cls, cw, te, curves):
    """Frozen probe + fine-tuning modes for one backbone -> list of (mode, metric rows, oof probs, test probs)."""
    log(f"== {name}: frozen linear probe")
    runs = [("frozen", *probe_cv(extract_features(name, m, imgs), m, Y, n_cls))]
    for mode in FT_MODES:
        oof, tp, rows = [np.full((len(m), K), np.nan) for K in n_cls], [np.zeros((len(te), K)) for K in n_cls], []
        for f in FT_FOLDS:
            log(f"== {name}: {mode} fine-tuning, fold {f}")
            va = np.where(m["fold"] == f)[0]
            Pva, Pte, curve = finetune(name, mode, f, m, imgs, Y, n_cls, cw[f], EPOCHS, BATCH, PATIENCE)
            curves += [{**c, "model": name, "mode": mode, "fold": f} for c in curve]
            for t in range(len(TASKS)):
                oof[t][va], tp[t] = Pva[t], tp[t] + Pte[t] / len(FT_FOLDS)
                rows += [{"fold": str(f), "task": TASKS[t], "split": "val", **metrics(Y[va, t], Pva[t])},
                         {"fold": str(f), "task": TASKS[t], "split": "test", **metrics(Y[te, t], Pte[t])}]
        rows += [{"fold": "ens", "task": TASKS[t], "split": "test", **metrics(Y[te, t], tp[t])} for t in range(len(TASKS))]
        runs.append((mode, pd.DataFrame(rows), oof, tp))
    return runs


def figures(S, R, C, maj, test_probs, Y, te, n_cls):
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))                        # 1. the ladder (validation folds)
    for r, met in enumerate(("macro_f1", "qwk")):
        for c, t in enumerate(TASKS):
            grouped_bars(axes[r, c], S[S["task"] == t], "model", "mode", f"{met}_mean", f"{met}_std", maj[t][met],
                         f"{t} - validation {met} (mean +- sd over folds; dashed = majority class)", met)
    axes[0, 0].legend(fontsize=8); save(fig, OUT / "ladder_validation.png")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))                        # 2. locked test, fold ensemble
    for c, t in enumerate(TASKS):
        grouped_bars(axes[c], S[S["task"] == t], "model", "mode", "test_weighted_f1", None, KROMP_REF[t],
                     f"{t} - gold test weighted F1 (dashed = Kromp 2023 baseline low end)", "weighted F1")
    axes[0].legend(fontsize=8); save(fig, OUT / "test_weighted_f1.png")
    if len(C):                                                               # 3. learning curves
        models = list(dict.fromkeys(C["model"])); fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 3.6), squeeze=False)
        for ax, mod in zip(axes[0], models):
            for (mode, f), d in C[C["model"] == mod].groupby(["mode", "fold"]):
                ax.plot(d["epoch"], d["val_score"], color="C0" if mode == "partial" else "C1", alpha=.7, label=f"{mode}" if f == FT_FOLDS[0] else None)
            ax.set(title=mod, xlabel="epoch", ylabel="val score"); ax.legend(fontsize=7)
        save(fig, OUT / "learning_curves.png")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))                      # 4. confusion matrix of the best config
    for c, t in enumerate(TASKS):
        b = S[S["task"] == t].sort_values("sel", ascending=False).iloc[0]; K = n_cls[c]
        cm = confusion_matrix(Y[te, c][Y[te, c] >= 0], test_probs[(b["model"], b["mode"])][c][Y[te, c] >= 0].argmax(1), labels=list(range(K)))
        heat(axes[c], cm, CLASS_NAMES[t][:K], CLASS_NAMES[t][:K], f"{t}: {b['config']} (gold test, rows = true)")
    save(fig, OUT / "confusion_best.png")
    cols = [f"test_recall_{i}" for i in range(max(n_cls))]                   # 5. per-class recall (rare grades)
    M = np.vstack([np.concatenate([S[(S["config"] == cf) & (S["task"] == t)][cols[:n_cls[k]]].to_numpy().ravel()
                                   for k, t in enumerate(TASKS)]) for cf in dict.fromkeys(S["config"])])
    fig, ax = plt.subplots(figsize=(11, .45 * len(M) + 1.5))
    heat(ax, M, [f"{t}-{CLASS_NAMES[t][i]}" for k, t in enumerate(TASKS) for i in range(n_cls[k])], list(dict.fromkeys(S["config"])),
         "Gold-test recall per class (rare grades: EXP-1/2, ICM-C, TE-C)", "{:.2f}")
    save(fig, OUT / "per_class_recall.png")


def main():
    t0 = time.time(); seed_all(); (OUT / "preds").mkdir(parents=True, exist_ok=True); log(hf_login())
    m = load_manifest(); Y, n_cls = targets(m); imgs = image_cache(m); cw = phase1_class_weights()
    te = np.where(m["fold"] == -1)[0]; res, curves, skipped, test_probs = [], [], {}, {}
    for name in RUN_BACKBONES:
        try: runs = run_backbone(name, m, imgs, Y, n_cls, cw, te, curves)
        except Exception as e:  # noqa: BLE001  e.g. gated FEMI without access, or out-of-memory
            skipped[name] = f"{type(e).__name__}: {str(e)[:90]}"; log(f"SKIPPED {name}: {skipped[name]}"); continue
        for mode, df, oof, tp in runs:
            res.append(df.assign(model=name, mode=mode)); test_probs[(name, mode)] = tp
            np.savez_compressed(OUT / "preds" / f"{name}_{mode}.npz", keys=m["image_key"].astype(str).to_numpy(), test_idx=te,
                                **{f"oof_{k}": oof[t] for t, k in enumerate(TASKS)}, **{f"test_{k}": tp[t] for t, k in enumerate(TASKS)})
    if not res: raise RuntimeError(f"No backbone could be run: {skipped}")

    R = pd.concat(res, ignore_index=True); R.to_csv(OUT / "results_all.csv", index=False)
    C = pd.DataFrame(curves); C.to_csv(OUT / "learning_curves.csv", index=False)
    V = R[R["split"].eq("val")].groupby(["model", "mode", "task"], sort=False)[["macro_f1", "qwk", "weighted_f1", "acc"]].agg(["mean", "std"])
    V.columns = [f"{a}_{b}" for a, b in V.columns]
    E = R[R["fold"].eq("ens")].set_index(["model", "mode", "task"])
    S = V.join(E[["macro_f1", "qwk", "weighted_f1", "acc"] + [c for c in E if c.startswith("recall_")]].add_prefix("test_")).reset_index()
    S["config"], S["sel"] = S["model"] + "/" + S["mode"], (S["macro_f1_mean"] + S["qwk_mean"]) / 2
    S.to_csv(OUT / "summary.csv", index=False)
    maj = majority_baseline(m, Y, n_cls)
    figures(S, R, C, maj, test_probs, Y, te, n_cls)

    # ---------------- verdict
    s1, cv = phase_status("phase1_outputs", "QC_VERDICT.txt"), m[m["fold"] >= 0]
    split_ok = not (set(m.loc[m["fold"] == -1, "image_key"]) & set(cv["image_key"])) and cv.groupby("group_id")["fold"].nunique().max() == 1
    best = S.loc[S.groupby("task", sort=False)["sel"].idxmax()].set_index("task")
    weak = S[S.apply(lambda r: r["macro_f1_mean"] <= maj[r["task"]]["macro_f1"] + .05, axis=1)]
    incomplete = [n for n in S["model"].unique() if set(S.loc[S["model"] == n, "mode"]) != {"frozen", *FT_MODES}]
    gain = {n: S[(S["model"] == n) & (S["mode"] != "frozen")]["sel"].mean() - S[(S["model"] == n) & (S["mode"] == "frozen")]["sel"].mean()
            for n in S["model"].unique() if len(FT_MODES)}
    checks = [
        ("PASS" if s1 and "FAIL" not in s1 else "FAIL" if s1 else "WARN", "Phase 1 verdict", s1 or "QC_VERDICT.txt not found"),
        ("PASS" if split_ok else "FAIL", "Locked test set and grouped folds respected", f"{len(cv)} CV images, {len(te)} test images"),
        ("PASS" if not skipped else "WARN", "All backbones ran", "yes" if not skipped else "; ".join(f"{k}: {v}" for k, v in skipped.items())),
        ("PASS" if np.isfinite(R[["macro_f1", "qwk"]].to_numpy(float)).all() else "FAIL", "All runs finished with finite metrics",
         f"{R['model'].nunique()} backbones x {R['mode'].nunique()} modes"),
        ("PASS" if not incomplete else "WARN", "Frozen -> partial -> full ladder complete", ", ".join(incomplete) or "yes"),
        ("PASS" if weak.empty else "WARN", "Every config beats the majority-class baseline (+0.05 F1)",
         "yes" if weak.empty else f"{len(weak)} config/task pairs do not: " + ", ".join(weak["config"] + ":" + weak["task"])[:120]),
        ("PASS" if all(best.loc[t, "test_weighted_f1"] >= KROMP_REF[t] for t in TASKS) else "WARN", "Best config reaches Kromp 2023 baseline (test wF1)",
         " | ".join(f"{t} {best.loc[t, 'test_weighted_f1']:.3f} (ref {KROMP_REF[t]})" for t in TASKS)),
        ("PASS" if (best["macro_f1_std"].fillna(0) <= .08).all() else "WARN", "Stable across folds (sd of val macro-F1 <= 0.08)",
         " | ".join(f"{t} sd {best.loc[t, 'macro_f1_std']:.3f}" for t in TASKS)),
        ("INFO", "Fine-tuning gain over frozen probe (val score)", " | ".join(f"{k} {v:+.3f}" for k, v in gain.items())),
    ]
    board = S.groupby("config", sort=False)[["macro_f1_mean", "qwk_mean", "test_weighted_f1", "test_qwk"]].mean().sort_values("qwk_mean", ascending=False)
    keys = [f"Best per task (chosen on validation):"] + [
        f"  {t}: {best.loc[t, 'config']:<22} val macro-F1 {best.loc[t, 'macro_f1_mean']:.3f}+-{best.loc[t, 'macro_f1_std']:.3f}  "
        f"QWK {best.loc[t, 'qwk_mean']:.3f} | gold test wF1 {best.loc[t, 'test_weighted_f1']:.3f}  QWK {best.loc[t, 'test_qwk']:.3f}" for t in TASKS]
    keys += ["Majority-class val macro-F1: " + ", ".join(f"{t} {maj[t]['macro_f1']:.3f}" for t in TASKS),
             "Leaderboard (mean over EXP/ICM/TE):        val F1   val QWK  test wF1  test QWK"]
    keys += [f"  {c:<38} {r.macro_f1_mean:.3f}    {r.qwk_mean:.3f}    {r.test_weighted_f1:.3f}     {r.test_qwk:.3f}" for c, r in board.iterrows()]
    keys += [f"folds {FT_FOLDS}, epochs {EPOCHS}, device {DEVICE}, quick_test {QUICK_TEST}, runtime {(time.time() - t0) / 60:.1f} min"]
    verdict(2, checks, keys, OUT)
    return S


if __name__ == "__main__":
    run_safely(2, main)
