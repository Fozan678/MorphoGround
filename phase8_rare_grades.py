"""MorphoGround - PHASE 8 (v2): rare-grade learning (EXP 1-2, ICM C, TE C).
Compares loss/sampling recipes on the same frozen backbone and the same MorphoGround head, 5 folds x several seeds:
CE -> class-weighted CE -> logit adjustment -> focal -> class-balanced sampling -> CORN -> CORN + supervised contrastive.
v2: CORN replaces the collapsed CORAL (whose 'rare-grade gains' were an artifact of predicting only the extreme grades); an
ordinal-collapse guard is used in model selection and collapsed variants can never be recommended; every variant also gets a
'_tuned' version whose per-class decision offsets are tuned on validation predictions of the OTHER folds (nested, no test
data); per-grade one-vs-rest AUROC shows whether rare embryos are ranked correctly even when argmax rarely picks them.
Frozen tokens are cached once, so the whole study is fast. Predictions are saved in the Phase 5/6 format.
Self-contained: paste into ONE Jupyter cell, or run  python phase8_rare_grades.py"""
from pathlib import Path

# ====================================== USER SETTINGS ========================================
PROJECT_DIR = Path.home() / "Embryo_Updated Project"   # same folder as the earlier phases
PHASE1_DIR  = PROJECT_DIR / "phase1_outputs"
CACHE_DIR   = PROJECT_DIR / "cache"                     # image cache, adapted encoders, Phase 5 checkpoint
IMG_SIZE    = 224
NUM_WORKERS = 4
QUICK_TEST  = False   # True = fast smoke test first

# ---- PHASE 8 settings
BACKBONE    = "auto"   # auto = best ViT encoder of Phase 3 (if run), else of Phase 2
VARIANTS    = ["ce", "ce_weighted", "logit_adjusted", "focal", "ce_balanced", "corn", "corn_supcon"]
TUNE_OFFSETS = True    # v2: also evaluate every variant with validation-tuned per-class decision offsets (nested over folds)
SCREEN_RECALL = 0.5    # v2: rare-grade screening - threshold on P(rare grade) set on validation to catch this share of rare cases
SCREEN_BUDGET = 0.20   # v2: alternative rule - send the 20% most rare-grade-like embryos to review (no silver threshold needed)
SEEDS       = [0, 1, 2]
FOLDS       = [0, 1, 2, 3, 4]
EPOCHS, BATCH, LR, PATIENCE = 40, 128, 3e-4, 10
N_TOKENS, D_MODEL, REL_LAYERS = 6, 256, 2
SUPCON_W, FOCAL_GAMMA = .1, 2.0
RARE        = {"EXP": ["1", "2"], "ICM": ["C"], "TE": ["C"]}   # grades counted as rare
OUT         = PROJECT_DIR / "phase8_outputs"
# ============================================================================================

import json, math, random, sys, time, warnings
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, recall_score
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


def hf_stats(model_id):
    try:
        from transformers import AutoImageProcessor
        p = AutoImageProcessor.from_pretrained(model_id); return tuple(p.image_mean), tuple(p.image_std)
    except Exception: return IMNET  # noqa: BLE001


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


def save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def heat(ax, M, xl, yl, title, fmt="{:d}"):
    M = np.asarray(M, dtype=float) if fmt != "{:d}" else np.asarray(M); ax.imshow(M, cmap="Blues"); ax.set_title(title, fontsize=9)
    ax.set_xticks(range(len(xl)), xl, fontsize=8); ax.set_yticks(range(len(yl)), yl, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "-" if np.isnan(M[i, j]) else fmt.format(M[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > .6 * np.nanmax(M) else "black")


def verdict(phase, checks, key_lines, out_dir, next_step=None):
    """Print + save a PASS / WARN / FAIL checklist. checks = [(status, name, detail), ...]"""
    st = Counter(s for s, _, _ in checks)
    overall = "FAIL" if st["FAIL"] else "PASS WITH WARNINGS" if st["WARN"] else "PASS"
    nxt = {"FAIL": f"Fix the FAIL items before Phase {phase + 1}.",
           "PASS WITH WARNINGS": f"OK to continue to Phase {phase + 1}; WARN items are worth a look.",
           "PASS": f"Ready for Phase {phase + 1}."}[overall]
    if next_step and overall != "FAIL": nxt = next_step
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


# ---------------------------------------------------------------- Phase 5 v2 model code (identical copy)
VIT_NAMES = ["vit_b16", "dinov2_b14", "mae_in1k", "femi", "mae_in1k_dapt", "femi_dapt"]


def choose_backbone():
    """BACKBONE, or 'auto': best ViT encoder of Phase 3 (if run), else of Phase 2, else DINOv2."""
    reg = CACHE_DIR / "phase3_backbones.json"                                   # how Phase 3 loaded each MAE encoder
    if reg.exists(): BACKBONES.update({k: tuple(v) for k, v in json.loads(reg.read_text()).items()})
    for n in ("mae_in1k_dapt", "femi_dapt"):
        if n not in BACKBONES and (CACHE_DIR / n).exists(): BACKBONES[n] = ("hf", str(CACHE_DIR / n))
    if BACKBONE != "auto": return BACKBONE, "set by user"
    f3, f2 = PROJECT_DIR / "phase3_outputs" / "summary.csv", PROJECT_DIR / "phase2_outputs" / "summary.csv"
    if f3.exists():
        s = pd.read_csv(f3, index_col=0); s = s[s.index.isin([n for n in VIT_NAMES if n in BACKBONES])]
        if len(s): return s["score"].idxmax(), "best ViT encoder in Phase 3"
    if f2.exists():
        s = pd.read_csv(f2); s = s[s["model"].isin(VIT_NAMES)]
        if len(s): return s.groupby("model")["sel"].mean().idxmax(), "best ViT backbone in Phase 2"
    return "dinov2_b14", "default"


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


class TokenEncoder(nn.Module):
    """ViT backbone -> (global token (B,D), patch tokens in spatial order (B,N,D))."""
    def __init__(self, name):
        super().__init__(); self.lib, mid = BACKBONES[name]
        if self.lib in ("timm", "timm_tokens"):
            import timm
            from timm.data import resolve_data_config
            arch, _, w = mid.partition("|")                     # "arch|local_weights.pth" = encoder adapted in Phase 3
            self.net = timm.create_model(arch, pretrained=not w, num_classes=0, **({"img_size": IMG_SIZE} if "dinov2" in arch else {}))
            if w: self.net.load_state_dict(torch.load(w, map_location="cpu"))
            cfg = resolve_data_config({}, model=self.net)
            self.mean, self.std, self.size, self.dim, self.npre = tuple(cfg["mean"]), tuple(cfg["std"]), IMG_SIZE, self.net.num_features, self.net.num_prefix_tokens
        else:
            from transformers import ViTMAEModel, logging as hf_logging
            hf_logging.set_verbosity_error()
            self.net = no_mask(ViTMAEModel.from_pretrained(mid))
            (self.mean, self.std), self.size, self.dim = hf_stats(mid), self.net.config.image_size, self.net.config.hidden_size

    def forward(self, x):
        if x.shape[-1] != self.size: x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        if self.lib in ("timm", "timm_tokens"):                  # timm_tokens (MAE): global = mean of patch tokens
            t = self.net.forward_features(x); p = t[:, self.npre:]
            return (p.mean(1) if self.lib == "timm_tokens" else t[:, 0]), p
        p = hf_tokens(self.net, x)                                # v2: no masking, no shuffling, any transformers version
        return p.mean(1), p


class CrossAttn(nn.Module):
    """Queries attend to patch tokens; returns updated queries and attention maps (B, Q, N)."""
    def __init__(self, d, heads=8):
        super().__init__(); self.norm = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q, kv):
        o, w = self.attn(self.norm(q), kv, kv, need_weights=True, average_attn_weights=True)
        q = q + o; return q + self.ff(q), w


class CritAttn(nn.Module):
    """v2: one query per Gardner criterion reads the relational tokens with SINGLE-head attention and a learnable
    temperature, so the criterion -> token weights are exactly the weights the grade is computed from."""
    def __init__(self, d, n_crit):
        super().__init__(); self.q = nn.Parameter(torch.randn(1, n_crit, d) * .02); self.norm = nn.LayerNorm(d)
        self.k, self.v, self.o = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.log_temp = nn.Parameter(torch.zeros(()))           # > 0 sharpens the attention during training

    def forward(self, R):
        R = self.norm(R); q = self.q.expand(len(R), -1, -1)
        W = ((q @ self.k(R).transpose(1, 2)) / math.sqrt(R.shape[-1]) * self.log_temp.exp()).softmax(-1)   # (B, C, K)
        return q + self.o(W @ self.v(R)), W


class CoralHead(nn.Module):
    """v1 CORAL head (one shared weight vector + K-1 thresholds). Kept only to reproduce the v1 collapse (ORDINAL='coral')."""
    def __init__(self, d, K):
        super().__init__(); self.norm, self.w = nn.LayerNorm(d), nn.Linear(d, 1, bias=False)
        self.b = nn.Parameter(torch.linspace(1, -1, K - 1))

    def forward(self, h): return self.w(self.norm(h)) + self.b


def make_head(kind, d, K):
    """ce: K logits | corn: K-1 conditional logits P(y>k | y>k-1), each task with its own weights | coral: v1."""
    if kind == "ce": return nn.Sequential(nn.LayerNorm(d), nn.Dropout(.2), nn.Linear(d, K))
    if kind == "corn": return nn.Sequential(nn.LayerNorm(d), nn.Dropout(.2), nn.Linear(d, K - 1))
    return CoralHead(d, K)


def class_probs(logits, kind):
    """Class probabilities (B, K) from any head. CORN: P(y>k) = prod_j<=k sigmoid; CORAL (v1): cumulative minimum."""
    if kind == "ce": return logits.float().softmax(1)
    s = torch.sigmoid(logits.float())
    s = torch.cumprod(s, 1) if kind == "corn" else torch.cummin(s, 1).values                     # P(y > k), monotone
    c = torch.cat([torch.ones_like(s[:, :1]), s, torch.zeros_like(s[:, :1])], 1)
    return (c[:, :-1] - c[:, 1:]).clamp_min(0)


def ordinal_loss(logits, y, w_cls, kind):
    """CORN (Shi et al. 2023): binary task k = 'y > k', trained only on samples with y >= k (conditional training sets).
    CORAL (v1): every task on every sample. Class weights w_cls[y] as for CE, so the ablation is fair."""
    k = y >= 0
    if not k.any(): return None
    lg, yy = logits[k].float(), y[k]; ks = torch.arange(lg.shape[1], device=y.device)
    tgt = (yy.unsqueeze(1) > ks).float()
    use = (yy.unsqueeze(1) >= ks).float() if kind == "corn" else torch.ones_like(tgt)
    w = w_cls[yy].unsqueeze(1) * use
    return (F.binary_cross_entropy_with_logits(lg, tgt, reduction="none") * w).sum() / w.sum().clamp_min(1e-9)


# ##################################### PHASE 8 #####################################
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
if QUICK_TEST: FOLDS, SEEDS, EPOCHS = [0], [0], 3
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
VARIANT_INFO = {"ce": "cross-entropy", "ce_weighted": "class-weighted CE (effective number)",
                "logit_adjusted": "logit-adjusted CE (Menon et al. 2021)", "focal": f"focal loss (gamma {FOCAL_GAMMA}) + class weights",
                "ce_balanced": "CE + class-balanced sampling (oversample rare grades)",
                "corn": "CORN ordinal + class weights (v2; replaces the collapsed CORAL)",
                "corn_supcon": "CORN + supervised contrastive on criterion embeddings"}


@torch.no_grad()
def token_cache(bb, m, imgs):
    """Frozen [global, patch] tokens of the chosen backbone for every image (float16, cached on disk)."""
    f, kf, keys = CACHE_DIR / f"tokens_{bb}.npy", CACHE_DIR / f"tokens_{bb}_keys.json", m["image_key"].astype(str).tolist()
    if f.exists() and kf.exists() and json.loads(kf.read_text()) == keys: return np.load(f, mmap_mode="r")
    log(f"Caching frozen {bb} tokens for {len(m)} images"); enc = TokenEncoder(bb).to(DEVICE).eval(); arr, i = None, 0
    for x, _ in loader(EmbryoDS(imgs, np.arange(len(m)), None, enc.mean, enc.std), 64, False):
        with autocast(): g, P = enc(x.to(DEVICE))
        t = torch.cat([g.unsqueeze(1), P], 1).half().cpu().numpy()
        if arr is None: arr = np.lib.format.open_memmap(f, mode="w+", dtype=np.float16, shape=(len(m),) + t.shape[1:])
        arr[i:i + len(t)] = t; i += len(t)
    arr.flush(); del arr, enc; kf.write_text(json.dumps(keys)); torch.cuda.empty_cache()
    return np.load(f, mmap_mode="r")


class RareHead(nn.Module):
    """MorphoGround head (morphology tokens + relational transformer + one query per criterion, v2 single-head criterion
    attention) on frozen tokens. kind = 'ce' (softmax) or 'corn' (ordinal)."""
    def __init__(self, D, n_cls, kind):
        super().__init__(); d = D_MODEL; self.head_type = kind
        self.proj = nn.Sequential(nn.Linear(D, d), nn.LayerNorm(d))
        self.queries = nn.Parameter(torch.randn(1, N_TOKENS, d) * .02)
        self.token_attn = nn.ModuleList([CrossAttn(d) for _ in range(2)])
        self.rel = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, .1, batch_first=True, norm_first=True), REL_LAYERS)
        self.crit = CritAttn(d, len(TASKS))
        self.heads = nn.ModuleList([make_head(kind, d, k) for k in n_cls])

    def forward(self, T):
        g, P = self.proj(T[:, 0]), self.proj(T[:, 1:]); q = self.queries.expand(len(T), -1, -1)
        for layer in self.token_attn: q, _ = layer(q, P)
        R = self.rel(torch.cat([g.unsqueeze(1), q], 1)); c, _ = self.crit(R[:, 1:])
        return [h(c[:, i]) for i, h in enumerate(self.heads)], c


def supcon(z, y, temp=.1):
    """Supervised contrastive loss (Khosla et al. 2020): same grade = positive."""
    k = y >= 0; z, y = F.normalize(z[k].float(), dim=1), y[k]
    if len(y) < 3: return z.new_zeros(())
    eye = torch.eye(len(y), device=z.device); S = z @ z.T / temp - eye * 1e9
    pos = (y[:, None] == y[None]).float() - eye; n = pos.sum(1)
    if not (n > 0).any(): return z.new_zeros(())
    logp = S - torch.logsumexp(S, 1, keepdim=True)
    return -((logp * pos).sum(1)[n > 0] / n[n > 0]).mean()


def crit_loss(variant, lg, y, w, prior):
    if variant.startswith("corn"): return ordinal_loss(lg, y, w, "corn")
    k = y >= 0
    if not k.any(): return None
    lg, yy = lg[k].float(), y[k]
    if variant in ("ce", "ce_balanced"): return F.cross_entropy(lg, yy)
    if variant == "ce_weighted": return F.cross_entropy(lg, yy, weight=w)
    if variant == "logit_adjusted": return F.cross_entropy(lg + torch.log(prior + 1e-12), yy)
    logp = F.log_softmax(lg, 1).gather(1, yy[:, None]).squeeze(1)                     # focal
    return (-(w[yy] / w.mean()) * (1 - logp.exp()) ** FOCAL_GAMMA * logp).mean()


@torch.no_grad()
def predict(model, T, idx):
    model.eval(); P, Z = [[] for _ in TASKS], []
    for i in range(0, len(idx), 512):
        with autocast(): logits, c = model(T[torch.as_tensor(idx[i:i + 512], device=T.device)].float().to(DEVICE))
        for t, lg in enumerate(logits): P[t].append(class_probs(lg, model.head_type).cpu().numpy())
        Z.append(c.float().cpu().numpy())
    return [np.concatenate(p) for p in P], np.concatenate(Z)


def collapse_report(y, P, min_prev=.05):
    """Ordinal-collapse guard (as Phase 5)."""
    k = (y >= 0) & np.isfinite(P).all(1); y, p, K = y[k], P[k].argmax(1), P.shape[1]
    if not len(y): return {"collapsed": False, "tv": 0.0, "dead": []}
    ty, tp = np.bincount(y, minlength=K) / len(y), np.bincount(p, minlength=K) / len(y)
    dead = [c for c in range(K) if ty[c] >= min_prev and (p[y == c] == c).mean() < .05]
    tv = float(.5 * np.abs(ty - tp).sum())
    return {"collapsed": bool(dead) or tv > .5, "tv": tv, "dead": dead}


def train_run(variant, seed, fold, T, Y, n_cls, cw, m):
    torch.manual_seed(seed); g = torch.Generator(device="cpu").manual_seed(seed)
    tr, va, te = np.where((m["fold"] >= 0) & (m["fold"] != fold))[0], np.where(m["fold"] == fold)[0], np.where(m["fold"] == -1)[0]
    model = RareHead(T.shape[-1], n_cls, "corn" if variant.startswith("corn") else "ce").to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=.05); nb = math.ceil(len(tr) / BATCH)
    sched, scaler = cosine_sched(opt, EPOCHS * nb, nb), make_scaler()
    W = [torch.tensor(w, dtype=torch.float32, device=DEVICE) for w in cw]
    prior = [torch.tensor(np.bincount(Y[tr, t][Y[tr, t] >= 0], minlength=K) / max((Y[tr, t] >= 0).sum(), 1), dtype=torch.float32, device=DEVICE)
             for t, K in enumerate(n_cls)]
    sw = None
    if variant.endswith("balanced"):                                                  # oversample images carrying a rare grade
        inv = [1 / np.maximum(np.bincount(Y[tr, t][Y[tr, t] >= 0], minlength=K), 1) for t, K in enumerate(n_cls)]
        sw = torch.tensor(np.max([np.where(Y[tr, t] >= 0, inv[t][np.clip(Y[tr, t], 0, None)], 0) for t in range(len(n_cls))], 0))
    Yt, tr_t = torch.as_tensor(Y, device=DEVICE), torch.as_tensor(tr, device=T.device)
    best, bad, state = -np.inf, 0, None
    for ep in range(EPOCHS):
        model.train()
        order = torch.multinomial(sw, len(tr), replacement=True, generator=g) if sw is not None else torch.randperm(len(tr), generator=g)
        for b in range(0, len(tr), BATCH):
            idx = tr_t[order[b:b + BATCH].to(T.device)]; x, y = T[idx].float().to(DEVICE), Yt[idx.to(DEVICE)]
            keep = torch.rand(x.shape[:2], device=DEVICE) > .1; keep[:, 0] = True       # token dropout + noise (feature augmentation)
            x = x * keep.unsqueeze(-1) + .05 * torch.randn_like(x)
            with autocast():
                logits, c = model(x)
                parts = [p for p in (crit_loss(variant, lg, y[:, t], W[t], prior[t]) for t, lg in enumerate(logits)) if p is not None]
                if "supcon" in variant: parts += [SUPCON_W * supcon(c[:, t], y[:, t]) for t in range(len(TASKS))]
                if not parts: continue
                loss = sum(parts)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        Pv = predict(model, T, va)[0]; s = sel_score(Y[va], Pv)
        s -= 1.0 if any(collapse_report(Y[va, t], Pv[t])["collapsed"] for t in range(len(TASKS))) else 0.0   # v2 collapse guard
        if s > best: best, bad, state = s, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else: bad += 1
        if bad >= PATIENCE: break
    if state is not None: model.load_state_dict(state)
    return predict(model, T, va)[0], *predict(model, T, te)


def tune_offsets(P, y, grid=np.linspace(-3, 3, 25), rounds=2):
    """v2: per-class log-probability offsets that maximise macro-F1 on validation predictions (coordinate search).
    Moves the decision boundary towards rare grades without retraining."""
    k = (y >= 0) & np.isfinite(P).all(1); L, y, K = np.log(np.clip(P[k], 1e-9, 1)), y[k], P.shape[1]; b = np.zeros(K)
    if len(np.unique(y)) < 2: return b
    labs = np.unique(y); f = lambda bb: f1_score(y, (L + bb).argmax(1), average="macro", labels=labs, zero_division=0)  # noqa: E731
    best = f(b)
    for _ in range(rounds):
        for c in range(K):
            for v in grid:
                bb = b.copy(); bb[c] = v; s = f(bb)
                if s > best + 1e-9: best, b = s, bb
    return b


def apply_offsets(P, b):
    L = np.log(np.clip(P, 1e-9, 1)) + b; e = np.exp(L - L.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def tuned_version(oof, tp, tfold, Y, m):
    """Nested: fold f's OOF predictions use offsets tuned on the OTHER folds' OOF; test uses offsets tuned on all OOF."""
    f_ = m["fold"].to_numpy(); o2, t2, tf2, offs = [], [], [], []
    for t in range(len(TASKS)):
        o = np.full_like(oof[t], np.nan)
        for f in FOLDS:
            other = np.isin(f_, FOLDS) & (f_ != f)
            o[f_ == f] = apply_offsets(oof[t][f_ == f], tune_offsets(oof[t][other], Y[other, t]))
        b = tune_offsets(oof[t][np.isin(f_, FOLDS)], Y[np.isin(f_, FOLDS), t]); offs.append(b.round(2).tolist())
        o2.append(o); t2.append(apply_offsets(tp[t], b)); tf2.append(np.stack([apply_offsets(x, b) for x in tfold[t]]))
    return o2, t2, tf2, offs


def rare_recall(y, P, t):
    idx = [CLASS_NAMES[t].index(c) for c in RARE[t]]; mt = metrics(y, P)
    return [mt.get(f"recall_{i}", np.nan) for i in idx]


def rare_auroc(y, P, t):
    """One-vs-rest AUROC of each rare grade: can the model RANK rare embryos first, even when argmax rarely picks them?"""
    k = y >= 0; out = {}
    for c in RARE[t]:
        i = CLASS_NAMES[t].index(c); pos = (y[k] == i)
        out[f"auroc_{t}-{c}"] = float(roc_auc_score(pos, P[k][:, i])) if 0 < pos.sum() < len(pos) else np.nan
    return out


def screening(yv, Pv, yt, Pt, t, target):
    """v2 rare-grade screening: flag an embryo for review when P(rare grade) >= a threshold chosen on VALIDATION predictions
    so that `target` of the validation rare cases are caught; report what that threshold does on the gold test."""
    out = {}
    for c in RARE[t]:
        i = CLASS_NAMES[t].index(c); kv = (yv >= 0) & np.isfinite(Pv).all(1); posv = yv[kv] == i
        if posv.sum() < 3: continue
        thr = float(np.quantile(Pv[kv][posv, i], 1 - target)); kt = yt >= 0; flag, post = Pt[kt, i] >= thr, yt[kt] == i
        fb = Pt[kt, i] >= np.quantile(Pt[kt, i], 1 - SCREEN_BUDGET)                   # fixed review budget (label-free rule)
        out[f"{t}-{c}"] = {"threshold": thr, "val_positives": int(posv.sum()), "test_positives": int(post.sum()),
                           "test_recall": float(flag[post].mean()) if post.any() else np.nan,
                           "test_precision": float(post[flag].mean()) if flag.any() else np.nan, "flag_rate": float(flag.mean()),
                           "budget_recall": float(fb[post].mean()) if post.any() else np.nan,
                           "budget_precision": float(post[fb].mean()) if fb.any() else np.nan,
                           "median_p_val_pos": float(np.median(Pv[kv][posv, i])),
                           "median_p_test_pos": float(np.median(Pt[kt, i][post])) if post.any() else np.nan}
    return out


def main():
    t0 = time.time(); seed_all(); (OUT / "preds").mkdir(parents=True, exist_ok=True)
    m = load_manifest(); Y, n_cls = targets(m); imgs = image_cache(m); cw = phase1_class_weights()
    bb, bb_src = choose_backbone(); log(f"Backbone (frozen): {bb} ({bb_src})"); Tn = token_cache(bb, m, imgs)
    try: T = torch.from_numpy(np.asarray(Tn)).to(DEVICE)
    except RuntimeError: T = torch.from_numpy(np.asarray(Tn)); log("tokens kept in CPU memory (GPU too small)")
    for old in ("coral", "coral_supcon", "coral_supcon_balanced"):                    # v1 files would confuse Phase 9
        f = OUT / "preds" / f"{old}.npz"
        if f.exists(): f.rename(f.with_name(f"v1_{old}.npz.bak"))
    te = np.where(m["fold"] == -1)[0]; cv = np.isin(m["fold"].to_numpy(), FOLDS)
    rows, emb, failed, test_probs, collapse, offsets, oofs = [], {}, {}, {}, {}, {}, {}

    def record(v, oof_seeds, tp, tfold):
        oof_mean = [np.nanmean([o[t] for o in oof_seeds], 0) for t in range(len(TASKS))]
        collapse[v] = {T_: collapse_report(Y[cv, t], oof_mean[t][cv])["collapsed"] for t, T_ in enumerate(TASKS)}
        for t, T_ in enumerate(TASKS):
            per_seed = [metrics(Y[cv, t], o[t][cv]) for o in oof_seeds]
            rr_seed = [np.nanmean(rare_recall(Y[cv, t], o[t][cv], T_)) for o in oof_seeds]
            mt_te, rr_te = metrics(Y[te, t], tp[t]), rare_recall(Y[te, t], tp[t], T_)
            rows.append({"variant": v, "task": T_, "val_macro_f1": np.mean([d["macro_f1"] for d in per_seed]),
                         "val_macro_f1_sd": np.std([d["macro_f1"] for d in per_seed]), "val_qwk": np.mean([d["qwk"] for d in per_seed]),
                         "val_rare_recall": np.mean(rr_seed), "val_rare_recall_sd": np.std(rr_seed),
                         "test_macro_f1": mt_te["macro_f1"], "test_qwk": mt_te["qwk"], "test_rare_recall": np.nanmean(rr_te),
                         **{f"test_recall_{CLASS_NAMES[T_][i]}": mt_te.get(f"recall_{i}", np.nan) for i in range(n_cls[t])},
                         **rare_auroc(Y[te, t], tp[t], T_), "collapsed": collapse[v][T_]})
        test_probs[v] = tp
        np.savez_compressed(OUT / "preds" / f"{v}.npz", keys=m["image_key"].astype(str).to_numpy(), test_idx=te, backbone=bb, version="v2",
                            **{f"oof_{k}": oof_mean[t] for t, k in enumerate(TASKS)},
                            **{f"test_{k}": tp[t] for t, k in enumerate(TASKS)}, **{f"test_folds_{k}": tfold[t] for t, k in enumerate(TASKS)})
        return oof_mean

    for v in VARIANTS:
        log(f"== {v}: {VARIANT_INFO[v]}")
        oof = {s: [np.full((len(m), K), np.nan) for K in n_cls] for s in SEEDS}; tf = {f: [[] for _ in TASKS] for f in FOLDS}
        try:
            for s in SEEDS:
                for f in FOLDS:
                    va = np.where(m["fold"] == f)[0]; Pva, Pte, Z = train_run(v, s, f, T, Y, n_cls, cw[f], m)
                    for t in range(len(TASKS)): oof[s][t][va] = Pva[t]; tf[f][t].append(Pte[t])
                    if s == SEEDS[0] and f == FOLDS[0]: emb[v] = Z
        except Exception as e:  # noqa: BLE001
            failed[v] = f"{type(e).__name__}: {str(e)[:100]}"; log(f"FAILED {v}: {failed[v]}"); continue
        tfold = [np.stack([np.mean(tf[f][t], 0) for f in FOLDS]) for t in range(len(TASKS))]; tp = [x.mean(0) for x in tfold]
        oof_mean = oofs[v] = record(v, list(oof.values()), tp, tfold)
        if TUNE_OFFSETS:                                                               # v2: validation-tuned decision offsets
            o2, t2, tf2, offsets[v] = tuned_version(oof_mean, tp, tfold, Y, m); oofs[f"{v}_tuned"] = record(f"{v}_tuned", [o2], t2, tf2)
    if not rows: raise RuntimeError(f"No variant finished: {failed}")
    R = pd.DataFrame(rows)
    agg = ["val_macro_f1", "val_qwk", "val_rare_recall", "val_rare_recall_sd", "val_macro_f1_sd", "test_macro_f1", "test_qwk", "test_rare_recall"]
    S = R.groupby("variant", sort=False)[agg].mean(); S["collapsed"] = R.groupby("variant", sort=False)["collapsed"].any()
    S["selection"] = (S["val_macro_f1"] + S["val_rare_recall"]) / 2
    healthy = S[~S["collapsed"]]; rec = (healthy if len(healthy) else S)["selection"].idxmax()
    ords = {}
    for v, Z in emb.items():                                                            # ordinal structure of criterion embeddings
        r = []
        for t in range(len(TASKS)):
            ok = np.where(Y[te, t] >= 0)[0]; a, b = np.random.default_rng(0).choice(ok, (2, 4000))
            za, zb = Z[a, t] / np.linalg.norm(Z[a, t], axis=1, keepdims=True), Z[b, t] / np.linalg.norm(Z[b, t], axis=1, keepdims=True)
            r.append(spearmanr(1 - (za * zb).sum(1), np.abs(Y[te[a], t] - Y[te[b], t]))[0])
        ords[v] = float(np.nanmean(r))
    S["ordinal_spearman"] = pd.Series(ords); R.to_csv(OUT / "results_per_task.csv", index=False); S.to_csv(OUT / "summary.csv")
    scr = {k: v for t, T_ in enumerate(TASKS) for k, v in screening(Y[cv, t], oofs[rec][t][cv], Y[te, t], test_probs[rec][t], T_, SCREEN_RECALL).items()}
    pd.DataFrame(scr).T.to_csv(OUT / "rare_screening.csv")
    (OUT / "recommended.json").write_text(json.dumps({"variant": rec, "backbone": bb, "file": f"phase8_outputs/preds/{rec}.npz",
                                                      "offsets": offsets.get(rec.replace("_tuned", ""), None) if rec.endswith("_tuned") else None}))

    fig, axes = plt.subplots(1, 2, figsize=(16, .45 * len(S) + 2.5))                  # figures
    cls = [c for c in R.columns if c.startswith("test_recall_")]
    M = np.vstack([R[R["variant"] == v].set_index("task")[cls].reindex(list(TASKS)).to_numpy().ravel() for v in S.index])
    lab = [f"{t}-{c[12:]}" for t in TASKS for c in cls]; keep = ~np.all(np.isnan(M), 0)
    heat(axes[0], M[:, keep], [l for l, k in zip(lab, keep) if k], list(S.index), "Gold-test recall per grade (rare: EXP-1/2, ICM-C, TE-C)", "{:.2f}")
    axes[1].errorbar(S["val_macro_f1"], S["val_rare_recall"], xerr=S["val_macro_f1_sd"], yerr=S["val_rare_recall_sd"], fmt="o")
    for v, r in S.iterrows(): axes[1].annotate(v + (" (collapsed)" if r["collapsed"] else ""), (r["val_macro_f1"], r["val_rare_recall"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    axes[1].set(xlabel="validation macro-F1", ylabel="validation rare-grade recall", title="Trade-off (mean +- sd over seeds)")
    save(fig, OUT / "rare_grade_results.png")
    base = rec.replace("_tuned", "")
    pair = [v for v in ("ce", base) if v in emb]
    if len(pair) == 2 and pair[0] != pair[1]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.3)); ok = np.where(Y[te, 1] >= 0)[0]
        for ax, v in zip(axes, pair):
            Zt = TSNE(2, init="pca", perplexity=min(30, len(ok) - 1), random_state=0).fit_transform(emb[v][ok, 1])
            for c, col in zip(range(3), ("#4C72B0", "#55A868", "#C44E52")):
                k = Y[te[ok], 1] == c; ax.scatter(Zt[k, 0], Zt[k, 1], s=8 if c < 2 else 30, c=col, label=f"ICM {CLASS_NAMES['ICM'][c]}")
            ax.set_title(f"ICM embedding (gold test): {v}", fontsize=9); ax.axis("off"); ax.legend(fontsize=7)
        save(fig, OUT / "icm_embeddings.png")

    ce = S.loc["ce"] if "ce" in S.index else None; r = S.loc[rec]; Rr = R[R["variant"] == rec].set_index("task")
    icmc, icmc_auc = Rr.loc["ICM", "test_recall_C"], Rr.loc["ICM", "auroc_ICM-C"]
    col = [v for v in S.index if S.loc[v, "collapsed"]]
    checks = [("PASS" if phase_status("phase1_outputs", "QC_VERDICT.txt") else "WARN", "Phase 1 folds and class weights", "loaded"),
              ("INFO", "Frozen backbone", f"{bb} ({bb_src}); tokens {tuple(Tn.shape)}"),
              ("PASS" if not failed else "WARN", "All loss variants trained", "yes" if not failed else str(failed)[:120]),
              ("PASS" if not col else "WARN", "No variant collapsed (validation)", "none" if not col else ", ".join(col) + " (excluded from the recommendation)")]
    if ce is not None:
        checks += [("PASS" if r["val_rare_recall"] > ce["val_rare_recall"] else "WARN", "Rare-grade recall improves over plain CE (val)",
                    f"ce {ce['val_rare_recall']:.3f} -> {rec} {r['val_rare_recall']:.3f} (test {ce['test_rare_recall']:.3f} -> {r['test_rare_recall']:.3f})"),
                   ("PASS" if r["val_macro_f1"] >= ce["val_macro_f1"] - .02 else "WARN", "Overall macro-F1 is not sacrificed (>= CE - 0.02)",
                    f"ce {ce['val_macro_f1']:.3f} -> {rec} {r['val_macro_f1']:.3f}")]
    checks += [("PASS" if icmc >= .2 else "WARN", "ICM grade C is recognised on the gold test (recall >= 0.20)",
                f"recall {icmc:.2f}, AUROC {icmc_auc:.2f} ({rec})" if pd.notna(icmc) else "no ICM-C embryo in the gold test"),
               ("PASS" if (S.loc[[v for v in S.index if not v.endswith("_tuned")], "val_rare_recall_sd"] <= .05).all() else "WARN",
                "Stable across seeds (sd of rare recall <= 0.05)",
                " | ".join(f"{k} {v:.3f}" for k, v in S["val_rare_recall_sd"].items() if not k.endswith("_tuned"))),
               ("PASS" if "ICM-C" in scr and scr["ICM-C"]["test_recall"] >= SCREEN_RECALL - .15 else "WARN",
                f"ICM-C screening keeps its validation recall on the gold test (target {SCREEN_RECALL:.0%})",
                (f"test recall {scr['ICM-C']['test_recall']:.2f}, precision {scr['ICM-C']['test_precision']:.2f}, flags "
                 f"{scr['ICM-C']['flag_rate']:.0%}; median P(C) of true C: validation {scr['ICM-C']['median_p_val_pos']:.2f} vs gold "
                 f"{scr['ICM-C']['median_p_test_pos']:.2f}") if "ICM-C" in scr else "too few ICM-C cases"),
               ("INFO", f"ICM-C with a {SCREEN_BUDGET:.0%} review budget (most C-like embryos flagged)",
                f"recall {scr['ICM-C']['budget_recall']:.2f}, precision {scr['ICM-C']['budget_precision']:.2f}" if "ICM-C" in scr else "n/a"),
               ("INFO", "Ordinal structure of embeddings (Spearman)", " | ".join(f"{k} {v:+.2f}" for k, v in S["ordinal_spearman"].dropna().items())),
               ("INFO", "Recommended variant (max of val macro-F1 + rare recall, healthy only)", f"{rec} -> phase8_outputs/preds/{rec}.npz")]
    keys = ["Variant                    val F1         val QWK  val rare-recall  test F1  test QWK  test rare-recall  ordinal rho"]
    keys += [f"  {v:<25} {q.val_macro_f1:.3f}+-{q.val_macro_f1_sd:.3f}  {q.val_qwk:.3f}    {q.val_rare_recall:.3f}+-{q.val_rare_recall_sd:.3f}      "
             f"{q.test_macro_f1:.3f}    {q.test_qwk:.3f}     {q.test_rare_recall:.3f}             "
             + ("-" if pd.isna(q.ordinal_spearman) else f"{q.ordinal_spearman:+.2f}") + (" COLLAPSED" if q.collapsed else "") for v, q in S.iterrows()]
    keys.append("Gold-test rare grades (" + rec + "): " + ", ".join(
        f"{t}-{c} recall {Rr.loc[t, f'test_recall_{c}']:.2f} / AUROC {Rr.loc[t, f'auroc_{t}-{c}']:.2f}" for t in TASKS for c in RARE[t]))
    keys.append(f"Rare-grade screening ({rec}, threshold for {SCREEN_RECALL:.0%} validation recall): " + ", ".join(
        f"{k} recall {v['test_recall']:.2f} / precision {v['test_precision']:.2f} / flagged {v['flag_rate']:.0%}" for k, v in scr.items()))
    keys.append(f"Rare-grade screening with a {SCREEN_BUDGET:.0%} review budget: " + ", ".join(
        f"{k} recall {v['budget_recall']:.2f} / precision {v['budget_precision']:.2f}" for k, v in scr.items()))
    keys.append(f"seeds {SEEDS}, folds {FOLDS}, epochs {EPOCHS}, tuned offsets {TUNE_OFFSETS}, device {DEVICE}, quick_test {QUICK_TEST}, "
                f"runtime {(time.time() - t0) / 60:.1f} min")
    verdict(8, checks, keys, OUT)
    return S


if __name__ == "__main__":
    run_safely(8, main)
