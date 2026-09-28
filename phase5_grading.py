"""MorphoGround - PHASE 5 (v2): hierarchical, relational and ordinal Gardner grading (+ label-free morphology grounding).
Ablation ladder on the Phase 1 folds (same backbone, same training recipe, same class weights for every variant):
  global_ce -> global_ordinal -> tokens_ordinal -> morphoground
v2 fixes
  * CORAL threshold collapse: the ordinal heads are CORN (conditional ordinal binary tasks, each with its own weights), and an
    ordinal-collapse guard (a common grade never predicted, or predicted-grade distribution far from the true one) is checked
    every epoch, used in model selection and reported as a FAIL item.
  * Uniform criterion attention / identical evidence maps: single-head criterion->token attention with a learnable temperature
    (the weights shown are exactly the weights used), plus two grounding losses in MorphoGround: sharp criterion attention
    (entropy) and criterion-specific evidence (low overlap between the EXP / ICM / TE evidence maps). Optional weak mask
    supervision when compartment masks exist for some training images (MASKS_DIR).
  * Hugging Face MAE encoders load with transformers 4.x and 5.x; fold-0 checkpoints of MorphoGround and of the unconstrained
    token model are saved with their architecture for Phase 4.
Saves out-of-fold / gold-test probabilities, per-fold test probabilities and criterion evidence maps for Phases 4, 6, 7 and 9.
Self-contained: paste into ONE Jupyter cell, or run  python phase5_grading.py"""
from pathlib import Path

# ====================================== USER SETTINGS ========================================
PROJECT_DIR = Path.home() / "Embryo_Updated Project"   # same folder as Phases 1-3
PHASE1_DIR  = PROJECT_DIR / "phase1_outputs"
CACHE_DIR   = PROJECT_DIR / "cache"                     # image cache + adapted encoders from earlier phases
IMG_SIZE, NUM_WORKERS = 224, 4
QUICK_TEST  = False   # True = fast smoke test first

# ---- PHASE 5 settings
BACKBONE    = "auto"   # auto = best ViT encoder of Phase 3 (if run), else of Phase 2; or "dinov2_b14", "vit_b16", "mae_in1k_dapt", "femi"
VARIANTS    = ["global_ce", "global_ordinal", "tokens_ordinal", "morphoground"]   # ablation ladder (research question 3)
ORDINAL     = "corn"   # corn = CORN heads (v2, no threshold collapse) | coral = v1 CORAL head (kept only to reproduce the bug)
N_TOKENS, D_MODEL, REL_LAYERS = 6, 256, 2                                   # morphology tokens, width, relational layers
CRIT_GLOBAL = False    # False = criterion queries read only the morphology tokens -> every criterion's evidence is token-based
GROUND_W    = {"diversity": .1, "compact": .05, "consistency": .1,          # token grounding (v1)
               "crit_entropy": .05, "crit_distinct": .2,                    # v2: sharp + criterion-specific evidence
               "mask": 1.0}                                                 # v2: only used when MASKS_DIR has masks
MASKS_DIR   = None     # optional compartment masks for some TRAINING images (same format as Phase 4), e.g. r"/home/<you>/masks"
MASK_LABELS = {"zona": 1, "TE": 2, "ICM": 3, "blastocoel": 4}               # pixel values in the mask images (0 = background)
EXPECTED    = {"EXP": ["blastocoel", "zona"], "ICM": ["ICM"], "TE": ["TE"]}   # compartment each criterion should rely on
MASK_BATCH  = 8
SAVE_CKPT   = ["morphoground", "tokens_ordinal"]                            # fold-0 checkpoints for Phase 4 (grounded vs not)
FOLDS       = [0, 1, 2, 3, 4]
EPOCHS, BATCH, PATIENCE = 25, 32, 7
LR_BACKBONE, LR_HEAD = 2e-5, 3e-4
OUT         = PROJECT_DIR / "phase5_outputs"
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


# ##################################### PHASE 5 #####################################
if QUICK_TEST: FOLDS, EPOCHS = [0], 2
ORD_NAME = {"corn": "CORN", "coral": "CORAL"}[ORDINAL]
VARIANT_INFO = {"global_ce": "global feature + softmax CE (Phase 2 recipe)",
                "global_ordinal": f"global feature + {ORD_NAME} ordinal heads",
                "tokens_ordinal": f"morphology tokens + relational transformer + {ORD_NAME} (unconstrained tokens)",
                "morphoground": f"tokens + relational + {ORD_NAME} + grounding losses (token diversity, compactness, consistency; "
                                "sharp, criterion-specific evidence" + ("; mask supervision" if MASKS_DIR else "") + ")"}
VIT_NAMES = ["vit_b16", "dinov2_b14", "mae_in1k", "femi", "mae_in1k_dapt", "femi_dapt"]
ARCH = {"n_tokens": N_TOKENS, "d_model": D_MODEL, "rel_layers": REL_LAYERS, "crit_global": CRIT_GLOBAL, "ordinal": ORDINAL}


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


class MorphoGrader(nn.Module):
    """global_*: global token -> heads.  tokens_ordinal / morphoground: K morphology query tokens read the patches, a
    relational transformer mixes [global, tokens], and one criterion query per Gardner criterion reads the tokens."""
    def __init__(self, enc, n_cls, variant, arch=None):
        super().__init__(); a = arch or ARCH; d = a["d_model"]
        self.enc, self.variant, self.arch = enc, variant, a
        self.head_type = "ce" if variant == "global_ce" else a["ordinal"]
        self.tokens, self.crit_global = variant in ("tokens_ordinal", "morphoground"), a["crit_global"]
        self.proj = nn.Sequential(nn.Linear(enc.dim, d), nn.LayerNorm(d))
        if self.tokens:
            self.queries = nn.Parameter(torch.randn(1, a["n_tokens"], d) * .02)
            self.token_attn = nn.ModuleList([CrossAttn(d) for _ in range(2)])                  # morphology query transformer
            self.rel = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, .1, batch_first=True, norm_first=True), a["rel_layers"])
            self.crit = CritAttn(d, len(TASKS))
        self.heads = nn.ModuleList([make_head(self.head_type, d, k) for k in n_cls])

    def forward(self, x):
        g, P = self.enc(x); g, P = self.proj(g), self.proj(P)
        if not self.tokens: return [h(g) for h in self.heads], None, None
        q = self.queries.expand(len(x), -1, -1)
        for layer in self.token_attn: q, A = layer(q, P)                                         # A: token -> patch attention
        R = self.rel(torch.cat([g.unsqueeze(1), q], 1))                                            # relational reasoning
        c, Wc = self.crit(R if self.crit_global else R[:, 1:])
        if not self.crit_global: Wc = F.pad(Wc, (1, 0))          # keep the (B, 3, 1+K) layout; column 0 = global token (weight 0)
        return [h(c[:, i]) for i, h in enumerate(self.heads)], A, Wc


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


def crit_loss(model, lg, y, w):
    if model.head_type == "ce":
        return F.cross_entropy(lg.float(), y, weight=w, ignore_index=-1, label_smoothing=.05) if (y >= 0).any() else None
    return ordinal_loss(lg, y, w, model.head_type)


def evidence(Wc, A):
    """Criterion evidence maps (B, 3, N): criterion -> token weights (renormalised over tokens) x token -> patch attention."""
    T_ = Wc[:, :, 1:].float(); T_ = T_ / T_.sum(2, keepdim=True).clamp_min(1e-9)
    return torch.einsum("bck,bkn->bcn", T_, A.float()), T_


def grounding_terms(A, Wc, A2=None):
    """Label-free grounding. v1: diversity (tokens look at different places), compactness (each token map is tight),
    consistency (same maps under a photometric change). v2: crit_entropy (each criterion reads few tokens; 1 = uniform)
    and crit_distinct (mean cosine between the EXP / ICM / TE evidence maps; 1 = identical maps)."""
    A = A.float(); K, N = A.shape[1], A.shape[2]; g = math.isqrt(N)
    An = F.normalize(A, dim=2); G = An @ An.transpose(1, 2)
    div = (G.sum((1, 2)) - G.diagonal(dim1=1, dim2=2).sum(1)).mean() / (K * (K - 1))
    lin = torch.linspace(0, 1, g, device=A.device); yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    C = torch.stack([yy.ravel(), xx.ravel()], 1); Ap = A / A.sum(2, keepdim=True); mu = Ap @ C
    comp = (Ap * ((C[None, None] - mu[:, :, None]) ** 2).sum(-1)).sum(-1).mean()
    cons = (1 - F.cosine_similarity(A, A2.float(), dim=2)).mean() if A2 is not None else A.new_zeros(())
    E, T_ = evidence(Wc, A)
    ent = -(T_ * T_.clamp_min(1e-9).log()).sum(2).mean() / math.log(K)
    En = F.normalize(E, dim=2); S = En @ En.transpose(1, 2); c = S.shape[1]
    distinct = (S.sum((1, 2)) - S.diagonal(dim1=1, dim2=2).sum(1)).mean() / (c * (c - 1))
    return div, comp, cons, ent, distinct


def photometric(x):
    a = torch.empty(len(x), 1, 1, 1, device=x.device).uniform_(.8, 1.2); b = torch.empty_like(a).uniform_(-.2, .2)
    mu = x.mean((1, 2, 3), keepdim=True); return (x - mu) * a + mu + b + .03 * torch.randn_like(x)


def load_train_masks(m, size):
    """Optional compartment masks of TRAINING images (CV pool only; never the gold test) -> {manifest row: (size, size)}."""
    if not MASKS_DIR: return {}
    files = {Path(f).stem.lower(): f for f in Path(MASKS_DIR).rglob("*") if Path(f).suffix.lower() in (".png", ".tif", ".tiff", ".bmp")}
    out = {}
    for i, (k, f) in enumerate(zip(m["image_key"].astype(str).str.lower(), m["fold"])):
        if f >= 0 and k in files:
            with Image.open(files[k]) as im: out[i] = np.asarray(im.convert("L").resize((size, size), Image.NEAREST)).astype(np.int16)
    log(f"Mask supervision: {len(out)} training images with compartment masks in {MASKS_DIR}")
    return out


def mask_targets(masks, g):
    """(n, H, W) label masks -> (n, 3, g*g) target evidence distribution per criterion (its EXPECTED compartments)."""
    n, H, _ = masks.shape; s = H // g; T_ = []
    for t in TASKS:
        hit = np.isin(masks, [MASK_LABELS[c] for c in EXPECTED[t]]).reshape(n, g, s, g, s).mean((2, 4)).reshape(n, g * g)
        T_.append(hit / np.maximum(hit.sum(1, keepdims=True), 1e-9))
    return np.stack(T_, 1).astype(np.float32)


def collapse_report(y, P, min_prev=.05):
    """Ordinal-collapse guard (the v1 CORAL bug): a grade with >= 5% prevalence that is (almost) never predicted correctly,
    or a predicted-grade distribution far from the true one (total variation > 0.5)."""
    k = y >= 0; y, p, K = y[k], P[k].argmax(1), P.shape[1]
    if not len(y): return {"collapsed": False, "tv": 0.0, "dead": []}
    ty, tp = np.bincount(y, minlength=K) / len(y), np.bincount(p, minlength=K) / len(y)
    dead = [c for c in range(K) if ty[c] >= min_prev and (p[y == c] == c).mean() < .05]
    tv = float(.5 * np.abs(ty - tp).sum())
    return {"collapsed": bool(dead) or tv > .5, "tv": tv, "dead": dead}


def collapsed_any(Yv, probs): return any(collapse_report(Yv[:, t], probs[t])["collapsed"] for t in range(len(TASKS)))


@torch.no_grad()
def predict(model, dl, maps=False):
    model.eval(); P, E, W, A_ = [[] for _ in TASKS], [], [], []
    for x, _ in dl:
        with autocast(): logits, A, Wc = model(x.to(DEVICE, non_blocking=True))
        for t, lg in enumerate(logits): P[t].append(class_probs(lg, model.head_type).cpu().numpy())
        if maps and A is not None:
            E.append(evidence(Wc, A)[0].cpu().numpy()); W.append(Wc.float().cpu().numpy()); A_.append(A.float().cpu().numpy())
    P = [np.concatenate(p) for p in P]
    return (P, np.concatenate(E), np.concatenate(W), np.concatenate(A_)) if E else (P, None, None, None)


def train_eval(variant, fold, bb, m, imgs, Y, n_cls, cw, masks):
    """Train one variant on the other folds; keep the best non-collapsed validation epoch; predict val + locked test."""
    enc = TokenEncoder(bb); model = MorphoGrader(enc, n_cls, variant).to(DEVICE)
    tr, va, te = np.where((m["fold"] >= 0) & (m["fold"] != fold))[0], np.where(m["fold"] == fold)[0], np.where(m["fold"] == -1)[0]
    dl_tr = loader(EmbryoDS(imgs, tr, Y, enc.mean, enc.std, TRAIN_AUG), BATCH, True)
    dl_va, dl_te = (loader(EmbryoDS(imgs, i, Y, enc.mean, enc.std), 64, False) for i in (va, te))
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": LR_BACKBONE},
                             {"params": [p for n, p in model.named_parameters() if not n.startswith("enc.")], "lr": LR_HEAD}], weight_decay=.05)
    sched, scaler = cosine_sched(opt, EPOCHS * len(dl_tr), len(dl_tr)), make_scaler()
    W = [torch.tensor(w, dtype=torch.float32, device=DEVICE) for w in cw]
    mtr = np.array([i for i in tr if i in masks]) if variant == "morphoground" else np.array([], int)
    mu_, sd_ = (torch.tensor(v, device=DEVICE).view(1, 3, 1, 1) for v in (enc.mean, enc.std))
    best, bad, state, curve, rng = -np.inf, 0, None, [], np.random.default_rng(fold)
    for ep in range(EPOCHS):
        model.train(); tot, nb, gl = 0.0, 0, Counter()
        for x, y in dl_tr:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE)
            with autocast():
                logits, A, Wc = model(x)
                parts = [p for p in (crit_loss(model, lg, y[:, t], W[t]) for t, lg in enumerate(logits)) if p is not None]
                if not parts: continue
                loss = sum(parts)
                if variant == "morphoground":
                    _, A2, _ = model(photometric(x)); div, comp, cons, ent, dst = grounding_terms(A, Wc, A2)
                    terms = {"diversity": div, "compact": comp, "consistency": cons, "crit_entropy": ent, "crit_distinct": dst}
                    if len(mtr):                                     # weak mask supervision on a few masked training images
                        pick = rng.choice(mtr, min(MASK_BATCH, len(mtr)), replace=False)
                        xm = torch.from_numpy(np.stack([np.array(imgs[i]) for i in pick])).to(DEVICE).permute(0, 3, 1, 2).float() / 255
                        _, Am, Wm = model((xm - mu_) / sd_); Em = evidence(Wm, Am)[0]
                        tm = torch.from_numpy(mask_targets(np.stack([masks[i] for i in pick]), math.isqrt(Em.shape[2]))).to(DEVICE)
                        ok = (tm.sum(2) > 0).float()
                        terms["mask"] = (-(tm * Em.clamp_min(1e-9).log()).sum(2) * ok).sum() / ok.sum().clamp_min(1)
                    loss = loss + sum(GROUND_W[k] * v for k, v in terms.items())
                    gl.update({k: float(v) for k, v in terms.items()})
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot, nb = tot + loss.item(), nb + 1
        Pv = predict(model, dl_va)[0]; s = sel_score(Y[va], Pv); col = collapsed_any(Y[va], Pv)
        curve.append({"epoch": ep + 1, "train_loss": tot / max(nb, 1), "val_score": s, "val_collapsed": col,
                      **{f"g_{k}": v / max(nb, 1) for k, v in gl.items()}})
        s_sel = s - (1.0 if col else 0.0)                         # never keep a collapsed epoch if a healthy one exists
        if s_sel > best: best, bad, state = s_sel, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: bad += 1
        if bad >= PATIENCE or not math.isfinite(tot): break
    if state is None: state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}   # e.g. NaN in epoch 1
    model.load_state_dict(state)
    if variant in SAVE_CKPT and fold == FOLDS[0]:
        ck = CACHE_DIR / f"phase5_{variant}_fold0.pt"; torch.save(state, ck)
        ck.with_suffix(".json").write_text(json.dumps({"backbone": bb, "variant": variant, "n_cls": n_cls, **ARCH}))
    Pva = predict(model, dl_va)[0]; Pte, E, Wc, A = predict(model, dl_te, maps=True)
    del model, enc; torch.cuda.empty_cache()
    return Pva, Pte, E, Wc, A, curve


def token_stats(A, Wc):
    """Specialisation of the morphology tokens and of the criterion evidence on the test images (one fold model)."""
    An = A / (np.linalg.norm(A, axis=2, keepdims=True) + 1e-9); G = An @ An.transpose(0, 2, 1); K = A.shape[1]
    div = float(((G.sum((1, 2)) - np.trace(G, axis1=1, axis2=2)) / (K * (K - 1))).mean())
    g = math.isqrt(A.shape[2]); yy, xx = np.meshgrid(np.linspace(0, 1, g), np.linspace(0, 1, g), indexing="ij")
    C = np.stack([yy.ravel(), xx.ravel()], 1); Ap = A / A.sum(2, keepdims=True); mu = Ap @ C
    comp = float((Ap * ((C[None, None] - mu[:, :, None]) ** 2).sum(-1)).sum(-1).mean())
    T_ = Wc[:, :, 1:] / np.maximum(Wc[:, :, 1:].sum(2, keepdims=True), 1e-9); m_ = .5 * (T_[:, 1] + T_[:, 2])
    kl = lambda p, q: (p * np.log((p + 1e-9) / (q + 1e-9))).sum(1)  # noqa: E731
    js = float((.5 * kl(T_[:, 1], m_) + .5 * kl(T_[:, 2], m_)).mean())
    ent = float((-(T_ * np.log(T_ + 1e-9)).sum(2) / np.log(K)).mean())
    E = np.einsum("bck,bkn->bcn", T_, A); En = E / (np.linalg.norm(E, axis=2, keepdims=True) + 1e-9)
    S = En @ En.transpose(0, 2, 1); sim = {f"{TASKS[i]}_vs_{TASKS[j]}": float(S[:, i, j].mean()) for i in range(3) for j in range(i + 1, 3)}
    return {"token_overlap": div, "token_spread": comp, "ICM_vs_TE_token_JS": js, "crit_attention_entropy": ent,
            "evidence_similarity": float(np.mean(list(sim.values()))), **{f"evidence_sim_{k}": v for k, v in sim.items()}}


def overlay(ax, img, emap, title):
    g = math.isqrt(len(emap)); e = emap.reshape(g, g); e = (e - e.min()) / (e.max() - e.min() + 1e-9)
    ax.imshow(img); ax.imshow(np.asarray(Image.fromarray((e * 255).astype(np.uint8)).resize(img.shape[:2][::-1], Image.BICUBIC)),
                              cmap="jet", alpha=.45); ax.set_title(title, fontsize=8); ax.axis("off")


def figures(S, P2, maps, imgs, m, Y, te, test_probs, n_cls, tstats):
    D = pd.concat([S, P2]) if len(P2) else S
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for c, t in enumerate(TASKS):
        d = D[D["task"] == t].melt(id_vars="variant", value_vars=["val_macro_f1", "val_qwk", "test_qwk"], var_name="metric")
        grouped_bars(axes[c], d, "variant", "metric", "value", None, None, f"{t}: ablation ladder (val = mean over folds, test = gold)", "score")
    axes[0].legend(fontsize=8); save(fig, OUT / "ablation_ladder.png")
    if tstats:
        T = pd.DataFrame(tstats).T; fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
        for ax, (col, lab) in zip(axes, [("token_overlap", "token map overlap (lower = more specialised)"),
                                         ("token_spread", "token spatial spread (lower = more compact)"),
                                         ("crit_attention_entropy", "criterion->token attention entropy (1 = uniform)"),
                                         ("evidence_similarity", "similarity of EXP/ICM/TE evidence maps (1 = identical)")]):
            ax.bar(T.index, T[col], color=["C0", "C3"][:len(T)]); ax.set_title(lab, fontsize=9)
        save(fig, OUT / "token_specialisation.png")
    if "morphoground" in maps:
        E, Wc = maps["morphoground"]
        fig, ax = plt.subplots(figsize=(1.1 * Wc.shape[2] + 2, 2.8))
        heat(ax, Wc.mean(0), ["global"] + [f"M{k + 1}" for k in range(Wc.shape[2] - 1)], list(TASKS),
             "Criterion -> token attention (MorphoGround, fold-0 model, mean over gold test)", "{:.2f}"); save(fig, OUT / "criterion_token_attention.png")
        pick = [i for c in range(3) for i in np.where(Y[te, 1] == c)[0][:2]][:6]
        fig, axes = plt.subplots(len(pick), 4, figsize=(9, 2.3 * len(pick)), squeeze=False)
        for r, i in enumerate(pick):
            img = np.array(imgs[te[i]]); axes[r, 0].imshow(img); axes[r, 0].axis("off")
            gt = "true " + "".join(CLASS_NAMES[t][Y[te[i], c]] if Y[te[i], c] >= 0 else "-" for c, t in enumerate(TASKS))
            axes[r, 0].set_title(gt, fontsize=8)
            for c, t in enumerate(TASKS):
                pr = CLASS_NAMES[t][test_probs["morphoground"][c][i].argmax()]
                overlay(axes[r, c + 1], img, E[i, c], f"{t} evidence -> pred {pr}")
        save(fig, OUT / "evidence_maps.png")
    best = S.groupby("variant")["val_qwk"].mean().idxmax()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for c, t in enumerate(TASKS):
        k = Y[te, c] >= 0; K = n_cls[c]
        heat(axes[c], confusion_matrix(Y[te, c][k], test_probs[best][c][k].argmax(1), labels=list(range(K))),
             CLASS_NAMES[t][:K], CLASS_NAMES[t][:K], f"{t}: {best} (gold test, rows = true)")
    save(fig, OUT / "confusion_best.png")
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))                              # predicted vs true grade distribution
    for c, t in enumerate(TASKS):
        k = Y[te, c] >= 0; K = n_cls[c]; w = .8 / (len(test_probs) + 1)
        axes[c].bar(np.arange(K) - .4 + w / 2, np.bincount(Y[te, c][k], minlength=K) / k.sum(), w, label="true", color="k")
        for j, (v, P) in enumerate(test_probs.items()):
            axes[c].bar(np.arange(K) - .4 + w * (j + 1.5), np.bincount(P[c][k].argmax(1), minlength=K) / k.sum(), w, label=v)
        axes[c].set_xticks(range(K), CLASS_NAMES[t][:K]); axes[c].set_title(f"{t}: predicted vs true grade frequency (gold test)", fontsize=9)
    axes[0].legend(fontsize=7); save(fig, OUT / "grade_distribution_check.png")


def main():
    t0 = time.time(); seed_all(); (OUT / "preds").mkdir(parents=True, exist_ok=True); CACHE_DIR.mkdir(parents=True, exist_ok=True)
    m = load_manifest(); Y, n_cls = targets(m); imgs = image_cache(m); cw = phase1_class_weights()
    bb, bb_src = choose_backbone(); te = np.where(m["fold"] == -1)[0]; log(f"Backbone: {bb} ({bb_src}); ordinal heads: {ORD_NAME}")
    masks = load_train_masks(m, IMG_SIZE)
    for old in ("global_coral", "tokens_coral"):                                   # v1 files would confuse Phases 6-9
        f = OUT / "preds" / f"{old}.npz"
        if f.exists(): f.rename(f.with_name(f"v1_{old}.npz.bak")); log(f"moved v1 file {f.name} -> v1_{old}.npz.bak")
    rows, curves, maps, test_probs, tstats, failed, collapse = [], [], {}, {}, {}, {}, {}
    for v in VARIANTS:
        oof, tf, Es = [np.full((len(m), K), np.nan) for K in n_cls], [[] for _ in TASKS], []
        try:
            for f in FOLDS:
                log(f"== {v} ({VARIANT_INFO[v]}) - fold {f}")
                va = np.where(m["fold"] == f)[0]
                Pva, Pte, E, Wc, A, curve = train_eval(v, f, bb, m, imgs, Y, n_cls, cw[f], masks)
                curves += [{**c, "variant": v, "fold": f} for c in curve]
                for t in range(len(TASKS)): oof[t][va] = Pva[t]; tf[t].append(Pte[t])
                if E is not None:
                    Es.append(E)
                    if f == FOLDS[0]: maps[v] = (None, Wc); tstats[v] = token_stats(A, Wc)
        except Exception as e:  # noqa: BLE001
            failed[v] = f"{type(e).__name__}: {str(e)[:90]}"; log(f"FAILED {v}: {failed[v]}"); continue
        tp = [np.mean(x, 0) for x in tf]; test_probs[v] = tp
        if Es: maps[v] = (np.mean(Es, 0), maps[v][1])
        cvm = np.isin(m["fold"].to_numpy(), FOLDS)
        collapse[v] = {t: {"val": collapse_report(Y[cvm, c], oof[c][cvm]), "test": collapse_report(Y[te, c], tp[c])} for c, t in enumerate(TASKS)}
        for t, T_ in enumerate(TASKS):
            for f in FOLDS:
                rows.append({"variant": v, "task": T_, "fold": str(f), "split": "val", **metrics(Y[m["fold"].to_numpy() == f, t], oof[t][m["fold"].to_numpy() == f])})
            rows.append({"variant": v, "task": T_, "fold": "ens", "split": "test", **metrics(Y[te, t], tp[t])})
        extra = {"evidence": maps[v][0], "wc_fold0": maps[v][1], "grid": math.isqrt(maps[v][0].shape[2])} if v in maps else {}
        np.savez_compressed(OUT / "preds" / f"{v}.npz", keys=m["image_key"].astype(str).to_numpy(), test_idx=te, backbone=bb,
                            head="ce" if v == "global_ce" else ORDINAL, version="v2",
                            **{f"oof_{k}": oof[t] for t, k in enumerate(TASKS)}, **{f"test_{k}": tp[t] for t, k in enumerate(TASKS)},
                            **{f"test_folds_{k}": np.stack(tf[t]) for t, k in enumerate(TASKS)}, **extra)
    if not rows: raise RuntimeError(f"No variant finished: {failed}")
    R = pd.DataFrame(rows); R.to_csv(OUT / "results_all.csv", index=False); pd.DataFrame(curves).to_csv(OUT / "learning_curves.csv", index=False)
    (OUT / "collapse_check.json").write_text(json.dumps(collapse, indent=2, default=float))
    V = R[R["split"] == "val"].groupby(["variant", "task"], sort=False).agg(val_macro_f1=("macro_f1", "mean"), val_qwk=("qwk", "mean"),
                                                                           val_f1_sd=("macro_f1", "std"), val_acc=("acc", "mean")).reset_index()
    Tst = R[R["fold"] == "ens"][["variant", "task", "qwk", "macro_f1", "weighted_f1", "acc"]].rename(
        columns={"qwk": "test_qwk", "macro_f1": "test_macro_f1", "weighted_f1": "test_weighted_f1", "acc": "test_acc"})
    S = V.merge(Tst, on=["variant", "task"]); S.to_csv(OUT / "summary.csv", index=False)
    if tstats: pd.DataFrame(tstats).T.to_csv(OUT / "token_stats.csv")
    f2 = PROJECT_DIR / "phase2_outputs" / "summary.csv"; P2 = pd.DataFrame()
    if f2.exists():
        s2 = pd.read_csv(f2); b2 = s2.loc[s2.groupby("task")["sel"].idxmax()]
        P2 = pd.DataFrame({"variant": "phase2_best", "task": b2["task"], "val_macro_f1": b2["macro_f1_mean"], "val_qwk": b2["qwk_mean"], "test_qwk": b2["test_qwk"]})
    figures(S, P2, maps, imgs, m, Y, te, test_probs, n_cls, tstats)

    # ---------------- verdict
    s2s, cv = phase_status("phase2_outputs", "PHASE2_VERDICT.txt"), m[m["fold"] >= 0]
    split_ok = not (set(m.loc[m["fold"] == -1, "image_key"]) & set(cv["image_key"])) and cv.groupby("group_id")["fold"].nunique().max() == 1
    sc = S.assign(s=(S["val_macro_f1"] + S["val_qwk"]) / 2).groupby("variant")["s"].mean()
    cmp = lambda a, b, lab: (("PASS" if sc[a] > sc[b] else "WARN"), lab, f"{b} {sc[b]:.3f} -> {a} {sc[a]:.3f} ({sc[a] - sc[b]:+.3f})") \
        if a in sc and b in sc else ("WARN", lab, "variant missing")  # noqa: E731
    col_v = {v: [t for t in TASKS if d[t]["val"]["collapsed"]] for v, d in collapse.items()}
    col_t = {v: [t for t in TASKS if d[t]["test"]["collapsed"]] for v, d in collapse.items()}
    grp = str(m["group_source"].iloc[0]) if "group_source" in m else "unknown"
    checks = [("PASS" if s2s and "FAIL" not in s2s else "FAIL" if s2s else "WARN", "Phase 2 verdict", s2s or "not found"),
              ("INFO", "Backbone / ordinal head", f"{bb} ({bb_src}) / {ORD_NAME}"),
              ("PASS" if split_ok else "FAIL", "Locked test set and grouped folds respected", f"{len(cv)} CV / {len(te)} test images; groups: {grp}"),
              ("PASS" if not failed else "WARN", "All variants trained", "yes" if not failed else str(failed)[:120]),
              ("PASS" if np.isfinite(R[["macro_f1", "qwk"]].to_numpy(float)).all() else "FAIL", "Finite metrics for every fold", f"{len(R)} rows"),
              ("PASS" if not any(col_v.values()) else "FAIL", "No ordinal collapse (validation, every variant)",
               " | ".join(f"{v} {'OK' if not c else 'COLLAPSED ' + '/'.join(c)} (EXP TV {collapse[v]['EXP']['val']['tv']:.2f})" for v, c in col_v.items())),
              ("PASS" if not any(col_t.values()) else "WARN", "No ordinal collapse (gold test)",
               " | ".join(f"{v} {'OK' if not c else '/'.join(c)}" for v, c in col_t.items())),
              cmp("global_ordinal", "global_ce", f"Ordinal ({ORD_NAME}) heads help"),
              cmp("tokens_ordinal", "global_ordinal", "RQ3: morphology tokens + relational reasoning help"),
              cmp("morphoground", "tokens_ordinal", "Grounding losses help grading")]
    if {"tokens_ordinal", "morphoground"} <= set(tstats):
        a, b = tstats["tokens_ordinal"], tstats["morphoground"]
        checks.append(("PASS" if b["token_overlap"] < a["token_overlap"] else "WARN", "Grounding makes tokens specialise (less overlap)",
                       f"overlap {a['token_overlap']:.3f} -> {b['token_overlap']:.3f}, spread {a['token_spread']:.3f} -> {b['token_spread']:.3f}"))
    if "morphoground" in tstats:
        b = tstats["morphoground"]
        checks += [("PASS" if b["crit_attention_entropy"] < .8 else "WARN", "Criteria read few tokens (attention entropy < 0.80; 1 = uniform)",
                    f"{b['crit_attention_entropy']:.3f}" + (f" (unconstrained tokens {tstats['tokens_ordinal']['crit_attention_entropy']:.3f})" if "tokens_ordinal" in tstats else "")),
                   ("PASS" if b["evidence_similarity"] < .8 else "WARN", "EXP / ICM / TE evidence maps differ (mean cosine < 0.80)",
                    ", ".join(f"{k[13:]} {v:.2f}" for k, v in b.items() if k.startswith("evidence_sim_")))]
    if len(P2) and "morphoground" in sc:
        p2s = ((P2["val_macro_f1"] + P2["val_qwk"]) / 2).mean()
        checks.append(("PASS" if sc["morphoground"] > p2s else "WARN", "MorphoGround beats Phase 2 best (val score)", f"{p2s:.3f} -> {sc['morphoground']:.3f}"))
        i2 = float(P2.loc[P2["task"] == "ICM", "val_qwk"].iloc[0]); i5 = float(S[(S["variant"] == "morphoground") & (S["task"] == "ICM")]["val_qwk"].iloc[0])
        checks.append(("INFO", "ICM bottleneck: val QWK Phase 2 best -> MorphoGround", f"{i2:.3f} -> {i5:.3f} ({i5 - i2:+.3f})"))
    sd = S.groupby("variant")["val_f1_sd"].max()
    checks.append(("PASS" if (sd.fillna(0) <= .08).all() else "WARN", "Stable across folds (max sd of val macro-F1 <= 0.08)",
                   " | ".join(f"{k} {v:.3f}" for k, v in sd.items())))
    keys = ["Variant            val F1  val QWK  val acc | val QWK EXP / ICM / TE | test QWK  test macro-F1  test acc EXP"]
    for v in [x for x in VARIANTS if x in sc]:
        d = S[S["variant"] == v].set_index("task")
        keys.append(f"  {v:<16} {d['val_macro_f1'].mean():.3f}   {d['val_qwk'].mean():.3f}   {d['val_acc'].mean():.3f}  | {d.loc['EXP', 'val_qwk']:.3f} / "
                    f"{d.loc['ICM', 'val_qwk']:.3f} / {d.loc['TE', 'val_qwk']:.3f}  | {d['test_qwk'].mean():.3f}     {d['test_macro_f1'].mean():.3f}"
                    f"          {d.loc['EXP', 'test_acc']:.3f}")
    if len(P2): keys.append(f"  {'phase2_best':<16} {P2['val_macro_f1'].mean():.3f}   {P2['val_qwk'].mean():.3f}  | " + " / ".join(f"{x:.3f}" for x in P2.set_index('task').loc[list(TASKS), 'val_qwk']))
    keys += [f"Tokens ({k}): " + ", ".join(f"{a} {b:.3f}" for a, b in v.items() if not a.startswith("evidence_sim_")) for k, v in tstats.items()]
    keys.append(f"folds {FOLDS}, epochs {EPOCHS}, tokens {N_TOKENS}, crit_global {CRIT_GLOBAL}, masks {len(masks)}, device {DEVICE}, "
                f"quick_test {QUICK_TEST}, runtime {(time.time() - t0) / 60:.1f} min")
    verdict(5, checks, keys, OUT)
    return S


if __name__ == "__main__":
    run_safely(5, main)
