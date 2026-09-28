"""MorphoGround - PHASE 4 (v2): is the model's evidence real, criterion-specific and anatomical?  (research question 2)
Uses the Phase 5 v2 fold-0 checkpoints on the locked gold test set: MorphoGround (grounded) and, for comparison, the
unconstrained token model (same architecture without grounding losses):
  1) deletion tests - hiding the top-evidence patches must hurt more than hiding random or low-evidence patches;
  2) criterion specificity - hiding ICM evidence must hurt ICM more than hiding EXP or TE evidence does (strict margin);
  3) token function vs token attention, criterion attention entropy, similarity of the EXP / ICM / TE evidence maps;
  4) optional anatomy: with compartment masks (MASKS_DIR) -> pointing game, evidence energy inside the right compartment,
     and the counterfactual 'hide the real ICM / TE / blastocoel'.
Run order: after Phase 5 v2. Self-contained: paste into ONE Jupyter cell, or run  python phase4_grounding.py"""
from pathlib import Path

# ====================================== USER SETTINGS ========================================
PROJECT_DIR = Path.home() / "Embryo_Updated Project"   # same folder as the earlier phases
PHASE1_DIR  = PROJECT_DIR / "phase1_outputs"
CACHE_DIR   = PROJECT_DIR / "cache"                     # image cache, adapted encoders, Phase 5 checkpoint
IMG_SIZE    = 224
NUM_WORKERS = 4
QUICK_TEST  = False   # True = fast smoke test first

# ---- PHASE 4 settings
MODELS      = ["morphoground", "tokens_ordinal"]   # Phase 5 fold-0 checkpoints: first = main model, second = comparison
MASKS_DIR   = None     # optional: folder of compartment masks named like the images, e.g. r"/home/<you>/masks"
MASK_LABELS = {"zona": 1, "TE": 2, "ICM": 3, "blastocoel": 4}          # pixel values in the mask images (0 = background)
EXPECTED    = {"EXP": ["blastocoel", "zona"], "ICM": ["ICM"], "TE": ["TE"]}   # compartment each criterion should rely on
FRACTIONS   = [.05, .1, .2, .3, .5]   # share of patches hidden in the deletion tests
N_RANDOM, BATCH = 3, 64
OUT         = PROJECT_DIR / "phase4_outputs"
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


def hf_stats(model_id):
    try:
        from transformers import AutoImageProcessor
        p = AutoImageProcessor.from_pretrained(model_id); return tuple(p.image_mean), tuple(p.image_std)
    except Exception: return IMNET  # noqa: BLE001


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


def evidence(Wc, A):
    """Criterion evidence maps (B, 3, N): criterion -> token weights (renormalised over tokens) x token -> patch attention."""
    T_ = Wc[:, :, 1:].float(); T_ = T_ / T_.sum(2, keepdim=True).clamp_min(1e-9)
    return torch.einsum("bck,bkn->bcn", T_, A.float()), T_


# ##################################### PHASE 4 #####################################
from scipy.stats import spearmanr  # noqa: E402
from PIL import ImageFilter  # noqa: E402
N_TEST_MAX = 40 if QUICK_TEST else None
ARCH = None   # always read from the checkpoint's .json


def preflight():
    """Phase 4 analyses the fold-0 checkpoints written by Phase 5 v2. Check they exist BEFORE loading anything, and say
    exactly what is missing and why (no traceback). Returns True when at least one usable checkpoint exists."""
    rows, usable = [], 0
    for n in MODELS:
        ck = CACHE_DIR / f"phase5_{n}_fold0.pt"; js = ck.with_suffix(".json")
        if not ck.exists() or not js.exists(): rows.append(f"  {n:<16} MISSING      {ck}"); continue
        meta = json.loads(js.read_text())
        if "ordinal" not in meta: rows.append(f"  {n:<16} v1 (CORAL)   {ck}  <- written by the old Phase 5; not usable"); continue
        rows.append(f"  {n:<16} OK           {ck}"); usable += 1
    if usable: return True
    s5 = phase_status("phase5_outputs", "PHASE5_VERDICT.txt"); pz = PROJECT_DIR / "phase5_outputs" / "preds" / "morphoground.npz"
    ver = "v2" if pz.exists() and "version" in np.load(pz, allow_pickle=True).files else "v1 (old)" if pz.exists() else "none"
    others = sorted(p.name for p in CACHE_DIR.glob("phase5_*.pt")) if CACHE_DIR.exists() else []
    bar = "=" * 100
    L = [bar, " PHASE 4 VERDICT:  NOT RUN YET - Phase 5 v2 must run first", bar,
         " Phase 4 tests the models that Phase 5 v2 trains and saves. Run order: 1 -> 2 -> 3 -> 5 -> 4 -> 6 -> 7 -> 8 -> 9.",
         f" Project folder: {PROJECT_DIR}" + ("" if PROJECT_DIR.exists() else "   <- DOES NOT EXIST: check PROJECT_DIR at the top"),
         " Checkpoints Phase 4 looked for:"] + rows + [
         f" Phase 5 verdict: {s5 or 'not found (Phase 5 has not been run in this project folder)'} | Phase 5 predictions: {ver}",
         f" Phase 5 checkpoints present in the cache: {', '.join(others) or 'none'}",
         "-" * 100,
         " How to fix:",
         "  1. Run phase5_grading.py (v2). Keep 'morphoground' and 'tokens_ordinal' in VARIANTS and in SAVE_CKPT (the defaults).",
         "     QUICK_TEST = True is enough to create the two checkpoints and smoke-test Phase 4; then run Phase 5 fully.",
         "  2. Check that Phase 5 ended with a verdict (not 'DID NOT RUN') and that the files above now exist.",
         "  3. Run this Phase 4 cell again.",
         " If Phase 5 v2 did run: make sure PROJECT_DIR / CACHE_DIR at the top of Phase 4 match Phase 5.", bar]
    print("\n" + "\n".join(L), flush=True)
    return False


def load_phase5_model(variant, n_cls):
    """Rebuild a Phase 5 v2 fold-0 model from its checkpoint (architecture read from the .json written next to it)."""
    ck = CACHE_DIR / f"phase5_{variant}_fold0.pt"; js = ck.with_suffix(".json")
    if not ck.exists() or not js.exists():
        return None, f"{ck.name} not found - run Phase 5 v2 with '{variant}' in VARIANTS and SAVE_CKPT"
    meta = json.loads(js.read_text())
    if "ordinal" not in meta: return None, f"{ck.name} is a v1 (CORAL) checkpoint - rerun Phase 5 v2"
    reg = CACHE_DIR / "phase3_backbones.json"
    if reg.exists(): BACKBONES.update({k: tuple(v) for k, v in json.loads(reg.read_text()).items()})
    for n in ("mae_in1k_dapt", "femi_dapt"):
        if n not in BACKBONES and (CACHE_DIR / n).exists(): BACKBONES[n] = ("hf", str(CACHE_DIR / n))
    arch = {k: meta[k] for k in ("n_tokens", "d_model", "rel_layers", "crit_global", "ordinal")}
    model = MorphoGrader(TokenEncoder(meta["backbone"]), meta.get("n_cls", n_cls), variant, arch)
    model.load_state_dict(torch.load(ck, map_location="cpu"))
    return model.to(DEVICE).eval(), meta


@torch.no_grad()
def run(model, X):
    """uint8 images (n, H, W, 3) -> class probabilities per criterion, token maps A (n,K,N), criterion weights Wc (n,3,1+K)."""
    mu, sd = (torch.tensor(v, device=DEVICE).view(1, 3, 1, 1) for v in (model.enc.mean, model.enc.std))
    P, As, Ws = [[] for _ in TASKS], [], []
    for i in range(0, len(X), BATCH):
        x = torch.from_numpy(np.ascontiguousarray(X[i:i + BATCH])).to(DEVICE).permute(0, 3, 1, 2).float() / 255
        with autocast(): logits, A, Wc = model((x - mu) / sd)
        for t, lg in enumerate(logits): P[t].append(class_probs(lg, model.head_type).cpu().numpy())
        As.append(A.float().cpu().numpy()); Ws.append(Wc.float().cpu().numpy())
    return [np.concatenate(p) for p in P], np.concatenate(As), np.concatenate(Ws)


def evidence_np(W, A):
    T_ = W[:, :, 1:] / np.maximum(W[:, :, 1:].sum(2, keepdims=True), 1e-9); return np.einsum("bck,bkn->bcn", T_, A), T_


def occlude(X, Bl, M):
    """Replace the patches in M (n, N) by a heavily blurred copy (stays in-distribution better than grey)."""
    g = math.isqrt(M.shape[1]); s = X.shape[1] // g
    pm = M.reshape(-1, g, g).repeat(s, 1).repeat(s, 2)[..., None]
    return np.where(pm, Bl, X)


def drop(P0, P1, valid):
    """Mean fall in the probability of the originally predicted grade (positive = the occlusion mattered)."""
    y = P0.argmax(1); r = np.arange(len(y)); return float((P0[r, y] - P1[r, y])[valid].mean()) if valid.any() else np.nan


def load_masks(keys, size):
    """Optional compartment masks (PNG/TIF, same name as the image; integer labels as in MASK_LABELS)."""
    if not MASKS_DIR: return None
    files = {Path(f).stem.lower(): f for f in Path(MASKS_DIR).rglob("*") if Path(f).suffix.lower() in (".png", ".tif", ".tiff", ".bmp")}
    if not files: raise FileNotFoundError(f"No mask images found in {MASKS_DIR}")
    out = np.full((len(keys), size, size), -1, np.int16)
    for i, k in enumerate(keys):
        if k.lower() in files:
            with Image.open(files[k.lower()]) as im: out[i] = np.asarray(im.convert("L").resize((size, size), Image.NEAREST))
    return out


def analyse(model, X, Bl, Y, te, keys, masks, rng):
    """All grounding tests for one model on the gold test images."""
    P0, A0, W0 = run(model, X); E0, T0 = evidence_np(W0, A0); N, K = A0.shape[2], A0.shape[1]
    valid = [np.ones(len(te), bool)] + [Y[te, t] >= 0 for t in (1, 2)]              # ICM/TE only where graded
    rows = []                                                                         # 1) deletion + criterion specificity
    for c, src in enumerate(TASKS):
        order = np.argsort(-E0[:, c], 1)
        for f in FRACTIONS:
            k = max(1, round(f * N))
            for kind, reps in (("top", 1), ("bottom", 1), ("random", N_RANDOM)):
                for _ in range(reps):
                    idx = order[:, :k] if kind == "top" else order[:, -k:] if kind == "bottom" else \
                        rng.permuted(np.tile(np.arange(N), (len(te), 1)), axis=1)[:, :k]
                    M = np.zeros((len(te), N), bool); np.put_along_axis(M, idx, True, 1)
                    P1 = run(model, occlude(X, Bl, M))[0]
                    rows += [{"src": src, "kind": kind, "frac": f, "tgt": tg, "drop": drop(P0[j], P1[j], valid[j])} for j, tg in enumerate(TASKS)]
        log(f"   deletion tests done for {src} evidence")
    D = pd.DataFrame(rows)
    f20 = min(FRACTIONS, key=lambda f: abs(f - .2))
    cross = D[(D["kind"] == "top") & (D["frac"] == f20)].pivot_table(index="src", columns="tgt", values="drop").loc[list(TASKS), list(TASKS)].to_numpy()
    rnd = D[(D["kind"] == "random") & (D["frac"] == f20) & (D["src"] == D["tgt"])].groupby("tgt")["drop"].mean()
    auc = D[D["src"] == D["tgt"]].groupby(["tgt", "kind"])["drop"].mean().unstack()
    gap = {t: float(cross[j, j] - rnd[t]) for j, t in enumerate(TASKS)}
    margin = {t: float(cross[j, j] - max(cross[i, j] for i in range(3) if i != j)) for j, t in enumerate(TASKS)}   # v2: strict

    tokF = np.zeros((K, len(TASKS)))                                                  # 2) what does each token do?
    for k in range(K):
        M = np.zeros((len(te), N), bool); np.put_along_axis(M, np.argsort(-A0[:, k], 1)[:, :round(.2 * N)], True, 1)
        P1 = run(model, occlude(X, Bl, M))[0]; tokF[k] = [drop(P0[j], P1[j], valid[j]) for j in range(len(TASKS))]
    tokW = T0.mean(0).T
    rho = [spearmanr(tokF[:, j], tokW[:, j])[0] if np.ptp(tokW[:, j]) > 0 else np.nan for j in range(len(TASKS))]
    En = E0 / (np.linalg.norm(E0, axis=2, keepdims=True) + 1e-9); S = En @ En.transpose(0, 2, 1)
    sim = {f"{TASKS[i]}-{TASKS[j]}": float(S[:, i, j].mean()) for i in range(3) for j in range(i + 1, 3)}
    ent = float((-(T0 * np.log(T0 + 1e-9)).sum(2) / np.log(K)).mean())

    ex = []                                                                           # occlusion examples
    for i in [int(np.argmax(Y[te, 1] == c)) for c in range(3) if (Y[te, 1] == c).any()][:3]:
        occs, txt = [], []
        for c, t in enumerate(TASKS):
            M = np.zeros((1, N), bool); M[0, np.argsort(-E0[i, c])[:round(.2 * N)]] = True; o = occlude(X[i:i + 1], Bl[i:i + 1], M)
            P1 = run(model, o)[0]; occs.append(o[0]); txt.append(f"{t} evidence hidden: P({t}) {P0[c][i].max():.2f}->{P1[c][0, P0[c][i].argmax()]:.2f}")
        ex.append((X[i], occs, txt))

    maskres, mask_checks = None, []                                                   # 3) anatomy (optional masks)
    if masks is not None:
        have = (masks >= 0).all((1, 2)); log(f"   masks found for {int(have.sum())}/{len(te)} gold-test images")
        comps = list(MASK_LABELS); g = math.isqrt(N); s = X.shape[1] // g
        frac = np.stack([(masks == MASK_LABELS[cp]).reshape(len(te), g, s, g, s).mean((2, 4)).reshape(len(te), N) for cp in comps], -1)
        pgr = []
        for c, t in enumerate(TASKS):
            ok = have & valid[c]; exp = [comps.index(e) for e in EXPECTED[t]]; fe = frac[..., exp].sum(-1)
            top = E0[:, c].argmax(1); hit = fe[np.arange(len(te)), top] >= .5
            en = (E0[:, c] * fe).sum(1) / (E0[:, c].sum(1) + 1e-9)
            pgr.append({"criterion": t, "pointing_game": float(hit[ok].mean()), "energy_in_expected": float(en[ok].mean()),
                        "area_baseline": float(fe[ok].mean())})
        pg = pd.DataFrame(pgr); comp = pd.DataFrame(index=comps, columns=list(TASKS), dtype=float)
        for cp in comps:
            pix = (masks == MASK_LABELS[cp])[..., None]; P1 = run(model, np.where(pix, Bl, X))[0]
            comp.loc[cp] = [drop(P0[j], P1[j], valid[j] & have) for j in range(len(TASKS))]
        tokcomp = np.einsum("bkn,bnc->kc", A0[have], frac[have]) / max(int(have.sum()), 1)
        maskres = (comp, pg, pd.DataFrame(tokcomp, index=[f"M{k + 1}" for k in range(K)], columns=comps))
        for _, r in pg.iterrows():
            mask_checks.append(("PASS" if r["energy_in_expected"] > r["area_baseline"] else "WARN", f"{r['criterion']} evidence sits in its compartment",
                                f"energy {r['energy_in_expected']:.2f} vs area {r['area_baseline']:.2f}, pointing game {r['pointing_game']:.2f}"))
        if {"ICM", "TE"} <= set(comps):
            a, b = comp.loc["ICM", "ICM"] > comp.loc["ICM", "TE"], comp.loc["TE", "TE"] > comp.loc["TE", "ICM"]
            mask_checks.append(("PASS" if a and b else "WARN", "Hiding the ICM hurts ICM most, hiding the TE hurts TE most",
                                f"ICM hidden: dICM {comp.loc['ICM', 'ICM']:.3f} vs dTE {comp.loc['ICM', 'TE']:.3f} | "
                                f"TE hidden: dTE {comp.loc['TE', 'TE']:.3f} vs dICM {comp.loc['TE', 'ICM']:.3f}"))
    summ = {"faithfulness_gap_20": gap, "specificity_margin_20": margin, "specific": {t: margin[t] > 0 for t in TASKS},
            "token_attention_spearman": dict(zip(TASKS, map(float, rho))), "evidence_similarity": sim,
            "evidence_similarity_mean": float(np.mean(list(sim.values()))), "crit_attention_entropy": ent, "n_tokens": K}
    if maskres: summ["pointing"] = maskres[1].set_index("criterion").to_dict("index")
    return {"D": D, "cross": cross, "rnd": rnd, "auc": auc, "tokF": tokF, "tokW": tokW, "rho": rho, "ex": ex,
            "maskres": maskres, "mask_checks": mask_checks, "summ": summ}


def figures(name, r):
    D, cross, tokF, tokW, ex, maskres = r["D"], r["cross"], r["tokF"], r["tokW"], r["ex"], r["maskres"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, t in zip(axes, TASKS):
        d = D[(D["src"] == t) & (D["tgt"] == t)].groupby(["kind", "frac"])["drop"].mean().unstack(0)
        for kind, st in (("top", "o-"), ("random", "s--"), ("bottom", "^:")):
            if kind in d: ax.plot(d.index * 100, d[kind], st, label=f"{kind} evidence occluded")
        ax.set(title=f"{t}: deletion test ({name})", xlabel="% of patches occluded", ylabel=f"drop in P(predicted {t} grade)"); ax.legend(fontsize=7)
    save(fig, OUT / f"deletion_curves_{name}.png")
    fig, ax = plt.subplots(figsize=(5, 4))
    heat(ax, cross, [f"{t} affected" for t in TASKS], [f"{t} evidence occluded" for t in TASKS],
         f"Criterion specificity, {name} (top 20% occluded)", "{:.3f}"); save(fig, OUT / f"criterion_specificity_{name}.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, .45 * len(tokF) + 2))
    heat(axes[0], tokF, list(TASKS), [f"M{k + 1}" for k in range(len(tokF))], "Function: drop when token region occluded", "{:.3f}")
    heat(axes[1], tokW, list(TASKS), [f"M{k + 1}" for k in range(len(tokW))], "Attention: criterion -> token weight", "{:.2f}")
    save(fig, OUT / f"token_function_vs_attention_{name}.png")
    fig, axes = plt.subplots(len(ex), 4, figsize=(10, 2.6 * len(ex)), squeeze=False)
    for i, (img, occs, txt) in enumerate(ex):
        axes[i, 0].imshow(img); axes[i, 0].set_title("original", fontsize=8)
        for c, (o, tt) in enumerate(zip(occs, txt)): axes[i, c + 1].imshow(o); axes[i, c + 1].set_title(tt, fontsize=7)
        for a in axes[i]: a.axis("off")
    save(fig, OUT / f"occlusion_examples_{name}.png")
    if maskres:
        comp, pg, _ = maskres
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        grouped_bars(axes[0], pg.melt(id_vars="criterion", value_vars=["pointing_game", "energy_in_expected", "area_baseline"], var_name="metric"),
                     "criterion", "metric", "value", None, None, f"Anatomical grounding of evidence maps ({name})", "fraction")
        axes[0].legend(fontsize=7)
        heat(axes[1], comp.to_numpy(float), list(comp.columns), list(comp.index), "Compartment counterfactual: drop per criterion", "{:.3f}")
        save(fig, OUT / f"anatomical_grounding_{name}.png")


def compare_figure(res):
    rows = [{"model": n, "criterion": t, "metric": met, "value": r["summ"][key][t]} for n, r in res.items() for t in TASKS
            for met, key in (("faithfulness gap", "faithfulness_gap_20"), ("specificity margin", "specificity_margin_20"))]
    D = pd.DataFrame(rows); fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, met in zip(axes, ("faithfulness gap", "specificity margin")):
        grouped_bars(ax, D[D["metric"] == met], "criterion", "model", "value", None, 0,
                     f"{met} (top-20% occlusion; > 0 = {'faithful' if met.startswith('f') else 'criterion-specific'})", "drop difference")
        ax.legend(fontsize=7)
    save(fig, OUT / "grounded_vs_unconstrained.png")


def main():
    if not preflight(): return None                                               # v2.1: clear message, no traceback
    t0 = time.time(); seed_all(); OUT.mkdir(parents=True, exist_ok=True)
    m = load_manifest(); Y, n_cls = targets(m); imgs = image_cache(m)
    te = np.where(m["fold"] == -1)[0][:N_TEST_MAX]; keys = m["image_key"].astype(str).to_numpy()[te]
    X = np.stack([np.array(imgs[i]) for i in te])
    Bl = np.stack([np.asarray(Image.fromarray(x).filter(ImageFilter.GaussianBlur(8))) for x in X])
    masks = load_masks(keys, X.shape[1]); res, meta, missing = {}, {}, {}
    for name in MODELS:
        model, info = load_phase5_model(name, n_cls)
        if model is None: missing[name] = info; log(f"SKIPPED {name}: {info}"); continue
        meta[name] = info; log(f"== {name}: Phase 5 fold-0 model (backbone {info['backbone']}, {info['ordinal']} heads) on {len(te)} gold-test images")
        res[name] = analyse(model, X, Bl, Y, te, keys, masks, np.random.default_rng(0)); figures(name, res[name])
        res[name]["D"].to_csv(OUT / f"deletion_results_{name}.csv", index=False)
        if res[name]["maskres"]:
            comp, pg, tc = res[name]["maskres"]
            pg.to_csv(OUT / f"pointing_game_{name}.csv", index=False); comp.to_csv(OUT / f"compartment_counterfactual_{name}.csv"); tc.to_csv(OUT / f"token_compartment_energy_{name}.csv")
        del model; torch.cuda.empty_cache()
    if not res: raise FileNotFoundError("No Phase 5 v2 checkpoint could be loaded: " + "; ".join(missing.values()))
    main_m = MODELS[0] if MODELS[0] in res else next(iter(res))
    if len(res) > 1: compare_figure(res)
    summ = {**res[main_m]["summ"], "main_model": main_m, "backbone": meta[main_m]["backbone"], "n_test": len(te),
            "models": {n: r["summ"] for n, r in res.items()}}
    (OUT / "phase4_summary.json").write_text(json.dumps(summ, indent=2, default=float))

    # ---------------- verdict
    r, s = res[main_m], res[main_m]["summ"]; cross, rnd = r["cross"], r["rnd"]
    s5 = phase_status("phase5_outputs", "PHASE5_VERDICT.txt")
    checks = [("PASS" if s5 and "FAIL" not in s5 else "FAIL" if s5 else "WARN", "Phase 5 verdict", s5 or "not found"),
              ("PASS" if not missing else "WARN", "Phase 5 v2 checkpoints reloaded", ", ".join(res) + (f" | missing: {missing}" if missing else "")),
              ("PASS" if all(v > 0 for v in s["faithfulness_gap_20"].values()) else "WARN", f"{main_m}: evidence is faithful (top-20% occlusion > random)",
               " | ".join(f"{t} {cross[j, j]:.3f} vs {rnd[t]:.3f}" for j, t in enumerate(TASKS))),
              ("PASS" if all(s["specific"].values()) else "WARN", f"{main_m}: criterion-specific (own evidence hurts own grade most)",
               " | ".join(f"{t} margin {v:+.3f}" for t, v in s["specificity_margin_20"].items())),
              ("PASS" if s["evidence_similarity_mean"] < .8 else "WARN", f"{main_m}: EXP / ICM / TE evidence maps differ (cosine < 0.80)",
               ", ".join(f"{k} {v:.2f}" for k, v in s["evidence_similarity"].items())),
              ("PASS" if np.nanmean(r["rho"]) > 0 else "WARN", f"{main_m}: token attention agrees with token function (Spearman)",
               " | ".join(f"{t} {x:+.2f}" for t, x in zip(TASKS, r["rho"])))]
    if len(res) > 1:
        other = next(n for n in res if n != main_m); so = res[other]["summ"]
        mg = lambda d, k: float(np.mean(list(d[k].values())))  # noqa: E731
        checks += [("PASS" if mg(s, "faithfulness_gap_20") > mg(so, "faithfulness_gap_20") else "WARN", "Grounding makes evidence more faithful",
                    f"mean gap {other} {mg(so, 'faithfulness_gap_20'):+.3f} -> {main_m} {mg(s, 'faithfulness_gap_20'):+.3f}"),
                   ("PASS" if mg(s, "specificity_margin_20") > mg(so, "specificity_margin_20") else "WARN", "Grounding makes evidence more criterion-specific",
                    f"mean margin {other} {mg(so, 'specificity_margin_20'):+.3f} -> {main_m} {mg(s, 'specificity_margin_20'):+.3f}; "
                    f"map similarity {so['evidence_similarity_mean']:.2f} -> {s['evidence_similarity_mean']:.2f}")]
    checks += r["mask_checks"] or [("INFO", "Anatomical metrics (pointing game, compartment counterfactual)",
                                    "need masks: set MASKS_DIR (annotate 100-300 gold-test images, or export from a segmentation model)")]
    keys_ = []
    for n, rr in res.items():
        a = rr["auc"]
        keys_ += [f"{n}: deletion (mean drop over fractions) " + " | ".join(f"{t}: top {a.loc[t, 'top']:.3f}, random {a.loc[t, 'random']:.3f}, "
                                                                         f"bottom {a.loc[t, 'bottom']:.3f}" for t in TASKS),
                  f"{n}: specificity matrix (rows = evidence hidden, cols = drop EXP/ICM/TE) " +
                  " ; ".join(f"{t}: " + "/".join(f"{v:.3f}" for v in rr["cross"][j]) for j, t in enumerate(TASKS)),
                  f"{n}: criterion attention entropy {rr['summ']['crit_attention_entropy']:.2f} (1 = uniform), map similarity "
                  f"{rr['summ']['evidence_similarity_mean']:.2f}, token -> criterion by function: " +
                  ", ".join(f"M{k + 1}->{TASKS[int(np.nanargmax(rr['tokF'][k]))]}" for k in range(len(rr["tokF"])))]
        if rr["maskres"]: keys_.append(f"{n}: pointing game " + ", ".join(f"{q.criterion} {q.pointing_game:.2f}" for q in rr["maskres"][1].itertuples()))
    keys_.append(f"fractions {FRACTIONS}, random repeats {N_RANDOM}, gold-test images {len(te)}, device {DEVICE}, runtime {(time.time() - t0) / 60:.1f} min")
    verdict(4, checks, keys_, OUT, "Grounding tested - continue with Phase 6 (reliability), then Phases 7-9.")
    return summ


if __name__ == "__main__":
    run_safely(4, main)
