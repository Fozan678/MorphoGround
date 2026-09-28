#!/usr/bin/env python3
"""MorphoGround - PHASE 1: dataset audit, QC and leakage-safe splits (Kromp et al., Sci Data 10:271, 2023).
Run: `python phase1_data_audit.py`   |   Jupyter: `from phase1_data_audit import *; m = run_phase1(Phase1Config())`
Gold set = locked test (fold -1). Silver set = 5-fold StratifiedGroupKFold, grouped by patient/cycle: an ID column if
the label files have one, otherwise the file-name prefix (Kromp names are NNNN_EE = cycle NNNN, embryo EE), always united
with duplicate clusters. The prefix hypothesis is tested on the data (embryo numbering + clustering of acquisition
statistics within a prefix). Only unreadable/missing images and train images that duplicate a test image are excluded
by default; TEST_SIBLINGS = "exclude" also removes sibling embryos of test embryos (sibling-free protocol).
v2: patient/cycle grouping by file-name prefix, sibling policy, fixed duplicate-label comparison ("4.0" vs "4").
At the end a PASS / WARN / FAIL verdict is printed in the terminal and saved as QC_VERDICT.txt."""
from pathlib import Path
from typing import Optional

# ========================================== USER SETTINGS ==========================================
HOME          = Path.home()                                                     # e.g. /home/<you>
IMAGES_DIR    = HOME / "Downloads/archive/Images/Images"                        # images (subfolders searched too)
SILVER_FILE   = HOME / "Downloads/archive/Gardner_train_silver.csv"             # silver-standard (train) labels
GOLD_FILE     = HOME / "Downloads/archive/Gardner_test_gold_onlyGardnerScores.csv"  # gold-standard (test) labels
COMPLETE_FILE = None  # optional: authors' GitHub complete.csv -> silver-vs-gold agreement on the 300 test images
CLINICAL_FILE = None  # optional: clinical annotations file
GROUP_BY      = "auto"   # CV groups: auto = patient/cycle ID column if present, else file-name prefix, else duplicates only
PREFIX_REGEX  = r"^0*(\d+)_\d+$"   # file name -> patient/cycle key (Kromp: NNNN_EE); leading zeros ignored
TEST_SIBLINGS = "keep"   # keep = official benchmark split (siblings of test embryos stay in training; flagged, and Phase 9
                         #        reports gold-test results with / without a training sibling) | exclude = sibling-free protocol
PROJECT_DIR   = HOME / "Embryo_Updated Project"                                  # all project results go here
OUT_DIR       = PROJECT_DIR / "phase1_outputs"                                   # this phase's results
# ===================================================================================================

import csv, hashlib, json, logging, os, platform, re, sys, time, warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.fft import dct
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
try: from tqdm.auto import tqdm
except ImportError: tqdm = lambda x, **_: x  # noqa: E731

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
LOG = logging.getLogger("phase1")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
Q, ND, CRITS = ["A", "B", "C"], "ND", ("EXP", "ICM", "TE")          # Q is ordinal: A best -> C worst
ND_TOK = {"nd", "n.d.", "notdefined", "not defined", "not_defined", "notapplicable", "not applicable", "not_applicable"}
MISS_TOK = {"na", "n/a", "n.a.", "nan", "none", "null", "-", "--", "?", "notassessable", "not assessable", "not_assessable"}
NUM = re.compile(r"^-?\d+(\.0+)?$")
FOLD_TEST, FOLD_EXCLUDED, FOLD_UNLABELED = -1, -2, -3
ROLES = {"image": {"image", "img", "images", "filename", "file", "imagename", "imagefile"}, "exp": {"exp", "expansion"},
         "icm": {"icm", "innercellmass"}, "te": {"te", "teq", "trophectoderm"},
         "patient": {"patient", "patientid", "pid", "couple", "woman", "female"},
         "cycle": {"cycle", "cycleid", "treatment", "treatmentid"}, "embryo": {"embryo", "embryoid", "blastocystid"}}
REQUIRED, OPTIONAL = ("image", "exp", "icm", "te"), ("patient", "cycle", "embryo")


@dataclass
class Phase1Config:
    images_dir: Path = IMAGES_DIR; silver_path: Path = SILVER_FILE; gold_path: Path = GOLD_FILE; out_dir: Path = OUT_DIR
    complete_path: Optional[Path] = COMPLETE_FILE; clinical_path: Optional[Path] = CLINICAL_FILE
    silver_sheet: Optional[str] = None; gold_sheet: Optional[str] = None       # only for Excel files (None = 1st sheet)
    silver_cols: dict = field(default_factory=dict)   # only if auto-detection fails, e.g. {"exp": "EXP_silver"}
    gold_cols: dict = field(default_factory=dict)
    exp_zero_based: Optional[bool] = None             # None = auto (a 0 in EXP -> zero-based coding)
    quality_map: dict = field(default_factory=dict)   # force ICM/TE coding, e.g. {"0": "A", "1": "B", "2": "C", "3": "ND"}
    n_folds: int = 5; seed: int = 42; num_workers: int = 4
    hash_threshold: int = 10       # duplicate candidates: pHash AND dHash Hamming distance <= this
    ncc_threshold: float = 0.92    # confirmed duplicate: band-passed texture correlation >= this
    leak_policy: str = "drop_train"   # drop_train | keep
    group_by: str = GROUP_BY          # auto | prefix | duplicates
    prefix_regex: str = PREFIX_REGEX
    test_siblings: str = TEST_SIBLINGS  # keep | exclude
    n_permutations: int = 1000        # permutation test of the prefix = patient/cycle hypothesis
    rare_class_threshold: int = 30; qc_z_threshold: float = 3.5; saturation_threshold: float = 0.05
    gallery_per_class: int = 6


# ---------------------------------------------------------------- utilities
def s_(v):
    """Clean string or None."""
    return None if v is None or (not isinstance(v, str) and pd.isna(v)) else (str(v).strip() or None)


def digest(path, algo="md5"):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def image_key(name):
    """Lower-case basename without an image extension (dots inside names are kept)."""
    b = str(name).strip().replace("\\", "/").split("/")[-1]
    return (Path(b).stem if Path(b).suffix.lower() in IMG_EXTS else b).lower()


def label_str(v):
    """Canonical label text for comparisons: 4, 4.0, '4' and '4.0' -> '4'; 'b ' -> 'B'; missing -> ''."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA: return ""
    s = str(v).strip()
    if s.lower() in ("nan", "<na>", "none", ""): return ""
    return str(int(float(s))) if NUM.match(s) else s.upper()


def icc1(x, codes):
    """One-way random-effects ICC(1) of x across groups (codes = integer group per row)."""
    ok = np.isfinite(x); x, codes = x[ok], codes[ok]
    _, codes = np.unique(codes, return_inverse=True); k, N = codes.max() + 1, len(x)
    if k < 2 or N <= k: return np.nan
    n = np.bincount(codes, minlength=k).astype(float); gm = np.bincount(codes, x, k) / n
    msb = (n * (gm - x.mean()) ** 2).sum() / (k - 1); msw = ((x - gm[codes]) ** 2).sum() / (N - k)
    n0 = (N - (n ** 2).sum() / N) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw)) if msb + (n0 - 1) * msw > 0 else np.nan


def prefix_evidence(m, col, n_perm=200, seed=0):
    """Is the file-name prefix a patient/cycle ID? (a) embryo numbers inside a prefix run 1..n (one cycle's blastocysts),
    (b) images sharing a prefix share acquisition conditions (same dish/session): ICC of contrast, sharpness and
    brightness across prefixes, tested against prefix labels permuted at random (group sizes kept)."""
    rd = m[m["readable"] & m[col].notna()]
    codes = pd.factorize(rd[col])[0]; sizes = rd.groupby(col).size()
    emb = pd.to_numeric(rd["image_key"].astype(str).str.extract(r"_0*(\d+)$")[0], errors="coerce")
    g = pd.DataFrame({"p": rd[col].to_numpy(), "e": emb.to_numpy()}).groupby("p")["e"].agg(["min", "max", "count"])
    consec = float(((g["min"] == 1) & (g["max"] == g["count"])).mean()) if len(g) else np.nan
    rng, stats = np.random.default_rng(seed), {}
    for name, x in (("contrast", rd["std_intensity"]), ("sharpness", np.log1p(rd["laplacian_var"].astype(float))),
                    ("brightness", rd["mean_intensity"])):
        x = pd.to_numeric(x, errors="coerce").to_numpy(float); obs = icc1(x, codes)
        null = np.array([icc1(x, rng.permutation(codes)) for _ in range(n_perm)])
        stats[name] = {"icc": obs, "null_mean": float(np.nanmean(null)), "null_p99": float(np.nanpercentile(null, 99)),
                       "p_perm": float((1 + np.sum(null >= obs)) / (n_perm + 1))}
    n_sig = sum(v["p_perm"] < .01 and v["icc"] > .1 for v in stats.values())
    return {"key": col, "n_keys": int(len(sizes)), "images_per_key": {"mean": float(sizes.mean()), "max": int(sizes.max()),
            "singletons": int((sizes == 1).sum())}, "embryo_numbers_run_1_to_n": consec, "acquisition_icc": stats,
            "n_significant_stats": int(n_sig), "supported": bool(consec >= .9 and n_sig >= 2)}


def _jd(o):  # json default
    return o.item() if isinstance(o, np.generic) else list(o) if isinstance(o, (set, tuple)) else str(o)


def md_table(df, max_rows=60, index=True):
    """Markdown table without the optional 'tabulate' dependency."""
    if df is None or len(df) == 0: return "_(empty)_\n"
    d = df.head(max_rows).reset_index() if index else df.head(max_rows)
    f = lambda v: ("" if v is None or (not isinstance(v, (str, dict)) and pd.isna(v))  # noqa: E731
                   else f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v))
    rows = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * d.shape[1]]
    rows += ["| " + " | ".join(f(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(rows) + (f"\n\n_... {len(df) - max_rows} more rows in the CSV_" if len(df) > max_rows else "") + "\n"


# ---------------------------------------------------------------- annotation files
def read_table(path, sheet=None):
    """Read xlsx/xls/csv as text (codes like 'NA'/'ND' stay strings); the CSV delimiter is sniffed."""
    p, kw = Path(path), dict(dtype=str, keep_default_na=False, na_values=[""])
    meta = {"path": str(p), "sha256": digest(p, "sha256")}
    if p.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        xl = pd.ExcelFile(p); sh = sheet if sheet is not None else xl.sheet_names[0]
        meta.update(sheets=xl.sheet_names, sheet_used=sh); df = pd.read_excel(xl, sheet_name=sh, **kw)
    else:
        with open(p, encoding="utf-8-sig", errors="replace") as fh: head = fh.read(8192)
        try: meta["delimiter"] = csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
        except csv.Error: meta["delimiter"] = ","
        df = pd.read_csv(p, sep=meta["delimiter"], encoding="utf-8-sig", skipinitialspace=True, **kw)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[[c for c in df.columns if not (c.lower().startswith("unnamed") and df[c].isna().all())]]
    df = df.apply(lambda c: c.map(s_)).dropna(how="all").reset_index(drop=True)
    meta.update(n_rows=len(df), columns=list(df.columns))
    return df, meta


def read_complete_csv(path):
    """complete.csv from the authors' GitHub: 'file, exp, icm, teq' (no header)."""
    with open(path, encoding="utf-8-sig") as fh: rows = [[x.strip() for x in l.split(",") if x.strip()][:4] for l in fh]
    df = pd.DataFrame([r for r in rows if len(r) == 4], columns=["image", "exp", "icm", "te"])
    return (df.iloc[1:] if len(df) and not NUM.match(df.iloc[0]["exp"]) else df).reset_index(drop=True)


def _tok(col):
    low = col.lower()
    return {t for t in re.split(r"[^a-z0-9]+", low) if t} | {re.sub(r"[^a-z0-9]", "", low)}


def detect_columns(df, prefer, overrides, tag):
    """Map roles (image/exp/icm/te/patient/cycle/embryo) to columns; fail loudly if ambiguous."""
    cols, found = list(df.columns), {}
    for role, col in overrides.items():
        if col not in cols: raise KeyError(f"[{tag}] column '{col}' for role '{role}' not found in {cols}")
        found[role] = col
    for role in [r for r in REQUIRED + OPTIONAL if r not in found]:
        free = [c for c in cols if c not in found.values()]
        cands = [c for c in free if _tok(c) & ROLES[role]] or \
                ([c for c in free if _tok(c) & {"name", "path", "filename"}] if role == "image" else [])
        if len(cands) > 1: cands = [c for c in cands if any(k in c.lower() for k in prefer)] or cands
        if len(cands) == 1: found[role] = cands[0]
        elif cands and role in REQUIRED:
            raise ValueError(f"[{tag}] several columns could be '{role}': {cands}. "
                             f"Set Phase1Config({tag}_cols={{'{role}': '{cands[0]}'}}).")
        elif role == "image": found["image"] = cols[0]; LOG.warning(f"[{tag}] no image column found; using '{cols[0]}'")
        elif role in REQUIRED: raise ValueError(f"[{tag}] no column found for '{role}'. Columns: {cols}")
    return found


def normalize(raw, kind, tag, cfg, audit):
    """EXP -> Gardner grade 1..6 (Int64, 0-based coding auto-detected); ICM/TE -> 'A'/'B'/'C'/'ND' or NaN."""
    vals = [s_(v) for v in raw]
    nums = [int(float(v)) for v in vals if v and NUM.match(v)]
    out, invalid = [], []
    if kind == "EXP":
        zb = cfg.exp_zero_based if cfg.exp_zero_based is not None else (bool(nums) and min(nums) == 0)
        for v in vals:
            g = int(float(v)) + int(zb) if v and NUM.match(v) else None
            if v and (g is None or not 1 <= g <= 6): invalid.append(v); g = None
            out.append(np.nan if g is None else g)
        coding, res = {"zero_based": zb}, pd.Series(out, index=raw.index, dtype="float64").astype("Int64")
    else:
        qmap = ({int(k): v.strip().upper() for k, v in cfg.quality_map.items()} if cfg.quality_map
                else {0: "A", 1: "B", 2: "C", 3: ND, -1: ND} if 0 in nums       # authors' GitHub coding
                else {1: "A", 2: "B", 3: "C", 4: ND, -1: ND})                   # 1-based fallback
        for v in vals:
            q = ("" if not v else qmap.get(int(float(v))) if NUM.match(v) else v.upper() if v.upper() in Q
                 else ND if v.lower() in ND_TOK else "" if v.lower() in MISS_TOK else None)
            if q is None: invalid.append(v)
            out.append(q or np.nan)
        coding = {"numeric_map": {str(k): v for k, v in qmap.items()} if nums else None}
        res = pd.Series(out, index=raw.index, dtype=object)
    audit.setdefault("coding", {})[f"{tag}.{kind}"] = {**coding, "raw_value_counts": dict(Counter(vals).most_common()),
                                                        "n_invalid": len(invalid), "invalid_examples": invalid[:10]}
    return res


def standardize(df, cols, tag, cfg, audit):
    """Common schema: image_key, EXP/ICM/TE_<tag>, raw codes, optional IDs, extra columns as <tag>__<col>."""
    out = pd.DataFrame({f"image_name_{tag}": df[cols["image"]].astype(str)})
    out["image_key"] = out[f"image_name_{tag}"].map(image_key)
    for c in CRITS:
        out[f"{c}_{tag}"] = normalize(df[cols[c.lower()]], c, tag, cfg, audit)
        out[f"{c}_raw_{tag}"] = df[cols[c.lower()]]
    for r in [r for r in OPTIONAL if r in cols]: out[f"{r}_id_{tag}"] = df[cols[r]].astype(str)
    for c in [c for c in df.columns if c not in cols.values()]: out[f"{tag}__{c}"] = df[c]
    dup = out["image_key"].duplicated(keep=False)
    conflicts = [k for k, g in out[dup].groupby("image_key")
                 if len(g[[f"{c}_{tag}" for c in CRITS]].astype(str).drop_duplicates()) > 1]
    if dup.any(): LOG.warning(f"[{tag}] {dup.sum()} rows share an image name ({len(conflicts)} conflicting); keeping first")
    audit.setdefault("duplicate_rows", {})[tag] = {"n_rows_with_repeated_name": int(dup.sum()),
                                                   "n_names_with_conflicting_labels": len(conflicts), "examples": conflicts[:10]}
    return out.drop_duplicates("image_key").reset_index(drop=True)


# ---------------------------------------------------------------- image QC
def index_images(folder):
    idx = defaultdict(list)
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if Path(f).suffix.lower() in IMG_EXTS and not f.startswith("._"): idx[image_key(f)].append(os.path.join(root, f))
    return idx


def _bits(b):
    return f"{int.from_bytes(np.packbits(b.astype(np.uint8).ravel()).tobytes(), 'big'):016x}"


def qc_one(path):
    """Readability, geometry, intensity, exposure, sharpness, MD5, pHash and dHash for one image."""
    rec = {"path": path, "readable": False, "error": None}
    try:
        rec.update(file_size_kb=round(os.path.getsize(path) / 1024, 1), md5=digest(path))
        with Image.open(path) as im:
            im.load(); rec.update(width=im.width, height=im.height, mode=im.mode)
            if im.mode.startswith("I;16") or im.mode in ("I", "F"):
                a = np.asarray(im, dtype=np.float32)
                gray, diff = Image.fromarray((a / max(float(a.max()), 1.0) * 255).astype(np.uint8)), 0
            else:
                rgb = im.convert("RGB"); a, gray = np.asarray(rgb, dtype=np.int16), rgb.convert("L")
                diff = int(max(np.abs(a[..., 0] - a[..., 1]).max(), np.abs(a[..., 1] - a[..., 2]).max()))
        g = np.asarray(gray, dtype=np.float32); p01, p99 = np.percentile(g, [1, 99])
        gs = np.asarray(gray.resize((512, max(1, round(gray.height * 512 / gray.width))), Image.BILINEAR), np.float32)
        d32 = dct(dct(np.asarray(gray.resize((32, 32), Image.LANCZOS), np.float32), axis=0, norm="ortho"),
                  axis=1, norm="ortho")[:8, :8]
        d9 = np.asarray(gray.resize((9, 8), Image.LANCZOS), np.float32)
        rec.update(channel_max_diff=diff, grayscale_content=diff <= 2, mean_intensity=float(g.mean()),
                   std_intensity=float(g.std()), p01=float(p01), p99=float(p99), frac_high_sat=float((g >= 250).mean()),
                   frac_low_sat=float((g <= 5).mean()), laplacian_var=float(ndimage.laplace(gs).var()),  # sharpness @512px
                   phash=_bits(d32 > np.median(d32.ravel()[1:])), dhash=_bits(d9[:, 1:] > d9[:, :-1]), readable=True)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def run_qc(paths, workers):
    if workers > 1:
        try:
            with ProcessPoolExecutor(workers) as ex:
                return pd.DataFrame(list(tqdm(ex.map(qc_one, paths, chunksize=16), total=len(paths), desc="Image QC")))
        except Exception as e:  # noqa: BLE001 (e.g. process-pool issues inside notebooks)
            LOG.warning(f"Parallel QC failed ({e}); running sequentially")
    return pd.DataFrame([qc_one(p) for p in tqdm(paths, desc="Image QC")])


def robust_z(x):
    x = np.asarray(x, float); med = np.nanmedian(x); mad = np.nanmedian(np.abs(x - med))
    return np.zeros_like(x) if not np.isfinite(mad) or mad == 0 else 0.6745 * (x - med) / mad


def add_qc_flags(qc, cfg):
    """Review flags (they do NOT exclude images, except 'unreadable')."""
    qc, ok = qc.copy(), qc["readable"].eq(True)
    flags = [[] if o else ["unreadable"] for o in ok]
    if ok.any():
        r, t = qc[ok], cfg.qc_z_threshold
        z = {"z_brightness": robust_z(r["mean_intensity"]), "z_contrast": robust_z(r["std_intensity"]),
             "z_sharpness": robust_z(np.log1p(r["laplacian_var"]))}
        modal = r.groupby(["width", "height"]).size().idxmax()
        for j, i in enumerate(np.where(ok)[0]):
            row = r.iloc[j]
            for name, cond in (("brightness_outlier", abs(z["z_brightness"][j]) > t), ("low_contrast", z["z_contrast"][j] < -t),
                               ("blurry", z["z_sharpness"][j] < -t), ("overexposed", row["frac_high_sat"] > cfg.saturation_threshold),
                               ("underexposed", row["frac_low_sat"] > cfg.saturation_threshold),
                               ("nonstandard_size", (row["width"], row["height"]) != modal)):
                if cond: flags[i].append(name)
        for k, v in z.items(): qc.loc[ok, k] = v
    qc["qc_flags"] = [";".join(f) for f in flags]
    qc["qc_review"] = [any(x != "nonstandard_size" for x in f) for f in flags]
    return qc


# ---------------------------------------------------------------- duplicates -> leakage-aware groups
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
popcount64 = lambda x: np.bitwise_count(x) if hasattr(np, "bitwise_count") else _POP8[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1)  # noqa


class UnionFind:
    def __init__(self, n): self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a: self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[max(ra, rb)] = min(ra, rb)

    def union_groups(self, df, col):
        for _, g in df.groupby(col):
            for i in g.index[1:]: self.union(g.index[0], i)


def texture_fft(path, size=128):
    """FFT of a band-passed thumbnail: removing low frequencies stops the shared 'round embryo on grey
    background' layout from making every pair of images look alike."""
    with Image.open(path) as im: g = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)
    g = ndimage.gaussian_filter(g - ndimage.gaussian_filter(g, 4), 0.7)
    return np.fft.rfft2((g - g.mean()) / (g.std() + 1e-6))


def max_shift_ncc(fa, fb, size=128, shift=12):
    """Max normalised cross-correlation over shifts up to +-shift pixels."""
    cc, c = np.fft.fftshift(np.fft.irfft2(fa * np.conj(fb), s=(size, size))) / size ** 2, size // 2
    return float(cc[c - shift:c + shift + 1, c - shift:c + shift + 1].max())


def find_duplicates(m, cfg, chunk=512):
    """Exact = same MD5. Near = pHash AND dHash distance <= hash_threshold, confirmed by texture NCC >= ncc_threshold.
    Returns (cluster root per row, confirmed pairs, all scored candidates). Expects m.index = 0..n-1."""
    uf, pairs, cands, scored, cache = UnionFind(len(m)), [], [], [], {}
    idx = np.where(m["readable"])[0]
    for _, g in m.iloc[idx].groupby("md5"):
        for o in g.index[1:]: uf.union(g.index[0], o); pairs.append((g.index[0], o, "exact_md5", 0, 0, 1.0))
    ph, dh = (np.array([int(h, 16) for h in m[c].iloc[idx]], dtype=np.uint64) for c in ("phash", "dhash"))
    for s in range(0, len(idx), chunk):
        dp, dd = popcount64(ph[s:s + chunk, None] ^ ph[None]), popcount64(dh[s:s + chunk, None] ^ dh[None])
        for a, b in zip(*np.nonzero((dp <= cfg.hash_threshold) & (dd <= cfg.hash_threshold))):
            ra, rb = idx[a + s], idx[b]
            if b > a + s and m.at[ra, "md5"] != m.at[rb, "md5"]: cands.append((ra, rb, int(dp[a, b]), int(dd[a, b])))
    LOG.info(f"near-duplicate candidates from hashes: {len(cands)}")
    if len(cands) > 200_000: LOG.warning("Very many hash candidates; consider lowering hash_threshold")
    fft = lambda i: cache[i] if i in cache else cache.setdefault(i, texture_fft(m.at[i, "path"]))  # noqa: E731
    for ra, rb, dp_, dd_ in tqdm(cands, desc="Verifying candidates"):
        ncc = round(max_shift_ncc(fft(ra), fft(rb)), 4)
        scored.append((m.at[ra, "image_key"], m.at[rb, "image_key"], dp_, dd_, ncc))
        if ncc >= cfg.ncc_threshold: uf.union(ra, rb); pairs.append((ra, rb, "near_verified", dp_, dd_, ncc))
    return (pd.Series([uf.find(i) for i in range(len(m))], index=m.index),
            pd.DataFrame(pairs, columns=["row_a", "row_b", "type", "phash_dist", "dhash_dist", "texture_ncc"]),
            pd.DataFrame(scored, columns=["image_key_a", "image_key_b", "phash_dist", "dhash_dist", "texture_ncc"]))


# ---------------------------------------------------------------- labels: integrity, targets, tables
def gardner_issues(exp, icm, te):
    """EXP 1-2 -> ICM/TE must be ND; EXP >= 3 -> ICM/TE must be A/B/C."""
    e = pd.to_numeric(exp, errors="coerce"); early, late = e.isin([1, 2]), (e >= 3).fillna(False)
    checks = {"EXP_missing": e.isna()}
    for c, q in (("ICM", icm.astype(object)), ("TE", te.astype(object))):
        checks.update({f"{c}_graded_at_EXP<=2": early & q.isin(Q), f"{c}_ND_at_EXP>=3": late & (q == ND),
                       f"{c}_missing_at_EXP>=3": late & q.isna()})
    arr = {k: v.fillna(False).to_numpy(bool) for k, v in checks.items()}
    return pd.Series([";".join(k for k, v in arr.items() if v[i]) for i in range(len(e))], index=exp.index)


def ordinal_encode(m):
    """Targets: EXP_ord 0..K-1, ICM/TE_ord A=0 B=1 C=2, -1 = ignore in the loss.
    ICM/TE graded at EXP 1-2 contradict Gardner and are set to -1 (EXP is kept)."""
    m["EXP_ord"] = (pd.to_numeric(m["EXP"], errors="coerce") - 1).fillna(-1).astype(int)
    for c in ("ICM", "TE"):
        bad = m["gardner_issues"].fillna("").str.contains(f"{c}_graded_at_EXP<=2", regex=False)
        m[f"{c}_ord"] = m[c].map({q: i for i, q in enumerate(Q)}).where(~bad).fillna(-1).astype(int)
    for c in CRITS: m[f"{c}_valid"] = m[f"{c}_ord"] >= 0
    return m


def count_table(m, crit):
    d = m[m["split"].isin(["train", "test"])]
    v = d[crit].astype(object).map(lambda x: "missing" if pd.isna(x) else int(x) if crit == "EXP" else x)
    ct = pd.crosstab(v, d["split"]).reindex(columns=["train", "test"], fill_value=0)
    order = sorted(i for i in ct.index if i != "missing") if crit == "EXP" else Q + [ND]
    ct = ct.reindex([o for o in order if o in ct.index] + [i for i in ct.index if i not in order]).fillna(0).astype(int)
    for s in ("train", "test"): ct[f"{s}_%"] = (100 * ct[s] / max(ct[s].sum(), 1)).round(2)
    ct.index.name = crit
    return ct


def imbalance_summary(m, thr):
    tr, rows = m[m["fold"] >= 0], []
    for c in CRITS:
        v = tr.loc[tr[f"{c}_valid"], c].astype(str).value_counts(); e = len(v) == 0
        rows.append({"criterion": c, "n_valid_train": int(v.sum()), "n_classes": len(v),
                     "majority": "" if e else f"{v.idxmax()} ({v.max()})", "minority": "" if e else f"{v.idxmin()} ({v.min()})",
                     "imbalance_ratio": np.nan if e else round(v.max() / max(v.min(), 1), 1),
                     "rare_classes(<thr)": ", ".join(f"{k}={n}" for k, n in v.items() if n < thr)})
    return pd.DataFrame(rows).set_index("criterion")


def joint_table(m, split):
    d = m[m["split"] == split]
    key = d["EXP"].astype("string").fillna("?") + "-" + d["ICM"].astype("string").fillna("?") + "-" + \
        d["TE"].astype("string").fillna("?")
    out = key.value_counts().rename("count").to_frame()
    out["%"] = (100 * out["count"] / max(len(d), 1)).round(2); out.index.name = "EXP-ICM-TE"
    return out


def silver_gold_agreement(m, tabs):
    """Silver (single expert) vs gold (consortium) on the test images: accuracy, kappa, QWK, confusion matrices."""
    t, rows, cms = m[m["split"] == "test"], [], {}
    kap = lambda x, y, **kw: float(cohen_kappa_score(x, y, **kw)) if len(set(x) | set(y)) > 1 else np.nan  # noqa
    for c in [c for c in CRITS if f"{c}_silver" in t and f"{c}_gold" in t]:
        d = t[[f"{c}_silver", f"{c}_gold"]].dropna()
        if not len(d): continue
        a, b = d.iloc[:, 0].astype(str), d.iloc[:, 1].astype(str)
        labels = sorted(set(a) | set(b), key=lambda x: (x not in Q + [str(i) for i in range(1, 7)], x))
        cms[c] = pd.DataFrame(confusion_matrix(b, a, labels=labels), index=pd.Index(labels, name="gold"),
                              columns=pd.Index(labels, name="silver"))
        cms[c].to_csv(tabs / f"silver_vs_gold_confusion_{c}.csv")
        if c == "EXP": oa, ob = a.astype(float).astype(int), b.astype(float).astype(int)
        else:
            keep, enc = a.isin(Q) & b.isin(Q), {q: i for i, q in enumerate(Q)}
            oa, ob = a[keep].map(enc), b[keep].map(enc)
        rows.append({"criterion": c, "n_pairs": len(d), "accuracy": float((a == b).mean()), "cohen_kappa": kap(b, a),
                     "quadratic_weighted_kappa": kap(ob, oa, weights="quadratic"), "n_pairs_for_qwk": len(oa)})
    return (pd.DataFrame(rows).set_index("criterion") if rows else pd.DataFrame()), cms


# ---------------------------------------------------------------- cross-validation on the silver set
def stratification_key(d, k):
    """Joint EXP-ICM-TE key, collapsed hierarchically when a stratum has fewer than k members."""
    e, i, t = (d[c].astype("string").fillna("?") for c in CRITS)
    key = e + "|" + i + "|" + t
    for coarser in (e + "|" + i, e + "|" + t, e):
        small = key.map(key.value_counts()) < k; key[small] = coarser[small]
    key[key.map(key.value_counts()) < k] = "rare"
    if 0 < (key == "rare").sum() < k: key[key == "rare"] = key[key != "rare"].mode().iloc[0]
    return key


def assign_folds(m, cfg):
    m["fold"] = np.select([m["split"] == "test", m["split"] == "train"], [FOLD_TEST, FOLD_EXCLUDED], FOLD_UNLABELED)
    d = m[(m["split"] == "train") & ~m["exclude_from_train"] & m["EXP_valid"]]
    strat = stratification_key(d, cfg.n_folds); m.loc[d.index, "strat_key"] = strat
    sgkf = StratifiedGroupKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed)
    for f, (_, va) in enumerate(sgkf.split(np.zeros(len(d)), strat.to_numpy(), d["group_id"].to_numpy())):
        m.loc[d.index[va], "fold"] = f
    tr = m[m["fold"] >= 0]
    absent = {f"fold{f}.{c}": [q for q in Q if not (tr.loc[tr["fold"] == f, c] == q).any()]
              for f in range(cfg.n_folds) for c in ("ICM", "TE")}
    chk = {"n_trainable": len(d), "n_groups": int(d["group_id"].nunique()),
           "max_folds_per_group": int(tr.groupby("group_id")["fold"].nunique().max()),
           "fold_sizes": tr["fold"].value_counts().sort_index().to_dict(), "test_in_folds": int((tr["split"] == "test").sum()),
           "classes_absent_in_val_folds": {k: v for k, v in absent.items() if v}}
    chk["group_leakage_free"] = chk["max_folds_per_group"] == 1
    return m, chk


def fold_distribution(m, k):
    tr, out = m[m["fold"] >= 0], []
    for c in CRITS:
        v = tr[tr[f"{c}_valid"]]; overall = v[c].astype(str).value_counts(normalize=True)
        for f in range(k):
            vf = v.loc[v["fold"] == f, c].astype(str); cnt, p = vf.value_counts(), vf.value_counts(normalize=True)
            out += [{"criterion": c, "fold": f, "class": cl, "count": int(cnt.get(cl, 0)), "prop": float(p.get(cl, 0.0)),
                     "overall_prop": float(op), "abs_dev": abs(float(p.get(cl, 0.0)) - float(op))} for cl, op in overall.items()]
    return pd.DataFrame(out)


def class_weights(labels, classes, beta=0.999):
    """Inverse-frequency and effective-number (Cui et al. 2019) class weights."""
    n = np.array([int((labels == c).sum()) for c in classes], float); ok = n > 0
    inv, eff = np.zeros_like(n), np.zeros_like(n)
    inv[ok] = n[ok].sum() / (ok.sum() * n[ok]); eff[ok] = (1 - beta) / (1 - beta ** n[ok])
    if ok.any(): eff[ok] /= eff[ok].mean()
    return {"classes": [str(c) for c in classes], "counts": n.astype(int).tolist(),
            "inverse_frequency": inv.round(4).tolist(), f"effective_number_beta{beta:.3f}": eff.round(4).tolist()}


def compute_class_weights(m, k):
    """Per fold, from that fold's TRAINING part only (+ 'all_train' for the final model)."""
    tr = m[m["fold"] >= 0]; exp_cls = sorted(int(x) for x in tr.loc[tr["EXP_valid"], "EXP"].dropna().unique())
    subsets = {**{f"fold{f}": tr[tr["fold"] != f] for f in range(k)}, "all_train": tr}
    return {name: {"EXP": class_weights(s.loc[s["EXP_valid"], "EXP"].astype(int), exp_cls),
                   **{c: class_weights(s.loc[s[f"{c}_valid"], c], Q) for c in ("ICM", "TE")}} for name, s in subsets.items()}


def get_fold_split(manifest, fold):
    """(train, val, locked_test) DataFrames for one CV fold, from a manifest DataFrame or CSV path."""
    m = pd.read_csv(manifest) if isinstance(manifest, (str, Path)) else manifest
    return tuple(p.reset_index(drop=True) for p in (m[(m["fold"] >= 0) & (m["fold"] != fold)], m[m["fold"] == fold],
                                                    m[m["fold"] == FOLD_TEST]))


# ---------------------------------------------------------------- figures
def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def plot_distributions(tables, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, c in zip(axes, CRITS):
        ct = tables[c]; x = np.arange(len(ct))
        for k, (s, col) in enumerate((("train", "#4C72B0"), ("test", "#DD8452"))):
            for b, n in zip(ax.bar(x + (k - .5) * .4, ct[f"{s}_%"], .4, label=s, color=col), ct[s]):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(n), ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, [str(i) for i in ct.index]); ax.set(title=f"{c} (% of split; numbers = counts)", ylabel="%")
        ax.legend()
    _save(fig, path)


def plot_heatmap(ct, title, path):
    fig, ax = plt.subplots(figsize=(1.1 * ct.shape[1] + 2, .7 * ct.shape[0] + 1.5))
    im, vmax = ax.imshow(ct.to_numpy(float), cmap="Blues"), (ct.to_numpy().max() if ct.size else 1)
    ax.set_xticks(range(ct.shape[1]), [str(c) for c in ct.columns]); ax.set_yticks(range(ct.shape[0]), [str(r) for r in ct.index])
    ax.set(xlabel=ct.columns.name or "", ylabel=ct.index.name or "", title=title)
    for r in range(ct.shape[0]):
        for c in range(ct.shape[1]):
            v = ct.iat[r, c]
            ax.text(c, r, str(v), ha="center", va="center", fontsize=8, color="white" if v > .6 * vmax else "black")
    fig.colorbar(im, ax=ax, fraction=.04); _save(fig, path)


def plot_hists(panels, path, width=4.5):
    """panels = [(values, title, threshold_line_or_None), ...]"""
    fig, axes = plt.subplots(1, len(panels), figsize=(width * len(panels), 3.6), squeeze=False)
    for ax, (v, title, vline) in zip(axes[0], panels):
        ax.hist(v, bins=60, color="#4C72B0"); ax.set_title(title, fontsize=9)
        if vline is not None: ax.axvline(vline, color="red", ls="--", label=f"threshold {vline}"); ax.legend()
    _save(fig, path)


def plot_fold_balance(fd, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, c in zip(axes, CRITS):
        d = fd[fd["criterion"] == c].pivot(index="fold", columns="class", values="prop").fillna(0); bottom = np.zeros(len(d))
        for cl in d.columns: ax.bar(d.index.astype(str), d[cl], bottom=bottom, label=str(cl)); bottom += d[cl].to_numpy()
        ax.set(title=f"{c}: class proportions per validation fold", xlabel="fold"); ax.legend(fontsize=7, ncol=3)
    _save(fig, path)


def plot_gallery(m, c, k, path, seed):
    tr = m[(m["split"] == "train") & m["readable"] & m[f"{c}_valid"]]; classes = sorted(tr[c].astype(str).unique())
    if not classes: return
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(classes), k, figsize=(1.8 * k, 1.9 * len(classes)), squeeze=False)
    for r, cl in enumerate(classes):
        pool = tr[tr[c].astype(str) == cl]; pick = pool.iloc[rng.permutation(len(pool))[:k]]
        for j, ax in enumerate(axes[r]):
            ax.axis("off")
            if j < len(pick):
                with Image.open(pick.iloc[j]["path"]) as im:
                    im = im.convert("L"); im.thumbnail((256, 256)); ax.imshow(np.asarray(im), cmap="gray")
            if j == 0: ax.set_title(f"{c}={cl} (n={len(pool)})", fontsize=8, loc="left")
    _save(fig, path)


# ---------------------------------------------------------------- QC verdict (terminal indicator)
def qc_verdict(m, audit, chk, fd, cand, agreement, cfg):
    """PASS / WARN / FAIL checklist + overall verdict + key numbers, as one text block for the terminal."""
    R, lab = [], m[m["split"] != "unlabeled"]
    n_lab, rd = max(len(lab), 1), m[m["readable"]]
    add = lambda status, name, detail: R.append((status, name, detail))  # noqa: E731
    lvl = lambda n, fail_frac=None: ("PASS" if n == 0 else "FAIL" if fail_frac is not None  # noqa: E731
                                     and n > fail_frac * n_lab else "WARN")
    has = lambda pat: lab["gardner_issues"].fillna("").str.contains(pat, regex=False)  # noqa: E731
    cod = audit["coding"]
    # labels
    inv = sum(v["n_invalid"] for v in cod.values())
    add(lvl(inv, .01), "Label codes recognised", f"{inv} unrecognised values")
    zs, zg = cod["silver.EXP"]["zero_based"], cod["gold.EXP"]["zero_based"]
    mt, me = (pd.to_numeric(lab.loc[lab["split"] == sp, "EXP"], errors="coerce").mean() for sp in ("train", "test"))
    add("PASS" if abs(mt - me) < .5 else "WARN", "EXP coding consistent silver vs gold",
        f"0-based silver={zs}, gold={zg}; mean EXP train {mt:.2f} / test {me:.2f}")
    g, nd, em = int(has("graded_at_EXP<=2").sum()), int((has("ND_at_EXP>=3") | has("missing_at_EXP>=3")).sum()), \
        int(has("EXP_missing").sum())
    add(lvl(g, .02), "Gardner: no ICM/TE grade at EXP 1-2",
        f"{g} images" + (" -> EXP coding may be shifted, see report section 2" if g > .02 * n_lab else ""))
    add(lvl(nd, .10), "Gardner: ICM/TE graded at EXP >= 3", f"{nd} images not graded (ignored in ICM/TE loss)")
    add(lvl(em), "Every label has an EXP grade", f"{em} labelled images without EXP")
    conf = sum(v["n_names_with_conflicting_labels"] for v in audit["duplicate_rows"].values())
    add(lvl(conf), "No conflicting duplicate rows in label files", f"{conf} image names")
    # images
    miss, unl = int((~lab["image_found"]).sum()), int((m["split"] == "unlabeled").sum())
    add(lvl(miss, .01), "Every label has an image file",
        f"{miss} missing" + (" -> check IMAGES_DIR / file names" if miss > .01 * n_lab else ""))
    add(lvl(unl), "Every image has a label", f"{unl} images without labels (used only for SSL)")
    unr = int((m["image_found"] & ~m["readable"]).sum())
    add(lvl(unr), "All image files readable", f"{unr} unreadable (excluded)")
    rev = int(rd["qc_review"].sum())
    add("PASS" if rev <= .05 * max(len(rd), 1) else "WARN", "Image quality (blur / exposure / contrast)",
        f"{rev} flagged ({rev / max(len(rd), 1):.1%}) -> tables/qc_flagged.csv")
    # duplicates / leakage
    leak = int(m["leak_with_test"].sum())
    add("PASS" if leak == 0 else "WARN" if cfg.leak_policy == "drop_train" else "FAIL", "No train image duplicates a test image",
        f"{leak} found" + (" (auto-excluded from training and SSL)" if leak and cfg.leak_policy == "drop_train" else ""))
    amb = int(cand["texture_ncc"].between(.75, cfg.ncc_threshold, inclusive="left").sum()) if len(cand) else 0
    add(lvl(amb), "Duplicate detection clear-cut", f"{amb} borderline pairs -> figures/duplicate_ncc_histogram.png")
    src, grp = m["group_source"].iloc[0], audit.get("grouping", {})
    ev = grp.get("prefix_evidence")
    if ev:
        s = ev["acquisition_icc"]
        add("PASS" if ev["supported"] else "WARN", "File-name prefix behaves like a patient/cycle ID",
            f"embryo no. 1..n in {ev['embryo_numbers_run_1_to_n']:.1%} of prefixes; ICC " +
            ", ".join(f"{k} {v['icc']:.2f} (p={v['p_perm']:.3f})" for k, v in s.items()))
    ok_grp = src.startswith(("patient", "cycle")) or (src.startswith("file-name prefix") and bool(ev and ev["supported"]))
    add("PASS" if ok_grp else "WARN", "Patient/cycle-level CV grouping", src)
    nsib, pol = chk.get("test_with_train_sibling", 0), cfg.test_siblings
    add("PASS" if nsib == 0 else "INFO", "Gold-test embryos with a sibling in training",
        f"{nsib} (policy '{pol}'" + ("; official split kept, Phase 9 reports with/without-sibling results)" if nsib else ")"))
    # folds
    sx = chk.get("sibling_keys_across_folds", 0)
    add("PASS" if chk["group_leakage_free"] and chk["test_in_folds"] == 0 and sx == 0 else "FAIL", "CV folds leakage-free",
        f"{chk['n_groups']} groups, max folds/group = {chk['max_folds_per_group']}, test in folds = {chk['test_in_folds']}, "
        f"patient/cycle keys across folds = {sx}")
    ab = chk["classes_absent_in_val_folds"]
    add("PASS" if not ab else "WARN", "Every ICM/TE class in every validation fold", str(ab) if ab else "yes")
    dev = float(fd["abs_dev"].max()) if len(fd) else 0.0
    add("PASS" if dev <= .05 else "WARN", "Folds class-balanced", f"max class-proportion deviation {dev:.3f}")
    n_img, n_test = int(m["image_found"].sum()), int((m["split"] == "test").sum())
    add("PASS" if (n_img, n_test) == (2344, 300) else "WARN", "Counts match the paper (2,344 images, 300 test)",
        f"{n_img} images, {n_test} test")
    if len(agreement):
        add("INFO", "Silver vs gold agreement on test (QWK)",
            ", ".join(f"{c}={v:.2f}" for c, v in agreement["quadratic_weighted_kappa"].items()))

    st = Counter(s for s, _, _ in R)
    overall = "FAIL" if st["FAIL"] else "PASS WITH WARNINGS" if st["WARN"] else "PASS"
    nxt = {"FAIL": "Fix the FAIL items before Phase 2 (details in phase1_report.md).",
           "PASS WITH WARNINGS": "OK to continue to Phase 2; WARN items are documented in phase1_report.md.",
           "PASS": "Ready for Phase 2."}[overall]
    tr = m[m["fold"] >= 0]
    dist = lambda c: dict(sorted(Counter(tr.loc[tr[f"{c}_valid"], c].astype(str)).items()))  # noqa: E731
    code = lambda t: f"EXP 0-based={cod[f'{t}.EXP']['zero_based']} | " + " | ".join(  # noqa: E731
        f"{c} {cod[f'{t}.{c}']['numeric_map'] or 'letters'}" for c in ("ICM", "TE"))
    W, bar = 100, "=" * 100
    L = [bar, f" PHASE 1 QC VERDICT:  {overall}    (PASS {st['PASS']}, WARN {st['WARN']}, FAIL {st['FAIL']})", bar]
    L += [f" [{s}] {n + ' ':.<52} {d}" for s, n, d in R]
    L += ["-" * W, " Key numbers",
          f"  images on disk {n_img} | train {int((m['split'] == 'train').sum())} | test {n_test} | unlabelled {unl}"
          f" | trainable in folds {len(tr)} | SSL-allowed {int(m['ssl_allowed'].sum())}",
          f"  fold sizes {chk['fold_sizes']} | CV groups {chk['n_groups']} ({src})",
          f"  test embryos with a training sibling {chk.get('test_with_train_sibling', 0)} | training images sibling of a test "
          f"embryo {grp.get('train_images_sibling_of_test', 0)} (policy '{cfg.test_siblings}')",
          f"  CV-pool classes  EXP {dist('EXP')} | ICM {dist('ICM')} | TE {dist('TE')}",
          f"  columns  silver {audit['columns']['silver']} | gold {audit['columns']['gold']}",
          f"  coding   silver: {code('silver')}", f"           gold:   {code('gold')}",
          "-" * W, f" Next: {nxt}",
          f" Results folder: {Path(cfg.out_dir).resolve()}",
          " Copy everything between the ==== lines and send it back for review.", bar]
    return "\n".join(L), overall


# ---------------------------------------------------------------- report
def write_report(path, x):
    cfg, audit = x["cfg"], x["audit"]
    J = lambda o: "```json\n" + json.dumps(o, indent=2, default=_jd) + "\n```\n"  # noqa: E731
    L = ["# MorphoGround - Phase 1 report: dataset audit, QC and leakage-safe splits\n",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} | seed={cfg.seed} | folds={cfg.n_folds} | hash threshold="
         f"{cfg.hash_threshold} | NCC threshold={cfg.ncc_threshold} | leak policy={cfg.leak_policy}\n",
         "## 0. QC verdict\n", "```\n" + x["verdict"] + "\n```\n",
         "## 1. Inventory\n", md_table(x["inventory"], index=False),
         "## 2. Column detection and label coding (verify before Phase 2)\n", J(audit["columns"])]
    L += [f"- **{k}**: numeric map / zero-based = `{v.get('numeric_map', v.get('zero_based'))}`, invalid = "
          f"{v['n_invalid']}, raw values = `{v['raw_value_counts']}`" for k, v in audit["coding"].items()]
    L.append("\nRaw ICM/TE codes against normalised EXP. If the coding is right, the code mapped to `ND` should appear "
             "only in EXP 1-2 columns.\n")
    for name, ct in x["raw_ct"].items(): L += [f"**{name}**\n", md_table(ct)]
    L += ["## 3. Label integrity (Gardner hierarchy)\n", md_table(x["issues"]) if len(x["issues"]) else "No hierarchy violations.\n",
          "\nDuplicate rows inside label files:\n", J(audit["duplicate_rows"]),
          "## 4. Image quality control\n", md_table(x["qc_summary"], index=False),
          "\nFlags are for human review. Only unreadable images are excluded automatically. See `tables/qc_flagged.csv` "
          "and `figures/qc_histograms.png`.\n", "## 5. Duplicates and leakage\n", md_table(x["dup_summary"], index=False),
          f"\nGrouping used for cross-validation: **{x['group_source']}**.\n",
          "Patient/cycle key evidence and sibling policy:\n", J(audit.get("grouping", {})),
          "Check `figures/duplicate_ncc_histogram.png` and `tables/duplicate_candidates_scored.csv`; if the verified pairs "
          "are not clearly separated from the rest, adjust `ncc_threshold` after looking at a few pairs by eye.\n",
          "## 6. Label distribution and imbalance\n"]
    for c in CRITS: L += [f"**{c}**\n", md_table(x["count_tables"][c])]
    L += ["**Imbalance summary (CV pool = trainable silver images, valid labels only)**\n", md_table(x["imbalance"]),
          "**Joint EXP-ICM-TE grades (train, top 25)**\n", md_table(x["joint_train"], max_rows=25),
          "## 7. Silver (single expert) vs gold (consortium) agreement on the test images\n"]
    L += ([md_table(x["agreement"]), "\nThis estimates silver-label noise: a model trained on silver labels cannot be "
           "expected to agree with gold labels much beyond the silver annotator.\n"] if len(x["agreement"]) else
          ["Silver labels for the test images are not available (set COMPLETE_FILE to the authors' complete.csv).\n"])
    L += ["## 8. Cross-validation folds (silver set)\n", J(x["fold_checks"]), md_table(x["fold_dev"]),
          "## 9. Notes for the paper and for later phases\n"] + [f"- {n}" for n in x["notes"]]
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------- main pipeline
def run_phase1(cfg: Phase1Config = None) -> pd.DataFrame:
    cfg = cfg or Phase1Config()
    t0, out = time.time(), Path(cfg.out_dir); figs, tabs = out / "figures", out / "tables"
    for name, p in (("IMAGES_DIR", cfg.images_dir), ("SILVER_FILE", cfg.silver_path), ("GOLD_FILE", cfg.gold_path),
                    ("COMPLETE_FILE", cfg.complete_path), ("CLINICAL_FILE", cfg.clinical_path)):
        if p and not Path(p).exists(): raise FileNotFoundError(f"{name} not found: {p}\n-> edit USER SETTINGS at the top")
    for d in (figs, tabs): d.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO); LOG.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(out / "phase1.log", mode="w")):
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")); LOG.addHandler(h)
    audit, notes = {"inputs": {}}, []

    # 1. annotation files -> one label table (gold = test, silver = train)
    LOG.info(f"Reading annotation files; results -> {out.resolve()}")
    s_raw, audit["inputs"]["silver"] = read_table(cfg.silver_path, cfg.silver_sheet)
    g_raw, audit["inputs"]["gold"] = read_table(cfg.gold_path, cfg.gold_sheet)
    audit["columns"] = {"silver": detect_columns(s_raw, ("silver",), cfg.silver_cols, "silver"),
                        "gold": detect_columns(g_raw, ("gold", "cons", "consensus", "majority"), cfg.gold_cols, "gold")}
    lab = standardize(s_raw, audit["columns"]["silver"], "silver", cfg, audit).merge(
        standardize(g_raw, audit["columns"]["gold"], "gold", cfg, audit), on="image_key", how="outer")
    for t, c in audit["columns"].items(): LOG.info(f"{t} columns: {c}")
    for k, v in audit["coding"].items(): LOG.info(f"coding {k}: {v.get('numeric_map', v.get('zero_based'))}, invalid={v['n_invalid']}")
    lab["in_silver_file"], lab["in_gold_file"] = lab["image_name_silver"].notna(), lab["image_name_gold"].notna()
    n_overlap = int((lab["in_silver_file"] & lab["in_gold_file"]).sum())
    if cfg.complete_path:
        comp = standardize(read_complete_csv(cfg.complete_path), {r: r for r in REQUIRED}, "complete", cfg, audit)
        lab = lab.merge(comp[["image_key"] + [f"{c}_complete" for c in CRITS]], on="image_key", how="left")
        ov = lab[lab["in_silver_file"] & lab["EXP_complete"].notna()]
        mism = audit["complete_vs_silver_mismatch"] = {
            c: int((ov[f"{c}_silver"].map(label_str) != ov[f"{c}_complete"].map(label_str)).sum()) for c in CRITS}
        if any(mism.values()): notes.append(f"complete.csv disagrees with the silver file on {mism} images - check the "
                                            "coding, or these are genuine label differences.")
        for c in CRITS: lab[f"{c}_silver"] = lab[f"{c}_silver"].combine_first(lab[f"{c}_complete"])
    lab["image_name"] = lab["image_name_gold"].combine_first(lab["image_name_silver"])
    lab["split"] = np.where(lab["in_gold_file"], "test", "train")
    lab["label_source"] = np.where(lab["in_gold_file"], "gold_consensus", "silver_expert")
    for c in CRITS: lab[c] = lab[f"{c}_gold"].where(lab["in_gold_file"], lab[f"{c}_silver"])
    for r in OPTIONAL:
        cols = [f"{r}_id_{t}" for t in ("gold", "silver") if f"{r}_id_{t}" in lab]
        if cols: lab[f"{r}_id"] = lab[cols].bfill(axis=1).iloc[:, 0]

    # 2. image files
    idx = index_images(cfg.images_dir)
    if not idx: raise FileNotFoundError(f"No images ({', '.join(sorted(IMG_EXTS))}) found under {cfg.images_dir}")
    m = lab.merge(pd.DataFrame({"image_key": list(idx), "path": [v[0] for v in idx.values()],
                                "n_files_same_name": [len(v) for v in idx.values()]}), on="image_key", how="outer")
    m["image_found"], m["split"] = m["path"].notna(), m["split"].fillna("unlabeled")
    m["image_name"] = m["image_name"].fillna(m["path"].map(lambda p: Path(p).name if isinstance(p, str) else p))
    for c in ("in_silver_file", "in_gold_file"): m[c] = m[c].eq(True)
    n_missing = int(((m["split"] != "unlabeled") & ~m["image_found"]).sum())
    if n_missing: LOG.warning(f"{n_missing} labelled images have no image file")

    # 3. image QC
    LOG.info(f"Running QC on {int(m['image_found'].sum())} image files under {cfg.images_dir}")
    qc = add_qc_flags(run_qc(m.loc[m["image_found"], "path"].tolist(), cfg.num_workers), cfg)
    m = m.merge(qc, on="path", how="left").reset_index(drop=True)
    m["readable"], m["qc_review"] = m["readable"].eq(True), m["qc_review"].eq(True)
    m["qc_flags"] = m["qc_flags"].where(m["image_found"], "image_missing").fillna("")

    # 4. duplicates and leakage
    roots, pairs, cand = find_duplicates(m, cfg)
    cand.sort_values("texture_ncc", ascending=False).to_csv(tabs / "duplicate_candidates_scored.csv", index=False)
    m["dup_root"] = roots; m["dup_group_size"] = m.groupby("dup_root")["image_key"].transform("size")
    m["dup_group"] = np.where(m["dup_group_size"] > 1, "D" + m["dup_root"].astype(str), "")
    has_test = m.groupby("dup_root")["split"].agg(lambda s: "test" in set(s))
    m["leak_with_test"] = m["dup_root"].map(has_test).astype(bool) & (m["split"] != "test")
    if len(pairs):
        for side in "ab":
            for c in ("image_key", "split"): pairs[f"{c}_{side}"] = m.loc[pairs[f"row_{side}"], c].to_numpy()
            for c in CRITS: pairs[f"{c}_{side}"] = [label_str(v) for v in m.loc[pairs[f"row_{side}"], c]]
        same = {c: pairs[f"{c}_a"] == pairs[f"{c}_b"] for c in CRITS}      # v2: '4.0' vs '4' no longer counts as different
        pairs["labels_agree"] = np.logical_and.reduce(list(same.values()))
        pairs["differing_criteria"] = ["|".join(c for c in CRITS if not same[c].iloc[i]) for i in range(len(pairs))]
        pairs["cross_split"] = pairs["split_a"] != pairs["split_b"]
    pairs.drop(columns=["row_a", "row_b"]).to_csv(tabs / "duplicate_pairs.csv", index=False)

    # 5. CV groups = patient/cycle (ID column, else file-name prefix) united with duplicate clusters
    uf, group_source, sib_col, evidence = UnionFind(len(m)), "duplicate clusters only (no patient/cycle key)", None, None
    m["prefix_id"] = m["image_key"].astype(str).str.extract(cfg.prefix_regex)[0]
    labelled = m["split"] != "unlabeled"
    id_col = next((c for c in ("patient_id", "cycle_id") if c in m and m.loc[m["split"] == "train", c].notna().mean() > .95), None)
    pref_cov = float(m.loc[labelled, "prefix_id"].notna().mean()) if labelled.any() else 0.0
    if cfg.group_by not in ("auto", "prefix", "duplicates"): raise ValueError(f"GROUP_BY must be auto, prefix or duplicates, not {cfg.group_by!r}")
    if cfg.group_by == "prefix" and pref_cov < .95:
        raise ValueError(f"GROUP_BY='prefix' but only {pref_cov:.0%} of image names match PREFIX_REGEX {cfg.prefix_regex!r}")
    if cfg.group_by == "auto" and id_col: sib_col, group_source = id_col, f"{id_col} + duplicate clusters"
    elif cfg.group_by in ("auto", "prefix") and pref_cov >= .95:
        sib_col, group_source = "prefix_id", "file-name prefix (patient/cycle) + duplicate clusters"
    if sib_col:
        uf.union_groups(m[m[sib_col].notna()], sib_col)
        if pref_cov >= .95:
            evidence = prefix_evidence(m, "prefix_id", cfg.n_permutations, cfg.seed)
            s = evidence["acquisition_icc"]
            LOG.info(f"Prefix = patient/cycle? embryo numbers 1..n in {evidence['embryo_numbers_run_1_to_n']:.1%} of prefixes; "
                     + ", ".join(f"ICC {k} {v['icc']:.2f} (p={v['p_perm']:.3f})" for k, v in s.items()))
    uf.union_groups(m, "dup_root")
    m["group_id"], m["group_source"] = ["G%05d" % uf.find(i) for i in range(len(m))], group_source
    has_test = m.groupby("group_id")["split"].agg(lambda s: "test" in set(s))
    m["sibling_of_test"] = m["group_id"].map(has_test).astype(bool) & (m["split"] != "test")   # same patient/cycle as a test embryo
    audit["grouping"] = {"source": group_source, "sibling_key": sib_col, "prefix_coverage": pref_cov, "prefix_evidence": evidence,
                         "test_siblings_policy": cfg.test_siblings,
                         "train_images_sibling_of_test": int((m["sibling_of_test"] & (m["split"] == "train")).sum())}

    # 6. label integrity, training targets, exclusion / SSL flags
    lbl = m["split"] != "unlabeled"
    m["gardner_issues"] = ""
    m.loc[lbl, "gardner_issues"] = gardner_issues(m.loc[lbl, "EXP"], m.loc[lbl, "ICM"], m.loc[lbl, "TE"])
    m = ordinal_encode(m)
    cnt = Counter((t, k) for t in ("silver", "gold") for s in gardner_issues(*(m.loc[m[f"in_{t}_file"], f"{c}_{t}"]
                                                                              for c in CRITS)) for k in s.split(";") if k)
    issues = pd.DataFrame([{"file": f, "issue": i, "n": n} for (f, i), n in cnt.items()]).set_index(["file", "issue"]) \
        if cnt else pd.DataFrame()
    if cfg.test_siblings not in ("keep", "exclude"): raise ValueError(f"TEST_SIBLINGS must be keep or exclude, not {cfg.test_siblings!r}")
    drop_sib = cfg.test_siblings == "exclude"
    m["exclude_from_train"] = (m["split"] == "train") & (~m["readable"] | (m["leak_with_test"] & (cfg.leak_policy == "drop_train"))
                                                        | (m["sibling_of_test"] & drop_sib))
    m["ssl_allowed"] = m["split"].isin(["train", "unlabeled"]) & m["readable"] & ~m["leak_with_test"] & ~(m["sibling_of_test"] & drop_sib)

    # 7. folds
    m, fold_checks = assign_folds(m, cfg)
    if not fold_checks["group_leakage_free"]: raise RuntimeError("A group appears in more than one fold - split is not leakage-free")
    cv_groups = set(m.loc[m["fold"] >= 0, "group_id"])
    m["test_has_train_sibling"] = (m["split"] == "test") & m["group_id"].isin(cv_groups)
    if sib_col:
        cvk = m[(m["fold"] >= 0) & m[sib_col].notna()]
        fold_checks["sibling_keys_across_folds"] = int((cvk.groupby(sib_col)["fold"].nunique() > 1).sum())
    fold_checks["test_with_train_sibling"] = int(m["test_has_train_sibling"].sum())
    LOG.info(f"{cfg.n_folds}-fold StratifiedGroupKFold sizes: {fold_checks['fold_sizes']}")
    fd = fold_distribution(m, cfg.n_folds); fd.to_csv(tabs / "fold_class_distribution.csv", index=False)
    fold_dev = fd.groupby("criterion")["abs_dev"].agg(mean_abs_prop_dev="mean", max_abs_prop_dev="max")

    # 8. optional clinical file
    if cfg.clinical_path:
        c_raw, audit["inputs"]["clinical"] = read_table(cfg.clinical_path)
        c_img = next((c for c in c_raw.columns if _tok(c) & ROLES["image"]), c_raw.columns[0])
        m = m.merge(c_raw.add_prefix("clin__").assign(image_key=c_raw[c_img].map(image_key)).drop_duplicates("image_key"),
                    on="image_key", how="left")
        m["in_clinical_subset"] = m[f"clin__{c_img}"].notna(); audit["clinical_matched"] = int(m["in_clinical_subset"].sum())

    # 9. tables
    tr, miss = m[m["fold"] >= 0], lambda s: s.astype(object).where(s.notna(), "missing")
    count_tables = {c: count_table(m, c) for c in CRITS}
    imbalance, joint_train = imbalance_summary(m, cfg.rare_class_threshold), joint_table(m, "train")
    agreement, cms = silver_gold_agreement(m, tabs)
    xt = {"EXP_x_ICM": pd.crosstab(tr["EXP"], miss(tr["ICM"])), "EXP_x_TE": pd.crosstab(tr["EXP"], miss(tr["TE"])),
          "ICM_x_TE": pd.crosstab(miss(tr["ICM"]), miss(tr["TE"]))}
    raw_ct = {}
    for t in ("silver", "gold"):
        sub = m[m[f"in_{t}_file"]]
        for c in ("ICM", "TE"):
            ct = raw_ct[f"{t}: raw {c} x EXP"] = pd.crosstab(sub[f"{c}_raw_{t}"].fillna("<empty>"),
                                                             sub[f"EXP_{t}"].astype("string").fillna("?"))
            ct.index.name, ct.columns.name = f"{c} raw ({t})", "EXP"; ct.to_csv(tabs / f"rawcode_{c}_x_EXP_{t}.csv")
    saves = {**{f"distribution_{c}": t for c, t in count_tables.items()}, **{f"crosstab_train_{k}": v for k, v in xt.items()},
             "imbalance_summary": imbalance, "joint_grades_train": joint_train, "joint_grades_test": joint_table(m, "test")}
    if len(agreement): saves["silver_vs_gold_agreement"] = agreement
    for name, t in saves.items(): t.to_csv(tabs / f"{name}.csv")
    m.loc[m["qc_flags"] != "", ["image_key", "split", "path", "qc_flags", "mean_intensity", "std_intensity",
                                "laplacian_var", "frac_high_sat"]].to_csv(tabs / "qc_flagged.csv", index=False)
    qc_summary = pd.DataFrame(sorted(Counter(f for s in m["qc_flags"] for f in s.split(";") if f).items()) or [("none", 0)],
                              columns=["flag", "n_images"])
    P = lambda t: int((pairs["type"] == t).sum()) if len(pairs) else 0  # noqa: E731
    dup_summary = pd.DataFrame([
        ("exact duplicate pairs (MD5)", P("exact_md5")), ("hash candidate pairs checked", len(cand)),
        ("near-duplicate pairs (hash + texture NCC verified)", P("near_verified")),
        ("duplicate clusters (size > 1)", int(m.loc[m["dup_group_size"] > 1, "dup_root"].nunique())),
        ("pairs crossing train/test/unlabeled", int(pairs["cross_split"].sum()) if len(pairs) else 0),
        ("non-test images duplicating a test image", int(m["leak_with_test"].sum())),
        ("labelled duplicate pairs with different labels", int((~pairs["labels_agree"] & (pairs["split_a"] != "unlabeled")
                                                               & (pairs["split_b"] != "unlabeled")).sum()) if len(pairs) else 0),
        ("CV groups", int(tr["group_id"].nunique()))], columns=["item", "value"])
    rd, has = m[m["readable"]], lambda c: c in m and m[c].notna().any()
    nv = lambda c: f"{int(m.loc[m['split'] == 'train', c].sum())} / {int(m.loc[m['split'] == 'test', c].sum())}"  # noqa
    inventory = pd.DataFrame([
        ("Image files on disk", int(m["image_found"].sum())),
        ("Image names matching >1 file", int((m["n_files_same_name"].fillna(1) > 1).sum())),
        ("Silver-labelled (train)", int((m["split"] == "train").sum())), ("Gold-labelled (locked test)", int((m["split"] == "test").sum())),
        ("Images in both label files", n_overlap), ("Label rows without image file", n_missing),
        ("Images without labels (SSL only)", int((m["split"] == "unlabeled").sum())),
        ("Unreadable images", int((m["image_found"] & ~m["readable"]).sum())),
        ("Patients", f"{m['patient_id'].nunique()} IDs" if has("patient_id") else "no ID column (paper: 837 patients)"),
        ("Embryos / images per embryo", f"{m['embryo_id'].nunique()} IDs" if has("embryo_id") else "no ID column (paper: 1 image per blastocyst)"),
        ("EXP labelled (train/test)", nv("EXP_valid")), ("ICM graded A-C (train/test)", nv("ICM_valid")),
        ("TE graded A-C (train/test)", nv("TE_valid")), ("Clinical subset matched", audit.get("clinical_matched", "not loaded")),
        ("Image size (modal)", f"{int(rd['width'].mode()[0])}x{int(rd['height'].mode()[0])}" if len(rd) else "-"),
        ("PIL modes", dict(Counter(rd["mode"])) if len(rd) else "-"),
        ("Colour content", f"{int(rd['grayscale_content'].eq(True).sum())} effectively grayscale" if len(rd) else "-"),
        ("SSL-allowed images", int(m["ssl_allowed"].sum())), ("Trainable silver images in folds", len(tr))],
        columns=["component", "value"])
    inventory.to_csv(tabs / "inventory.csv", index=False)

    # 10. notes
    if group_source.startswith("duplicate"):
        notes.append("No patient/cycle key (no ID column, file names do not match PREFIX_REGEX). CV groups guard against "
                     "duplicate images only; sibling embryos of one patient can still sit in different folds.")
    elif sib_col == "prefix_id":
        ev = audit["grouping"]["prefix_evidence"] or {}
        notes.append(f"CV groups = file-name prefix (treated as patient/cycle). Evidence: embryo numbers run 1..n inside "
                     f"{ev.get('embryo_numbers_run_1_to_n', float('nan')):.1%} of prefixes; acquisition statistics cluster by prefix ("
                     + ", ".join(f"{k} ICC {v['icc']:.2f}, permutation p={v['p_perm']:.3f}" for k, v in ev.get("acquisition_icc", {}).items())
                     + "). Ask the dataset authors to confirm the prefix meaning for the paper.")
    nsib = int(m["test_has_train_sibling"].sum())
    if nsib:
        notes.append(f"{nsib}/{int((m['split'] == 'test').sum())} gold-test embryos share a patient/cycle with training images "
                     f"(policy '{cfg.test_siblings}'). The official split is kept for comparability; Phase 9 reports gold-test "
                     "results separately for embryos with and without a training sibling. TEST_SIBLINGS='exclude' gives the "
                     "sibling-free protocol (fewer training images).")
    if m["leak_with_test"].any():
        notes.append(f"{int(m['leak_with_test'].sum())} non-test images duplicate a gold test image (policy "
                     f"'{cfg.leak_policy}'); they are also barred from SSL pretraining.")
    if len(issues): notes.append("Gardner hierarchy violations exist - inspect `gardner_issues` in manifest.csv. ICM/TE "
                                 "graded at EXP 1-2 are set to ignore (*_ord = -1); EXP is kept.")
    if fold_checks["classes_absent_in_val_folds"]:
        notes.append(f"Some validation folds miss a class: {fold_checks['classes_absent_in_val_folds']}. Use macro metrics "
                     "with care; consider pooling out-of-fold predictions.")
    notes += ["The gold test set is locked: never use it for early stopping, calibration, conformal thresholds or "
              "MAE/DINO pretraining.", "ICM/TE losses must ignore targets with *_ord = -1 (ND at EXP 1-2, or missing)."]

    # 11. figures
    LOG.info("Writing figures")
    plot_distributions(count_tables, figs / "label_distribution.png")
    for k, v in xt.items():
        if v.size: plot_heatmap(v, f"Train (CV pool): {k.replace('_x_', ' x ')}", figs / f"heatmap_{k}.png")
    for c, cm in cms.items(): plot_heatmap(cm, f"Test images: gold (rows) vs silver (cols) - {c}", figs / f"silver_vs_gold_{c}.png")
    r = qc[qc["readable"].eq(True)]
    plot_hists([(r["mean_intensity"], "Mean intensity", None), (r["std_intensity"], "Contrast (std)", None),
                (np.log1p(r["laplacian_var"]), "Sharpness: log1p(Laplacian var)", None),
                (r["frac_high_sat"], "Fraction saturated (>=250)", None)], figs / "qc_histograms.png")
    if len(cand): plot_hists([(cand["texture_ncc"], "Near-duplicate verification: band-passed texture NCC (true duplicates "
                               "should form a separate mode near 1)", cfg.ncc_threshold)], figs / "duplicate_ncc_histogram.png", 8)
    plot_fold_balance(fd, figs / "fold_balance.png")
    for c in CRITS:
        try: plot_gallery(m, c, cfg.gallery_per_class, figs / f"gallery_{c}.png", cfg.seed)
        except Exception as e: LOG.warning(f"gallery {c} failed: {e}")  # noqa: BLE001

    # 12. save manifest, class weights, run info, report
    (out / "class_weights.json").write_text(json.dumps(compute_class_weights(m, cfg.n_folds), indent=2, default=_jd))
    front = ["image_key", "image_name", "path", "split", "fold", "group_id", "strat_key", "EXP", "ICM", "TE", "EXP_ord",
             "ICM_ord", "TE_ord", "EXP_valid", "ICM_valid", "TE_valid", "label_source", "EXP_silver", "ICM_silver",
             "TE_silver", "EXP_gold", "ICM_gold", "TE_gold", "exclude_from_train", "ssl_allowed", "leak_with_test",
             "prefix_id", "sibling_of_test", "test_has_train_sibling",
             "dup_group", "dup_group_size", "gardner_issues", "qc_flags", "qc_review", "image_found", "readable"]
    front = [c for c in front if c in m]
    m = m[front + [c for c in m.columns if c not in front and c not in {"dup_root", "image_name_silver", "image_name_gold"}]]
    m = m.sort_values(["split", "fold", "image_key"]).reset_index(drop=True)
    m.to_csv(out / "manifest.csv", index=False)
    (out / "phase1_run_info.json").write_text(json.dumps(
        {"config": asdict(cfg), "runtime_sec": round(time.time() - t0, 1), "audit": audit, "fold_checks": fold_checks,
         "group_source": group_source, "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                                   "pandas": pd.__version__, "pillow": Image.__version__,
                                                   "scikit-learn": __import__("sklearn").__version__}}, indent=2, default=_jd))
    verdict, overall = qc_verdict(m, audit, fold_checks, fd, cand, agreement, cfg)
    (out / "QC_VERDICT.txt").write_text(verdict + "\n", encoding="utf-8")
    write_report(out / "phase1_report.md", dict(
        cfg=cfg, audit=audit, inventory=inventory, raw_ct=raw_ct, issues=issues, qc_summary=qc_summary,
        dup_summary=dup_summary, group_source=group_source, count_tables=count_tables, imbalance=imbalance,
        joint_train=joint_train, agreement=agreement, fold_checks=fold_checks, fold_dev=fold_dev, notes=notes,
        verdict=verdict))
    LOG.info(f"Done in {time.time() - t0:.1f}s -> {out.resolve()}")
    print("\n" + verdict, flush=True)
    return m


if __name__ == "__main__":
    try:
        run_phase1()
    except Exception as err:
        import builtins, traceback
        print("\n" + "=" * 100 + f"\n PHASE 1 QC VERDICT:  DID NOT RUN\n {type(err).__name__}: {err}\n"
              " Copy this block (and the traceback below) and send it back.\n" + "=" * 100, flush=True)
        if hasattr(builtins, "get_ipython"): traceback.print_exc()   # Jupyter: plain traceback
        else: raise
