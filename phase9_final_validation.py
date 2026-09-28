"""MorphoGround - PHASE 9: final validation and paper package (CPU only, a few minutes).
Collects every phase: gold-test results of all systems with 95% bootstrap CIs, paired bootstrap + McNemar tests
(Holm-corrected), the strategy's ablation matrix, answers to research questions RQ1-RQ5, an offline expert-evaluation
page (blind grading, then rating of the model and its explanations), LaTeX tables, and a reproducibility record.
Rerun it after embryologists return their ratings (CSV files into phase9_outputs/expert_ratings/).
v2: Phase 5 v2 names (global_ordinal, tokens_ordinal); every system is checked for ordinal collapse (collapsed systems are
reported but never used in comparisons); gold-test results are split by whether the embryo has a sibling (same patient/cycle)
in training; Phase 4 grounded-vs-unconstrained results, Phase 6 gold-calibrated coverage and the Phase 7 structured
explanations feed the research-question answers; the expert page uses the Phase 6 model and the final Phase 7 texts.
Self-contained: paste into ONE Jupyter cell, or run  python phase9_final_validation.py"""
from pathlib import Path

# ====================================== USER SETTINGS ========================================
PROJECT_DIR = Path.home() / "Embryo_Updated Project"   # same folder as the earlier phases
PHASE1_DIR  = PROJECT_DIR / "phase1_outputs"
CACHE_DIR   = PROJECT_DIR / "cache"                     # image cache, adapted encoders, Phase 5 checkpoint
IMG_SIZE    = 224
QUICK_TEST  = False

# ---- PHASE 9 settings
N_BOOT      = 2000     # bootstrap resamples of the 300 gold-test embryos
ALPHA_STAT  = .05
N_EXPERT    = 40       # embryos in the expert evaluation page
SEED        = 0
OUT         = PROJECT_DIR / "phase9_outputs"
# ============================================================================================

import json, math, sys, time, warnings
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, recall_score
warnings.filterwarnings("ignore", category=UserWarning)


TASKS = ("EXP", "ICM", "TE")


CLASS_NAMES = {"EXP": ["1", "2", "3", "4", "5", "6"], "ICM": ["A", "B", "C"], "TE": ["A", "B", "C"]}


def log(msg): print(f"{time.strftime('%H:%M:%S')} | {msg}", flush=True)


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


def save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


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


# ##################################### PHASE 9 #####################################
import base64, hashlib, io, platform  # noqa: E402
from html import escape  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
import scipy, sklearn  # noqa: E402
if QUICK_TEST: N_BOOT = 200
EXPERT_DIR = OUT / "expert_ratings"
METS = ("acc", "macro_f1", "weighted_f1", "qwk")
MAIN_SYS = "P5:morphoground"   # replaced by the Phase 6 source when Phase 6 was run
VERDICT_FILES = {1: ("phase1_outputs", "QC_VERDICT.txt"), **{k: (f"phase{k}_outputs", f"PHASE{k}_VERDICT.txt") for k in range(2, 9)}}


def cm_metrics(y, p, K):
    """Fast accuracy / macro-F1 / weighted-F1 / QWK from a confusion matrix (identical definitions to sklearn)."""
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K).astype(float); n = cm.sum()
    tp, cs, rs = np.diag(cm), cm.sum(0), cm.sum(1); prec, rec = tp / np.maximum(cs, 1e-12), tp / np.maximum(rs, 1e-12)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    i, j = np.indices((K, K)); Wq = (i - j) ** 2.0; E = np.outer(rs, cs) / n
    return {"acc": tp.sum() / n, "macro_f1": f1[(cs + rs) > 0].mean(), "weighted_f1": (f1 * rs).sum() / n,
            "qwk": 1 - (Wq * cm).sum() / max((Wq * E).sum(), 1e-12)}


def load_systems(m, te):
    """Gold-test probabilities of every saved model (Phase 2 configs, Phase 5 variants, Phase 8 variants)."""
    S, bad = {}, []
    for ph, folder in (("P2", "phase2_outputs"), ("P5", "phase5_outputs"), ("P8", "phase8_outputs")):
        d = PROJECT_DIR / folder / "preds"
        for f in sorted(d.glob("*.npz")) if d.exists() else []:
            z = np.load(f, allow_pickle=True)
            if not (np.array_equal(z["test_idx"], te) and len(z["keys"]) == len(m) and (z["keys"].astype(str) == m["image_key"].astype(str).to_numpy()).all()):
                bad.append(f"{ph}:{f.stem}"); continue
            S[f"{ph}:{f.stem}"] = [z[f"test_{t}"] for t in TASKS]
            if all(f"oof_{t}" in z.files for t in TASKS): OOF[f"{ph}:{f.stem}"] = [z[f"oof_{t}"] for t in TASKS]
    return S, bad


OOF = {}


def collapse_report(y, P, min_prev=.05):
    """Ordinal-collapse guard (as Phase 5): a grade with >= 5% prevalence (almost) never predicted, or TV distance > 0.5."""
    k = (y >= 0) & np.isfinite(P).all(1); y, p, K = y[k], P[k].argmax(1), P.shape[1]
    if not len(y): return {"collapsed": False, "tv": 0.0}
    ty, tp = np.bincount(y, minlength=K) / len(y), np.bincount(p, minlength=K) / len(y)
    dead = [c for c in range(K) if ty[c] >= min_prev and (p[y == c] == c).mean() < .05]
    return {"collapsed": bool(dead) or float(.5 * np.abs(ty - tp).sum()) > .5, "tv": float(.5 * np.abs(ty - tp).sum())}


def sibling_mask(m, te):
    """True for gold-test embryos whose patient/cycle also has training images (Phase 1 v2 column, else file-name prefix)."""
    if "test_has_train_sibling" in m: return m["test_has_train_sibling"].astype(str).str.lower().eq("true").to_numpy()[te]
    pre = m["image_key"].astype(str).str.extract(r"^0*(\d+)_\d+$")[0]
    return pre.iloc[te].isin(set(pre[m["fold"] >= 0].dropna())).to_numpy()


def read_csv(folder, name, **kw):
    f = PROJECT_DIR / folder / name
    return pd.read_csv(f, **kw) if f.exists() else None


def fmt_ci(v, lo, hi): return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (len(p) - r) * p[i])); adj[i] = run
    return adj


def expert_kit(m, imgs, te, sys_probs, n_cls):
    """Offline HTML form: blind grading first, then the model's grades, evidence maps and explanation to rate."""
    EXPERT_DIR.mkdir(parents=True, exist_ok=True); keys = m["image_key"].astype(str).to_numpy()[te]
    rel = read_csv("phase6_outputs", "reliability_test.csv", keep_default_na=False, dtype=str)
    ex = read_csv("phase7_outputs", "explanations.csv", keep_default_na=False)
    rz = PROJECT_DIR / "phase6_outputs" / "reliability_test.npz"
    evf = Path(str(np.load(rz, allow_pickle=True)["source"])) if rz.exists() else PROJECT_DIR / "phase5_outputs" / "preds" / "morphoground.npz"
    ev = np.load(evf, allow_pickle=True)["evidence"] if evf.exists() and "evidence" in np.load(evf, allow_pickle=True).files else None
    text = {}
    if ex is not None:
        for cond in ("evidence_grounded", "evidence_grounded+repair", "evidence_grounded+verify_edit", "structured"):   # last wins
            text.update(dict(zip(ex.loc[ex["condition"] == cond, "image_key"].astype(str), ex.loc[ex["condition"] == cond, "text"])))
    rng = np.random.default_rng(0); has = [i for i, k in enumerate(keys) if k in text]; rest = [i for i in range(len(keys)) if keys[i] not in text]
    pick = list(rng.permutation(has))[:N_EXPERT]; pick += list(rng.permutation(rest))[:N_EXPERT - len(pick)]
    P = sys_probs.get(MAIN_SYS) or sys_probs.get("P5:morphoground") or next(iter(sys_probs.values()))
    opts = lambda vals: "".join(f"<option>{v}</option>" for v in [""] + vals)  # noqa: E731
    cases = []
    for n, i in enumerate(pick):
        img = np.array(imgs[te[i]]); fig, axes = plt.subplots(1, 4 if ev is not None else 1, figsize=(10 if ev is not None else 3, 2.7), squeeze=False)
        axes[0, 0].imshow(img); axes[0, 0].set_title("image", fontsize=8)
        if ev is not None:
            g = math.isqrt(ev.shape[2])
            for c, t in enumerate(TASKS):
                e = ev[i, c].reshape(g, g); e = (e - e.min()) / (e.max() - e.min() + 1e-9)
                axes[0, c + 1].imshow(img); axes[0, c + 1].imshow(np.asarray(Image.fromarray((e * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0]), Image.BICUBIC)), cmap="jet", alpha=.45)
                axes[0, c + 1].set_title(f"{t} evidence", fontsize=8)
        for a in axes[0]: a.axis("off")
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="jpeg", dpi=90, pil_kwargs={"quality": 82}); plt.close(fig)
        comp = base64.b64encode(buf.getvalue()).decode()
        b2 = io.BytesIO(); Image.fromarray(img).save(b2, format="JPEG", quality=90); raw = base64.b64encode(b2.getvalue()).decode()
        if rel is not None:
            r = rel[rel["image_key"].astype(str) == keys[i]].iloc[0]
            grade = "; ".join(f"{t} {r[f'{t}_pred']} (conf {float(r[f'{t}_conf']):.2f}, plausible {r[f'{t}_set']})" for t in TASKS)
            grade += f"; review required: {r['review_required']}" + (f" ({r['reasons']})" if r["reasons"] else "")
        else:
            grade = "; ".join(f"{t} {CLASS_NAMES[t][P[c][i].argmax()]} (conf {P[c][i].max():.2f})" for c, t in enumerate(TASKS))
        sel = lambda nm, vals: f'<select name="{nm}">{opts(vals)}</select>'  # noqa: E731
        cases.append(f'''<div class="case" data-key="{escape(keys[i])}"><h3>Case {n + 1} / {len(pick)} <small>({escape(keys[i])})</small></h3>
<img src="data:image/jpeg;base64,{raw}" class="raw"><p><b>1. Your own grade (blind):</b> EXP {sel("expert_EXP", ["1", "2", "3", "4", "5", "6"])}
ICM {sel("expert_ICM", ["A", "B", "C", "ND"])} TE {sel("expert_TE", ["A", "B", "C", "ND"])}</p>
<button onclick="this.nextElementSibling.style.display='block';this.style.display='none'">Reveal the model assessment</button>
<div class="model"><img src="data:image/jpeg;base64,{comp}" class="comp"><p><b>Model:</b> {escape(grade)}</p>
<p><b>Explanation:</b> {escape(text.get(keys[i], "(no explanation generated for this embryo)"))}</p>
<p><b>2. Rate:</b> agree with model EXP {sel("agree_EXP", ["yes", "no", "unsure"])} ICM {sel("agree_ICM", ["yes", "no", "unsure"])}
TE {sel("agree_TE", ["yes", "no", "unsure"])} | explanation factually correct (1-5) {sel("factual", list("12345"))}
useful (1-5) {sel("useful", list("12345"))} | evidence regions plausible {sel("evidence_plausible", ["yes", "no", "unsure"])}</p>
<p>Comment: <input name="comment" size="80"></p></div></div>''')
    fields = ["expert_EXP", "expert_ICM", "expert_TE", "agree_EXP", "agree_ICM", "agree_TE", "factual", "useful", "evidence_plausible", "comment"]
    page = f'''<!DOCTYPE html><html><head><meta charset="utf-8"><title>MorphoGround expert evaluation</title><style>
body{{font-family:Arial,sans-serif;max-width:1100px;margin:auto;padding:10px}} .case{{border:1px solid #bbb;border-radius:6px;padding:10px;margin:14px 0}}
.raw{{width:260px}} .comp{{width:100%}} .model{{display:none;background:#f6f8fa;padding:8px;border-radius:6px}} button{{padding:6px 12px}}</style></head><body>
<h2>MorphoGround - embryologist evaluation ({len(pick)} gold-test blastocysts)</h2>
<p>For each case: (1) grade the blastocyst yourself <b>before</b> revealing the model, (2) reveal and rate the model's grades, evidence maps
and explanation. Your answers stay in this page; press <b>Download ratings</b> at the end and return the CSV.</p>
<p>Your name / ID: <input id="rater" size="30"></p>{"".join(cases)}
<button onclick="dl()">Download ratings (CSV)</button><script>
function dl(){{const f={json.dumps(fields)};let rows=[["rater","image_key"].concat(f).join(",")];const r=document.getElementById("rater").value||"anonymous";
document.querySelectorAll(".case").forEach(c=>{{rows.push([r,c.dataset.key].concat(f.map(n=>{{const e=c.querySelector("[name="+n+"]");
return '"'+(e?e.value:"").replace(/"/g,'""')+'"'}})).join(","))}});const b=new Blob([rows.join("\\n")],{{type:"text/csv"}});
const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="ratings_"+r.replace(/\\W+/g,"_")+".csv";a.click();}}
</script></body></html>'''
    (OUT / "expert_evaluation.html").write_text(page, encoding="utf-8")
    return len(pick)


def analyse_ratings(m, Y, te, sys_probs):
    files = sorted(EXPERT_DIR.glob("*.csv")) if EXPERT_DIR.exists() else []
    if not files: return None, None
    Rt = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files], ignore_index=True)
    key2i = {k: i for i, k in enumerate(m["image_key"].astype(str).to_numpy()[te])}
    P = sys_probs.get(MAIN_SYS) or sys_probs.get("P5:morphoground") or next(iter(sys_probs.values())); rows = []
    for c, t in enumerate(TASKS):
        enc = {v: i for i, v in enumerate(CLASS_NAMES[t])}
        d = Rt[Rt["image_key"].isin(key2i) & Rt[f"expert_{t}"].isin(enc)]
        ii = d["image_key"].map(key2i).to_numpy(); ex = d[f"expert_{t}"].map(enc).to_numpy(); gold = Y[te[ii], c]; mod = P[c][ii].argmax(1); ok = gold >= 0
        q = lambda a, b: cohen_kappa_score(a, b, weights="quadratic") if len(set(a) | set(b)) > 1 else np.nan  # noqa: E731
        rows.append({"criterion": t, "n_ratings": len(d), "expert_vs_gold_qwk": q(ex[ok], gold[ok]), "model_vs_gold_qwk": q(mod[ok], gold[ok]),
                     "expert_vs_model_qwk": q(ex, mod), "expert_agrees_with_model_%": 100 * (d[f"agree_{t}"] == "yes").mean()})
    A = pd.DataFrame(rows).set_index("criterion"); num = lambda s: pd.to_numeric(s, errors="coerce")  # noqa: E731
    extra = {"raters": Rt["rater"].nunique(), "factual_mean": num(Rt["factual"]).mean(), "useful_mean": num(Rt["useful"]).mean(),
             "evidence_plausible_%": 100 * (Rt["evidence_plausible"] == "yes").mean()}
    if Rt["rater"].nunique() >= 2:
        rs = Rt["rater"].unique(); ks = []
        for t in TASKS:
            piv = Rt.pivot_table(index="image_key", columns="rater", values=f"expert_{t}", aggfunc="first")
            for a in range(len(rs)):
                for b in range(a + 1, len(rs)):
                    pr = piv[[rs[a], rs[b]]].replace("", np.nan).dropna()
                    if len(pr) > 1 and pr.nunique().max() > 1: ks.append(cohen_kappa_score(pr.iloc[:, 0], pr.iloc[:, 1]))
        extra["inter_rater_kappa"] = float(np.mean(ks)) if ks else np.nan
    A.to_csv(OUT / "expert_agreement.csv"); return A, extra


def latex(df, caption, label):
    cols = "l" + "c" * (df.shape[1])
    body = "\n".join(" & ".join([str(i)] + [str(v) for v in r]) + r" \\" for i, r in zip(df.index, df.itertuples(index=False)))
    return (f"\\begin{{table}}[t]\\centering\\small\n\\caption{{{caption}}}\\label{{{label}}}\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n"
            + " & ".join([""] + [str(c) for c in df.columns]) + r" \\" + "\n\\midrule\n" + body.replace("_", r"\_").replace("%", r"\%")
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main():
    t0 = time.time(); (OUT / "tables").mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED)
    m = load_manifest(); Y, n_cls = targets(m); te = np.where(m["fold"] == -1)[0]
    status = {k: phase_status(*v) for k, v in VERDICT_FILES.items()}
    SP, bad = load_systems(m, te)
    if not SP: raise FileNotFoundError("No prediction files found - run Phases 2/5/8 first")
    global MAIN_SYS
    rz = PROJECT_DIR / "phase6_outputs" / "reliability_test.npz"
    if rz.exists():
        src = Path(str(np.load(rz, allow_pickle=True)["source"])); MAIN_SYS = {"phase5_outputs": "P5", "phase8_outputs": "P8", "phase2_outputs": "P2"}.get(
            src.parent.parent.name, "P5") + ":" + src.stem
    cvm = (m["fold"] >= 0).to_numpy()
    COL = {s: any(collapse_report(Y[cvm, c], O[c][cvm])["collapsed"] for c in range(len(TASKS))) for s, O in OOF.items()}
    collapsed = sorted(s for s, v in COL.items() if v)
    if collapsed: log(f"collapsed systems (reported, never compared): {collapsed}")
    log(f"{len(SP)} systems on {len(te)} gold-test embryos; bootstrap {N_BOOT}x")

    # ---------------- 1. point estimates + bootstrap CIs (same resamples for every system -> paired)
    boot, point = {}, {}
    for c, t in enumerate(TASKS):
        v = Y[te, c] >= 0; y = Y[te[v], c]; K = n_cls[c]; preds = {s: P[c][v].argmax(1) for s, P in SP.items()}
        idx = rng.integers(0, len(y), (N_BOOT, len(y)))
        for s, p in preds.items():
            point[(s, t)] = cm_metrics(y, p, K); boot[(s, t)] = {k: np.empty(N_BOOT) for k in METS}
            for b in range(N_BOOT):
                r = cm_metrics(y[idx[b]], p[idx[b]], K)
                for k in METS: boot[(s, t)][k][b] = r[k]
    rows = []
    for s in SP:
        row = {"system": s}
        for t in TASKS:
            for k in METS:
                lo, hi = np.percentile(boot[(s, t)][k], [2.5, 97.5]); row.update({f"{t}_{k}": point[(s, t)][k], f"{t}_{k}_lo": lo, f"{t}_{k}_hi": hi})
        row["mean_macro_f1"] = np.mean([point[(s, t)]["macro_f1"] for t in TASKS]); row["mean_qwk"] = np.mean([point[(s, t)]["qwk"] for t in TASKS])
        rows.append(row)
    MT = pd.DataFrame(rows).set_index("system"); MT["collapsed"] = [COL.get(s, False) for s in MT.index]
    MT.to_csv(OUT / "tables" / "gold_test_all_systems.csv")

    # ---------------- 2. which systems to compare (chosen on validation, never on test)
    s2, rec8 = read_csv("phase2_outputs", "summary.csv"), None
    best2 = per_bb = None
    if s2 is not None:
        g2 = s2.groupby(["model", "mode"])["sel"].mean(); best2 = "P2:{}_{}".format(*g2.idxmax())
        per_bb = {mdl: "P2:{}_{}".format(mdl, g2.loc[mdl].idxmax()) for mdl in g2.index.get_level_values(0).unique()}
    rj = PROJECT_DIR / "phase8_outputs" / "recommended.json"
    if rj.exists(): rec8 = "P8:" + json.loads(rj.read_text())["variant"]
    pairs = [(a, b, lab) for a, b, lab in (("P5:morphoground", best2, "MorphoGround vs best Phase 2 baseline"),
                                            ("P5:morphoground", "P5:global_ce", "MorphoGround vs same-backbone global CE"),
                                            ("P5:global_ordinal", "P5:global_ce", "Ordinal (CORN) heads vs CE (both global)"),
                                            ("P5:tokens_ordinal", "P5:global_ordinal", "Morphology tokens + relations vs global (both ordinal)"),
                                            ("P5:morphoground", "P5:tokens_ordinal", "Grounding losses vs unconstrained tokens"),
                                            (rec8, "P8:ce", "Rare-grade recipe vs plain CE (Phase 8)"),
                                            (rec8, best2, "Rare-grade recipe vs best Phase 2 baseline"))
             if a in SP and b in SP and a != b and not COL.get(a) and not COL.get(b)]
    comp = []
    for a, b, lab in pairs:
        for k in ("qwk", "macro_f1"):
            ps = []
            for c, t in enumerate(TASKS):
                dlt = boot[(a, t)][k] - boot[(b, t)][k]; lo, hi = np.percentile(dlt, [2.5, 97.5])
                p = min(1.0, 2 * min((dlt <= 0).mean(), (dlt >= 0).mean()) + 1 / N_BOOT); ps.append(p)
                v = Y[te, c] >= 0; ca, cb = SP[a][c][v].argmax(1) == Y[te[v], c], SP[b][c][v].argmax(1) == Y[te[v], c]
                n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
                comp.append({"comparison": lab, "A": a, "B": b, "metric": k, "task": t, "A_value": point[(a, t)][k], "B_value": point[(b, t)][k],
                             "delta": point[(a, t)][k] - point[(b, t)][k], "ci_lo": lo, "ci_hi": hi, "p_boot": p,
                             "mcnemar_p": binomtest(min(n01, n10), n01 + n10, .5).pvalue if n01 + n10 else 1.0})
            for r_, pa in zip(comp[-3:], holm(ps)): r_["p_holm"] = pa
    C = pd.DataFrame(comp); C.to_csv(OUT / "tables" / "paired_comparisons.csv", index=False)

    # ---------------- 2b. v2: gold-test results split by sibling status (same patient/cycle in training or not)
    sib, srows = sibling_mask(m, te), []
    for s in [x for x in dict.fromkeys([best2, "P5:global_ce", "P5:morphoground", MAIN_SYS, rec8]) if x in SP and not COL.get(x)]:
        for grp, gm in (("sibling in training", sib), ("no sibling in training", ~sib)):
            for c, t in enumerate(TASKS):
                v = gm & (Y[te, c] >= 0); y, p = Y[te[v], c], SP[s][c][v].argmax(1)
                if len(y) < 5: continue
                q = [cm_metrics(y[i], p[i], n_cls[c])["qwk"] for i in rng.integers(0, len(y), (N_BOOT, len(y)))]
                lo, hi = np.percentile(q, [2.5, 97.5])
                srows.append({"system": s, "group": grp, "task": t, "n": int(len(y)), "qwk": cm_metrics(y, p, n_cls[c])["qwk"], "qwk_lo": lo, "qwk_hi": hi})
    SB = pd.DataFrame(srows); SB.to_csv(OUT / "tables" / "sibling_stratified.csv", index=False)

    # ---------------- 3. ablation matrix (strategy table) + other evaluation levels
    s3, s6 = read_csv("phase3_outputs", "summary.csv", index_col=0), read_csv("phase6_outputs", "reliability_summary.csv", index_col=0)
    s7 = read_csv("phase7_outputs", "factuality_summary.csv", index_col=0)
    p4f = PROJECT_DIR / "phase4_outputs" / "phase4_summary.json"; s4 = json.loads(p4f.read_text()) if p4f.exists() else None
    ab = []
    lookup = lambda s: (MT.loc[s, "mean_macro_f1"], MT.loc[s, "mean_qwk"]) if s in MT.index else (np.nan, np.nan)  # noqa: E731
    for lab, s, flags in [("ResNet-50", (per_bb or {}).get("resnet50"), "------"), ("ViT-B/16", (per_bb or {}).get("vit_b16"), "------"),
                          ("DINOv2", (per_bb or {}).get("dinov2_b14"), "S-----"), ("Global CE (Phase 5)", "P5:global_ce", "S-----"),
                          ("+ Ordinal (CORN)", "P5:global_ordinal", "S--O--"), ("+ Morphology + Relational", "P5:tokens_ordinal", "SMRO--"),
                          ("+ Grounding (MorphoGround)", "P5:morphoground", "SMRO--"), ("+ Rare-grade recipe (Phase 8)", rec8, "SMRO--")]:
        f1, q = lookup(s)
        ab.append({"model": lab, **{c: ("Y" if flags[i] != "-" else "-") for i, c in enumerate(("SSL", "Morphology", "Relational", "Ordinal", "Uncertainty", "VLM"))},
                   "test_macro_F1": f1, "test_QWK": q, "extra": ""})
    if s3 is not None and "mae_in1k_dapt" in s3.index:
        ab.insert(3, {"model": "MAE adapted to embryos (frozen probe)", "SSL": "Y", **{c: "-" for c in ("Morphology", "Relational", "Ordinal", "Uncertainty", "VLM")},
                      "test_macro_F1": np.nan, "test_QWK": s3.loc["mae_in1k_dapt", "test_qwk"], "extra": "Phase 3 linear probe"})
    if s6 is not None:
        ab.append({"model": "+ Uncertainty (keep 80% most certain)", "SSL": "Y", "Morphology": "Y", "Relational": "Y", "Ordinal": "Y", "Uncertainty": "Y", "VLM": "-",
                   "test_macro_F1": s6["macro_f1_selective"].mean(), "test_QWK": s6["qwk_selective"].mean(), "extra": f"coverage {s6['test_coverage_at_tau'].mean():.0%}"})
    if s7 is not None:
        fin = next(c for c in ("structured", "evidence_grounded+verify_edit", "evidence_grounded+repair", "evidence_grounded") if c in s7.index)
        ab.append({"model": "+ Evidence-grounded VLM", "SSL": "Y", "Morphology": "Y", "Relational": "Y", "Ordinal": "Y", "Uncertainty": "Y", "VLM": "Y",
                   "test_macro_F1": np.nan, "test_QWK": np.nan,
                   "extra": f"contradictions {s7.loc[fin, 'contradiction_rate']:.1%} (prediction-only {s7.loc['prediction_only', 'contradiction_rate']:.1%})"})
    AB = pd.DataFrame(ab).set_index("model"); AB.to_csv(OUT / "tables" / "ablation_matrix.csv")

    # ---------------- 4. research questions
    RQ = {}
    if s3 is not None:
        ssl = s3[s3.index.isin(["dinov2_b14", "mae_in1k", "femi", "mae_in1k_dapt", "femi_dapt"])]["score"]; sup = s3[s3.index.isin(["resnet50", "vit_b16"])]["score"]
        dapt = [(i, f"{i}_dapt") for i in ("mae_in1k", "femi") if f"{i}_dapt" in s3.index and i in s3.index]
        gain = [s3.loc[d, "score"] - s3.loc[i, "score"] for i, d in dapt]
        st = "supported" if len(ssl) and len(sup) and ssl.max() > sup.max() and gain and max(gain) > 0 else \
             "partially supported (general SSL beats supervised; embryo adaptation not shown)" if len(ssl) and len(sup) and ssl.max() > sup.max() else "not supported"
        RQ["RQ1 self-supervised embryo representations"] = (st, f"best SSL {ssl.idxmax()} {ssl.max():.3f} vs supervised {sup.max():.3f}; adaptation gains {', '.join(f'{x:+.3f}' for x in gain) or 'not measured'}")
    else: RQ["RQ1 self-supervised embryo representations"] = ("not tested", "Phase 3 summary missing")
    if s4:
        faith, spec = all(v > 0 for v in s4["faithfulness_gap_20"].values()), all(s4["specific"].values())
        anat = "pointing" in s4 and all(s4["pointing"][t]["energy_in_expected"] > s4["pointing"][t]["area_baseline"] for t in ("ICM", "TE") if t in s4["pointing"])
        extra4 = ""
        if len(s4.get("models", {})) > 1:
            mm_ = s4["models"]; o = next(k for k in mm_ if k != s4.get("main_model"))
            g_ = lambda d: float(np.mean(list(d["specificity_margin_20"].values())))  # noqa: E731
            extra4 = f"; specificity margin {o} {g_(mm_[o]):+.3f} -> {s4['main_model']} {g_(s4):+.3f}"
        RQ["RQ2 anatomically grounded tokens"] = ("supported" if faith and spec and anat else "partially supported" if faith else "not supported",
                                                  f"faithful {faith}, criterion-specific {spec}, evidence-map similarity "
                                                  f"{s4.get('evidence_similarity_mean', float('nan')):.2f}, anatomical (masks) "
                                                  f"{anat if 'pointing' in s4 else 'not measured'}{extra4}")
    else: RQ["RQ2 anatomically grounded tokens"] = ("not tested", "Phase 4 not run")
    r3 = C[(C["comparison"] == "MorphoGround vs same-backbone global CE") & (C["metric"] == "qwk")] if len(C) else pd.DataFrame()
    RQ["RQ3 relational + ordinal grading"] = (("supported" if (r3["delta"] > 0).all() and (r3["p_holm"] < ALPHA_STAT).any() else
                                               "trend, not significant" if r3["delta"].mean() > 0 else "not supported"),
                                              " | ".join(f"{r.task} dQWK {r.delta:+.3f} [{r.ci_lo:+.3f},{r.ci_hi:+.3f}] p={r.p_holm:.3f}" for r in r3.itertuples())) \
        if len(r3) else ("not tested", "Phase 5 variants missing")
    RQ["RQ4 uncertainty flags embryos for review"] = (("supported" if (s6["aurc_1-maxprob"] < s6["aurc_random"]).all() and (s6["auroc_err_1-maxprob"] >= .65).all()
                                                       else "partially supported"),
                                                      " | ".join(f"{t} AUROC(err) {r['auroc_err_1-maxprob']:.2f}, acc {r.acc_all:.2f}->{r.acc_selective:.2f}"
                                                                 + (f", gold-cal. coverage {r['coverage_gold_crossfit']:.2f}" if "coverage_gold_crossfit" in s6 else "")
                                                                 for t, r in s6.iterrows())) \
        if s6 is not None else ("not tested", "Phase 6 not run")
    if s7 is not None:
        g, p_ = s7.loc[fin], s7.loc["prediction_only"]
        dfr = g.get("deferral_compliance", np.nan)
        ok5 = (g["contradiction_rate"] <= p_["contradiction_rate"] and g["hallucination_rate"] <= .05 and g["completeness"] >= .9
               and (pd.isna(dfr) or dfr >= .9))
        RQ["RQ5 evidence-grounded explanations"] = ("supported" if ok5 else "partially supported",
                                                    f"{fin}: contradictions {p_['contradiction_rate']:.1%} (prediction-only) -> {g['contradiction_rate']:.1%}, "
                                                    f"hallucinations {p_['hallucination_rate']:.1%} -> {g['hallucination_rate']:.1%}, complete "
                                                    f"{g['completeness']:.0%}, review statements {dfr:.0%}; descriptive accuracy needs expert ratings")
    else: RQ["RQ5 evidence-grounded explanations"] = ("not tested", "Phase 7 not run")

    # ---------------- 5. expert evaluation kit + ratings (if returned)
    imgs = image_cache(m); n_exp = expert_kit(m, imgs, te, SP, n_cls); EA, extra = analyse_ratings(m, Y, te, SP)

    # ---------------- 6. figures + paper tables + reproducibility
    key_sys = [s for s in dict.fromkeys([(per_bb or {}).get("resnet50"), (per_bb or {}).get("vit_b16"), (per_bb or {}).get("dinov2_b14"),
                                         "P5:global_ce", "P5:global_ordinal", "P5:tokens_ordinal", "P5:morphoground", "P8:ce", rec8])
               if s in MT.index and not COL.get(s)]
    fig, axes = plt.subplots(2, 3, figsize=(17, 1.2 + .45 * len(key_sys) * 2))
    for r_, k in enumerate(("macro_f1", "qwk")):
        for c, t in enumerate(TASKS):
            ax = axes[r_, c]; v = MT.loc[key_sys]
            ax.errorbar(v[f"{t}_{k}"], range(len(key_sys)), xerr=[v[f"{t}_{k}"] - v[f"{t}_{k}_lo"], v[f"{t}_{k}_hi"] - v[f"{t}_{k}"]], fmt="o", capsize=3)
            ax.set_yticks(range(len(key_sys)), key_sys if c == 0 else [""] * len(key_sys), fontsize=8); ax.invert_yaxis()
            ax.set_title(f"{t} - gold test {k} (95% bootstrap CI)", fontsize=9); ax.grid(axis="x", alpha=.3)
    save(fig, OUT / "main_results_ci.png")
    if len(C):
        cq = C[C["metric"] == "qwk"]; labs = [f"{r.comparison} | {r.task}" for r in cq.itertuples()]
        fig, ax = plt.subplots(figsize=(10, .35 * len(cq) + 1.5))
        ax.errorbar(cq["delta"], range(len(cq)), xerr=[cq["delta"] - cq["ci_lo"], cq["ci_hi"] - cq["delta"]], fmt="o", capsize=3)
        ax.axvline(0, c="gray", ls="--"); ax.set_yticks(range(len(cq)), labs, fontsize=7); ax.invert_yaxis()
        ax.set(xlabel="difference in gold-test QWK (A - B), 95% paired bootstrap CI", title="Paired comparisons"); save(fig, OUT / "forest_plot.png")
    tab = pd.DataFrame({t: [fmt_ci(MT.loc[s, f"{t}_qwk"], MT.loc[s, f"{t}_qwk_lo"], MT.loc[s, f"{t}_qwk_hi"]) for s in key_sys] for t in TASKS}, index=key_sys)
    tab.insert(0, "mean macro-F1", [f"{MT.loc[s, 'mean_macro_f1']:.3f}" for s in key_sys])
    abt = AB.copy(); abt["test_macro_F1"] = abt["test_macro_F1"].map(lambda x: "" if pd.isna(x) else f"{x:.3f}"); abt["test_QWK"] = abt["test_QWK"].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    tex = latex(tab, "Gold-test QWK with 95\\% bootstrap CIs (Kromp et al. 2023 test set, 300 embryos).", "tab:main") + "\n" + \
        latex(abt, "Ablation matrix (Y = component present).", "tab:ablation")
    if len(C): tex += "\n" + latex(C[C["metric"] == "qwk"].set_index("comparison")[["task", "delta", "ci_lo", "ci_hi", "p_holm"]].round(3), "Paired comparisons (QWK).", "tab:pairs")
    (OUT / "tables" / "paper_tables.tex").write_text(tex)
    files = [PHASE1_DIR / "manifest.csv"] + sorted((PROJECT_DIR).glob("phase*_outputs/preds/*.npz"))
    repro = {"versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                          "scikit-learn": sklearn.__version__}, "n_gold_test": len(te), "cv_fold_sizes": m.loc[m["fold"] >= 0, "fold"].value_counts().sort_index().to_dict(),
             "bootstrap": N_BOOT, "seed": SEED, "sha256": {str(f.relative_to(PROJECT_DIR)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.exists()},
             "phase_verdicts": status}
    (OUT / "reproducibility.json").write_text(json.dumps(repro, indent=2, default=str))
    md = ["# MorphoGround - final results\n", "## Research questions\n"] + [f"- **{k}**: {v[0]} - {v[1]}" for k, v in RQ.items()]
    md += ["\n## Gold-test QWK (95% CI)\n", tab.to_markdown() if hasattr(tab, "to_markdown") and _has_tabulate() else tab.to_string(),
           "\n## Ablation matrix\n", abt.to_string(), "\n## Paired comparisons\n", C.round(4).to_string() if len(C) else "none",
           "\n## Gold-test QWK by sibling status\n", SB.round(3).to_string() if len(SB) else "not available",
           "\n## Collapsed systems (excluded from comparisons)\n", ", ".join(collapsed) or "none"]
    (OUT / "RESULTS_SUMMARY.md").write_text("\n".join(md))

    # ---------------- verdict
    checks = [("PASS" if all(status[k] and "FAIL" not in status[k] for k in status if status[k]) else "FAIL", "No phase ended in FAIL",
               " | ".join(f"P{k} {'-' if not v else 'PWW' if v == 'PASS WITH WARNINGS' else v}" for k, v in status.items())),
              ("PASS" if all(status[k] for k in (1, 2, 5)) else "WARN", "Core phases present (1, 2, 5)", "yes" if all(status[k] for k in (1, 2, 5)) else "missing"),
              ("PASS" if not bad else "FAIL", "Every prediction file uses the same locked gold test set", f"{len(SP)} systems" + (f"; misaligned {bad}" if bad else "")),
              ("PASS" if not COL.get("P5:morphoground") and not COL.get(MAIN_SYS) else "FAIL", "Main models are not collapsed",
               f"main {MAIN_SYS}; collapsed: {', '.join(collapsed) or 'none'}"),
              ("INFO", "Gold test by sibling status (QWK, main model)", " | ".join(
                  f"{r.task} {r.group.split()[0]} {r.qwk:.2f} (n={r.n})" for r in SB[SB['system'] == MAIN_SYS].itertuples()) if len(SB) else "n/a"),
              ("PASS", "Bootstrap CIs and paired tests", f"{N_BOOT} paired resamples, Holm-corrected over EXP/ICM/TE, McNemar on accuracy"),
              ("PASS", "Expert evaluation kit written", f"expert_evaluation.html with {n_exp} embryos (blind grading, then rating)"),
              ("INFO", "Expert ratings", "none yet: put returned CSVs in phase9_outputs/expert_ratings/ and rerun Phase 9" if EA is None else
               f"{extra['raters']} rater(s); expert-vs-gold QWK " + ", ".join(f"{t} {EA.loc[t, 'expert_vs_gold_qwk']:.2f}" for t in TASKS)
               + f"; factual {extra['factual_mean']:.2f}/5, useful {extra['useful_mean']:.2f}/5")]
    checks += [("INFO", k, f"{v[0]} - {v[1]}"[:200]) for k, v in RQ.items()]
    keys = ["Gold test (mean over EXP/ICM/TE):       macro-F1   QWK      EXP QWK [95% CI]        ICM QWK [95% CI]        TE QWK [95% CI]"]
    keys += [f"  {s:<36} {MT.loc[s, 'mean_macro_f1']:.3f}      {MT.loc[s, 'mean_qwk']:.3f}    " + "   ".join(
        fmt_ci(MT.loc[s, f"{t}_qwk"], MT.loc[s, f"{t}_qwk_lo"], MT.loc[s, f"{t}_qwk_hi"]) for t in TASKS) for s in key_sys]
    keys += [f"{r.comparison} [{r.task}] dQWK {r.delta:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] p(Holm) {r.p_holm:.3f}"
             for r in C[C["metric"] == "qwk"].itertuples()] if len(C) else []
    keys.append("Reference (Kromp 2023, reported): DL baselines EXP 0.77-0.85, ICM/TE 0.51-0.69; human experts 0.70-0.83 (precision/recall/F1)")
    keys.append(f"outputs: RESULTS_SUMMARY.md, tables/paper_tables.tex, main_results_ci.png, forest_plot.png | runtime {(time.time() - t0) / 60:.1f} min")
    verdict(9, checks, keys, OUT, "Validation complete - write up from RESULTS_SUMMARY.md and send expert_evaluation.html to embryologists.")
    return MT, C, RQ


def _has_tabulate():
    try: import tabulate  # noqa: F401
    except ImportError: return False
    return True


if __name__ == "__main__":
    run_safely(9, main)
