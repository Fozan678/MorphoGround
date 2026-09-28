"""MorphoGround - PHASE 6 (v2): reliability - calibration, uncertainty, selective prediction (abstention) and conformal sets.
Every fold model is calibrated on its own validation fold and applied to its gold-test predictions (no test data used to fit).
v2: (1) the input predictions are chosen automatically (Phase 5 MorphoGround if it passes the ordinal-collapse check, else the
best healthy Phase 5 variant) and a collapsed input is a FAIL; (2) silver-calibrated conformal sets cannot promise 90% coverage
of GOLD labels (silver != gold labels breaks exchangeability), so a second, gold-calibrated scheme is reported: each gold
embryo's set is calibrated on the OTHER gold embryos (grouped K-fold by patient/cycle, never its own label) - the coverage a
small expert-consensus calibration set would give in deployment.
Writes the per-embryo review queue used by Phase 7. CPU only (no GPU needed).
Self-contained: paste into ONE Jupyter cell, or run  python phase6_reliability.py"""
from pathlib import Path

# ====================================== USER SETTINGS ========================================
PROJECT_DIR = Path.home() / "Embryo_Updated Project"   # same folder as Phases 1-3
PHASE1_DIR  = PROJECT_DIR / "phase1_outputs"

# ---- PHASE 6 settings
SOURCE          = "auto"  # auto = Phase 5 MorphoGround if not collapsed, else best healthy Phase 5 variant; or e.g.
                          # "phase5_outputs/preds/global_ce.npz" / "phase8_outputs/preds/logit_adjusted.npz" (relative to PROJECT_DIR)
GOLD_CAL_SPLITS = 5       # gold-calibrated conformal: grouped K-fold over the gold set (0 = off)
CARD_SETS       = "classwise"   # plausible grades shown on the Phase 7 card: classwise | marginal (silver-calibrated) | gold
ALPHA           = 0.10    # conformal prediction: target 90% coverage
TARGET_COVERAGE = 0.80    # selective prediction: keep the 80% most certain predictions per criterion
OUT             = PROJECT_DIR / "phase6_outputs"
# ============================================================================================

import json, math, sys, time, warnings
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
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


# ##################################### PHASE 6 #####################################
from scipy.optimize import minimize_scalar  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def softmax_T(p, T):
    L = np.log(np.clip(p, 1e-8, 1)) / T; L -= L.max(1, keepdims=True); e = np.exp(L); return e / e.sum(1, keepdims=True)


def fit_temperature(p, y):
    L = np.log(np.clip(p, 1e-8, 1))
    def nll(lt):
        z = L / np.exp(lt); z = z - z.max(1, keepdims=True)
        return -np.mean(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1)))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def ece(p, y, bins=15):
    c, ok = p.max(1), p.argmax(1) == y; e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        k = (c > lo) & (c <= hi); e += k.mean() * abs(ok[k].mean() - c[k].mean()) if k.any() else 0
    return float(e)


def brier(p, y): return float(((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean())


def conformal_q(p, y, alpha):
    s = np.sort(1 - p[np.arange(len(y)), y]); n = len(s); return float(s[min(n, int(np.ceil((n + 1) * (1 - alpha)))) - 1])


def make_sets(p, q):
    """Split-conformal (LAC) sets; q scalar (marginal) or per-class vector (class-conditional). Never empty."""
    S = (1 - p) <= np.broadcast_to(q, p.shape); S[np.arange(len(p)), p.argmax(1)] = True; return S


def uncertainty(p):
    k = np.arange(p.shape[1]); mu = p @ k
    return {"1-maxprob": 1 - p.max(1), "entropy": -(p * np.log(np.clip(p, 1e-12, 1))).sum(1),
            "ordinal_sd": np.sqrt(np.clip(p @ k ** 2 - mu ** 2, 0, None))}


def risk_coverage(u, correct):
    o = np.argsort(u, kind="stable"); c = correct[o].astype(float); n = np.arange(1, len(c) + 1)
    return n / len(c), 1 - np.cumsum(c) / n


def process_task(t, K, D, m, Y, te, gids):
    """Per fold model: temperature + conformal calibrated on its own validation fold, applied to its test predictions."""
    ti = TASKS.index(t)
    fold_ids = [f for f in sorted(int(x) for x in m["fold"].unique() if x >= 0) if np.isfinite(D["oof"][m["fold"].to_numpy() == f]).all()]
    yt, okt = Y[te, ti], Y[te, ti] >= 0
    oof_cal = np.full_like(D["oof"], np.nan); cal_tests, Ts, cov_m, cov_c, votes_m, votes_c = [], [], [], [], 0, 0
    per_fold = D["folds"] is not None and len(D["folds"]) == len(fold_ids)
    groups = [(i, f) for i, f in enumerate(fold_ids)] if per_fold else [(None, None)]
    for i, f in groups:
        va = np.where(((m["fold"] == f) if f is not None else (m["fold"] >= 0)).to_numpy() & (Y[:, ti] >= 0) & np.isfinite(D["oof"]).all(1))[0]
        pv, yv = D["oof"][va], Y[va, ti]; T_ = fit_temperature(pv, yv); Ts.append(T_)
        pv_c, pt_c = softmax_T(pv, T_), softmax_T(D["folds"][i] if per_fold else D["test"], T_)
        oof_cal[va] = pv_c; cal_tests.append(pt_c)
        qm = conformal_q(pv_c, yv, ALPHA)
        qc = np.array([conformal_q(pv_c[yv == c], yv[yv == c], ALPHA) if (yv == c).sum() >= 5 else 1.0 for c in range(K)])
        Sm, Sc = make_sets(pt_c, qm), make_sets(pt_c, qc)
        cov_m.append(Sm[okt][np.arange(okt.sum()), yt[okt]].mean()); cov_c.append(Sc[okt][np.arange(okt.sum()), yt[okt]].mean())
        votes_m, votes_c = votes_m + Sm, votes_c + Sc
    P = np.mean(cal_tests, 0); need = len(groups) / 2
    set_m, set_c = np.asarray(votes_m >= need), np.asarray(votes_c >= need)        # majority vote of the fold models' sets
    set_m[np.arange(len(P)), P.argmax(1)] = True; set_c[np.arange(len(P)), P.argmax(1)] = True
    raw = D["test"]; va_all = np.where(np.isfinite(oof_cal).all(1) & (Y[:, ti] >= 0))[0]
    u_val, u_te = uncertainty(oof_cal[va_all]), uncertainty(P)
    tau = float(np.quantile(u_val["1-maxprob"], TARGET_COVERAGE))                  # threshold chosen on validation only
    keep = u_te["1-maxprob"] <= tau; pred, correct = P.argmax(1), P.argmax(1) == yt
    mi = None
    if per_fold:
        H = lambda q: -(q * np.log(np.clip(q, 1e-12, 1))).sum(-1)  # noqa: E731
        mi = H(P) - np.mean([H(c) for c in cal_tests], 0)
    res = {"task": t, "T_mean": float(np.mean(Ts)), "ece_before": ece(raw[okt], yt[okt]), "ece_after": ece(P[okt], yt[okt]),
           "brier_before": brier(raw[okt], yt[okt]), "brier_after": brier(P[okt], yt[okt]),
           "coverage_marginal_mean": float(np.mean(cov_m)), "coverage_marginal_sd": float(np.std(cov_m)),
           "coverage_classwise_mean": float(np.mean(cov_c)), "set_size_marginal": float(set_m[okt].sum(1).mean()),
           "set_size_classwise": float(set_c[okt].sum(1).mean()), "singleton_rate": float((set_m[okt].sum(1) == 1).mean()),
           "tau_1-maxprob": tau, "test_coverage_at_tau": float(keep[okt].mean()),
           "acc_all": float(correct[okt].mean()), "acc_selective": float(correct[okt & keep].mean()) if (okt & keep).any() else np.nan,
           "qwk_all": metrics(yt, P)["qwk"], "qwk_selective": metrics(np.where(keep, yt, -1), P)["qwk"],
           "macro_f1_all": metrics(yt, P)["macro_f1"], "macro_f1_selective": metrics(np.where(keep, yt, -1), P)["macro_f1"]}
    for name, u in {**u_te, **({"fold_disagreement(MI)": mi} if mi is not None else {})}.items():
        cov, risk = risk_coverage(u[okt], correct[okt]); res[f"aurc_{name}"] = float(risk.mean())
        res[f"auroc_err_{name}"] = float(roc_auc_score(~correct[okt], u[okt])) if 0 < correct[okt].mean() < 1 else np.nan
    res["aurc_random"] = 1 - res["acc_all"]
    res["aurc_oracle"] = float(risk_coverage(-correct[okt].astype(float), correct[okt])[1].mean())
    for c in range(K):
        k = okt & (yt == c)
        res[f"cov_marg_{CLASS_NAMES[t][c]}"] = float(set_m[k, c].mean()) if k.any() else np.nan
        res[f"cov_cls_{CLASS_NAMES[t][c]}"] = float(set_c[k, c].mean()) if k.any() else np.nan
    set_g = None
    if GOLD_CAL_SPLITS and okt.sum() >= 2 * GOLD_CAL_SPLITS:                    # v2: gold-calibrated (cross-fitted) sets
        set_g, cg, cg_sd = gold_crossfit(P, yt, okt, gids, ALPHA, GOLD_CAL_SPLITS)
        res.update({"coverage_gold_crossfit": cg, "coverage_gold_crossfit_sd": cg_sd, "set_size_gold_crossfit": float(set_g[okt].sum(1).mean())})
        for c in range(K):
            k = okt & (yt == c); res[f"cov_gold_{CLASS_NAMES[t][c]}"] = float(set_g[k, c].mean()) if k.any() else np.nan
    extra = {"P": P, "raw": raw, "set_m": set_m, "set_c": set_c, "set_g": set_g, "u": u_te, "mi": mi, "keep": keep, "correct": correct,
             "okt": okt, "yt": yt}
    return res, extra


def collapse_report(y, P, min_prev=.05):
    """Ordinal-collapse guard (as Phase 5): a grade with >= 5% prevalence (almost) never predicted, or predicted-grade
    distribution far from the true one (total variation > 0.5)."""
    k = (y >= 0) & np.isfinite(P).all(1); y, p, K = y[k], P[k].argmax(1), P.shape[1]
    if not len(y): return {"collapsed": False, "tv": 0.0, "dead": []}
    ty, tp = np.bincount(y, minlength=K) / len(y), np.bincount(p, minlength=K) / len(y)
    dead = [c for c in range(K) if ty[c] >= min_prev and (p[y == c] == c).mean() < .05]
    tv = float(.5 * np.abs(ty - tp).sum())
    return {"collapsed": bool(dead) or tv > .5, "tv": tv, "dead": dead}


def choose_source(m, Y):
    """Predictions to make reliable: SOURCE, or 'auto' = MorphoGround if healthy, else the best healthy Phase 5 variant."""
    cv = (m["fold"] >= 0).to_numpy()
    health = lambda f: {t: collapse_report(Y[cv, c], np.load(f, allow_pickle=True)[f"oof_{t}"][cv]) for c, t in enumerate(TASKS)}  # noqa: E731
    if SOURCE != "auto":
        f = PROJECT_DIR / SOURCE
        if not f.exists(): raise FileNotFoundError(f"{f} not found")
        return f, "set by user", health(f)
    d = PROJECT_DIR / "phase5_outputs" / "preds"
    cands = {f.stem: f for f in sorted(d.glob("*.npz"))} if d.exists() else {}
    if not cands: raise FileNotFoundError(f"No prediction files in {d} - run Phase 5 first")
    H = {v: health(f) for v, f in cands.items()}; ok = [v for v in cands if not any(h["collapsed"] for h in H[v].values())]
    if "morphoground" in ok: return cands["morphoground"], "MorphoGround (passes the collapse check)", H["morphoground"]
    s5 = PROJECT_DIR / "phase5_outputs" / "summary.csv"
    if ok and s5.exists():
        S = pd.read_csv(s5); S = S[S["variant"].isin(ok)]
        if len(S):
            v = S.assign(s=(S["val_macro_f1"] + S["val_qwk"]) / 2).groupby("variant")["s"].mean().idxmax()
            return cands[v], f"{v}: best healthy Phase 5 variant (MorphoGround missing or collapsed)", H[v]
    if ok: return cands[ok[0]], f"{ok[0]}: first healthy Phase 5 variant", H[ok[0]]
    v = "morphoground" if "morphoground" in cands else next(iter(cands))
    return cands[v], f"{v} (WARNING: every Phase 5 variant is collapsed)", H[v]


def gold_crossfit(P, yt, okt, groups, alpha, n_splits):
    """Gold-calibrated split-conformal sets: each gold embryo's set uses a threshold computed on the OTHER gold embryos
    (GroupKFold by patient/cycle, so no embryo - and no sibling - calibrates its own set). Marginal LAC scores."""
    from sklearn.model_selection import GroupKFold
    idx = np.where(okt)[0]; S, cov = np.zeros(P.shape, bool), []
    n_splits = min(n_splits, len(np.unique(groups[idx])))
    for cal, ev in GroupKFold(n_splits).split(idx, groups=groups[idx]):
        c, e = idx[cal], idx[ev]; S[e] = make_sets(P[e], conformal_q(P[c], yt[c], alpha)); cov.append(float(S[e, yt[e]].mean()))
    rest = np.where(~okt)[0]
    if len(rest): S[rest] = make_sets(P[rest], conformal_q(P[idx], yt[idx], alpha))
    return S, float(np.mean(cov)), float(np.std(cov))


def figures(R, X, n_cls):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))                                              # reliability diagrams
    for ax, (k, t) in zip(axes, enumerate(TASKS)):
        x = X[t]; ok = x["okt"]
        for p, lab, st in ((x["raw"], "before", "o--"), (x["P"], "after temperature scaling", "o-")):
            c, hit = p[ok].max(1), p[ok].argmax(1) == x["yt"][ok]; b = np.clip((c * 10).astype(int), 0, 9)
            xs = [c[b == i].mean() for i in range(10) if (b == i).sum() >= 5]; ys = [hit[b == i].mean() for i in range(10) if (b == i).sum() >= 5]
            ax.plot(xs, ys, st, label=f"{lab} (ECE {ece(p[ok], x['yt'][ok]):.3f})")
        ax.plot([0, 1], [0, 1], c="gray", lw=1); ax.set(xlabel="confidence", ylabel="accuracy", title=f"{t}: reliability (gold test)"); ax.legend(fontsize=7)
    save(fig, OUT / "reliability_diagrams.png")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))                                              # risk-coverage
    for ax, t in zip(axes, TASKS):
        x = X[t]; ok = x["okt"]
        for name, u in {**x["u"], **({"fold disagreement (MI)": x["mi"]} if x["mi"] is not None else {})}.items():
            cov, risk = risk_coverage(u[ok], x["correct"][ok]); ax.plot(cov, risk, label=name)
        ax.axhline(1 - x["correct"][ok].mean(), c="gray", ls="--", label="random abstention")
        ax.axvline(TARGET_COVERAGE, c="k", ls=":", lw=1); ax.set(xlabel="coverage (fraction kept)", ylabel="error rate on kept", title=f"{t}: risk-coverage")
        ax.legend(fontsize=7)
    save(fig, OUT / "risk_coverage.png")
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5))                                              # conformal
    for c, t in enumerate(TASKS):
        x = X[t]; ok = x["okt"]; K = n_cls[c]
        hs, hl = [x["set_m"][ok].sum(1), x["set_c"][ok].sum(1)], ["marginal (silver cal.)", "class-conditional (silver cal.)"]
        if x["set_g"] is not None: hs.append(x["set_g"][ok].sum(1)); hl.append("gold-calibrated (cross-fitted)")
        axes[0, c].hist(hs, bins=np.arange(.5, K + 1.5), label=hl)
        axes[0, c].set(title=f"{t}: conformal set size (target {1 - ALPHA:.0%})", xlabel="set size"); axes[0, c].legend(fontsize=7)
        kinds = (("marginal", "marg"), ("class-conditional", "cls")) + ((("gold-calibrated", "gold"),) if f"cov_gold_{CLASS_NAMES[t][0]}" in R else ())
        d = pd.DataFrame([{"class": CLASS_NAMES[t][k], "kind": kind, "cov": R.loc[t, f"cov_{kk}_{CLASS_NAMES[t][k]}"]}
                          for k in range(K) for kind, kk in kinds])
        grouped_bars(axes[1, c], d, "class", "kind", "cov", None, 1 - ALPHA, f"{t}: coverage per true grade (dashed = target)", "coverage")
    axes[1, 0].legend(fontsize=7); save(fig, OUT / "conformal_sets.png")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))                                              # uncertainty vs error
    for ax, t in zip(axes, TASKS):
        x = X[t]; ok = x["okt"]; u = x["u"]["1-maxprob"][ok]; cr = x["correct"][ok]
        ax.boxplot([u[cr], u[~cr]]); ax.set_xticks([1, 2], ["correct", "wrong"]); ax.set(title=f"{t}: uncertainty (1 - max prob)")
    save(fig, OUT / "uncertainty_vs_error.png")


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    m = load_manifest(); Y, n_cls = targets(m); te = np.where(m["fold"] == -1)[0]
    src, src_why, health = choose_source(m, Y); log(f"Source: {src.name} ({src_why})")
    z = np.load(src, allow_pickle=True); is_v2 = "version" in z.files and str(z["version"]) == "v2"
    groups = m["group_id"].astype(str).to_numpy()[te] if "group_id" in m else m["image_key"].astype(str).to_numpy()[te]
    aligned = len(z["keys"]) == len(m) and (z["keys"].astype(str) == m["image_key"].astype(str).to_numpy()).all()
    if not aligned: raise RuntimeError("Prediction file rows do not match the Phase 1 manifest")
    D = {t: {"oof": z[f"oof_{t}"], "test": z[f"test_{t}"], "folds": z[f"test_folds_{t}"] if f"test_folds_{t}" in z.files else None} for t in TASKS}
    rows, X = [], {}
    for k, t in enumerate(TASKS):
        log(f"== {t}: calibration, conformal sets, selective prediction"); r, X[t] = process_task(t, n_cls[k], D[t], m, Y, te, groups); rows.append(r)
    R = pd.DataFrame(rows).set_index("task"); R.to_csv(OUT / "reliability_summary.csv")
    figures(R, X, n_cls)

    # per-embryo output for Phase 7 (review queue)
    pe = X["EXP"]["P"].argmax(1); card = pd.DataFrame({"image_key": m["image_key"].to_numpy()[te]})
    for t in TASKS:
        x = X[t]; defined = np.ones(len(te), bool) if t == "EXP" else pe >= 2          # ICM/TE graded only from EXP 3
        card[f"{t}_pred"] = np.where(defined, [CLASS_NAMES[t][i] for i in x["P"].argmax(1)], "ND")
        card[f"{t}_conf"] = x["P"].max(1).round(3)
        shown = {"marginal": x["set_m"], "classwise": x["set_c"], "gold": x["set_g"] if x["set_g"] is not None else x["set_c"]}[CARD_SETS]
        card[f"{t}_set"] = ["|".join(CLASS_NAMES[t][i] for i in np.where(s)[0]) for s in shown]      # 90% plausible grades (CARD_SETS)
        card[f"{t}_set_marginal"] = ["|".join(CLASS_NAMES[t][i] for i in np.where(s)[0]) for s in x["set_m"]]
        card[f"{t}_set_classwise"] = ["|".join(CLASS_NAMES[t][i] for i in np.where(s)[0]) for s in x["set_c"]]
        card[f"{t}_uncertain"] = defined & ~x["keep"]
    card["review_required"] = card[[f"{t}_uncertain" for t in TASKS]].any(axis=1)
    card["reasons"] = card.apply(lambda r: "; ".join(f"{t} uncertain (conf {r[f'{t}_conf']:.2f}, plausible {r[f'{t}_set']})"
                                                     for t in TASKS if r[f"{t}_uncertain"]), axis=1)
    card.to_csv(OUT / "reliability_test.csv", index=False)
    np.savez_compressed(OUT / "reliability_test.npz", test_idx=te, source=str(src), card_sets=CARD_SETS, **{f"P_{t}": X[t]["P"] for t in TASKS},
                        **{f"set_{t}": X[t]["set_m"] for t in TASKS}, **{f"set_classwise_{t}": X[t]["set_c"] for t in TASKS},
                        review=card["review_required"].to_numpy())

    # ---------------- verdict
    s5 = phase_status("phase5_outputs", "PHASE5_VERDICT.txt")
    best_u = {t: min((c for c in R.columns if c.startswith("aurc_") and c not in ("aurc_random", "aurc_oracle")), key=lambda c: R.loc[t, c]) for t in TASKS}
    checks = [("PASS" if s5 and "FAIL" not in s5 else "FAIL" if s5 else "WARN", "Phase 5 verdict", s5 or "not found"),
              ("PASS", "Predictions aligned with the Phase 1 manifest", f"{src.name}, {len(te)} gold-test embryos"),
              ("INFO", "Source of predictions", f"{src.relative_to(PROJECT_DIR)} - {src_why}"),
              ("PASS" if not any(h["collapsed"] for h in health.values()) else "FAIL", "Input predictions are not collapsed (validation)",
               " | ".join(f"{t} TV {h['tv']:.2f}" + (" COLLAPSED" if h["collapsed"] else "") for t, h in health.items())),
              ("PASS" if is_v2 or "phase8" in str(src) else "WARN", "Predictions come from Phase 5 v2", "yes" if is_v2 else "v1 file - rerun Phase 5 v2"),
              ("PASS" if (R["ece_after"] <= R["ece_before"] + .005).all() else "WARN", "Temperature scaling does not worsen calibration (test ECE)",
               " | ".join(f"{t} {R.loc[t, 'ece_before']:.3f}->{R.loc[t, 'ece_after']:.3f}" for t in TASKS)),
              ("PASS" if (abs(R["coverage_marginal_mean"] - (1 - ALPHA)) <= .05).all() else "INFO", f"Silver-calibrated conformal coverage of gold labels",
               " | ".join(f"{t} {R.loc[t, 'coverage_marginal_mean']:.3f}+-{R.loc[t, 'coverage_marginal_sd']:.3f}" for t in TASKS)
               + ("" if (abs(R["coverage_marginal_mean"] - (1 - ALPHA)) <= .05).all() else " (below target = silver/gold label shift)")),
              ("PASS" if (R["aurc_1-maxprob"] < R["aurc_random"]).all() else "WARN", "Abstaining on uncertain cases lowers error (AURC < random)",
               " | ".join(f"{t} {R.loc[t, 'aurc_1-maxprob']:.3f} vs {R.loc[t, 'aurc_random']:.3f}" for t in TASKS)),
              ("PASS" if (R["auroc_err_1-maxprob"] >= .65).all() else "WARN", "Uncertainty detects errors (AUROC >= 0.65)",
               " | ".join(f"{t} {R.loc[t, 'auroc_err_1-maxprob']:.3f}" for t in TASKS)),
              ("PASS" if (R["acc_selective"] > R["acc_all"]).all() else "WARN", f"Accuracy improves when keeping {TARGET_COVERAGE:.0%} most certain",
               " | ".join(f"{t} {R.loc[t, 'acc_all']:.3f}->{R.loc[t, 'acc_selective']:.3f}" for t in TASKS))]
    if "coverage_gold_crossfit" in R:
        checks.append(("PASS" if (R["coverage_gold_crossfit"] >= 1 - ALPHA - .03).all() else "WARN",
                       f"Gold-calibrated conformal coverage >= {1 - ALPHA - .03:.0%} (cross-fitted, grouped)",
                       " | ".join(f"{t} {R.loc[t, 'coverage_gold_crossfit']:.3f} (set {R.loc[t, 'set_size_gold_crossfit']:.2f})" for t in TASKS)))
    rare = {t: [c for c in R.columns if c.startswith("cov_cls_")] for t in TASKS}
    low = [f"{t}-{c[8:]} {R.loc[t, c]:.2f}" for t in TASKS for c in rare[t] if pd.notna(R.loc[t, c]) and R.loc[t, c] < 1 - ALPHA - .1]
    checks += [("PASS" if not low else "WARN", "Class-conditional conformal covers every grade (>= target - 0.10)", ", ".join(low) or "yes"),
               ("INFO", "Best uncertainty score per criterion (lowest AURC)", " | ".join(f"{t} {best_u[t][5:]}" for t in TASKS)),
               ("INFO", "Embryos sent to human review (gold test)", f"{int(card['review_required'].sum())} / {len(card)} ({card['review_required'].mean():.0%})")]
    keys = ["Criterion  ECE before->after  T     cov(marg)  set size  cov(class-cond)  AURC vs random  AUROC(err)  acc all->kept  QWK all->kept"]
    keys += [f"  {t:<8} {r.ece_before:.3f} -> {r.ece_after:.3f}  {r.T_mean:.2f}  {r.coverage_marginal_mean:.3f}      {r.set_size_marginal:.2f}      "
             f"{r.coverage_classwise_mean:.3f}            {r['aurc_1-maxprob']:.3f} / {r.aurc_random:.3f}   {r['auroc_err_1-maxprob']:.3f}       "
             f"{r.acc_all:.3f}->{r.acc_selective:.3f}  {r.qwk_all:.3f}->{r.qwk_selective:.3f}" for t, r in R.iterrows()]
    keys.append("Per-grade coverage (class-conditional): " + " | ".join(f"{t}: " + ", ".join(f"{c[8:]} {R.loc[t, c]:.2f}" for c in rare[t] if pd.notna(R.loc[t, c])) for t in TASKS))
    if "coverage_gold_crossfit" in R:
        keys.append("Gold-calibrated sets (cross-fitted): " + " | ".join(f"{t} coverage {R.loc[t, 'coverage_gold_crossfit']:.3f}, size "
                                                                       f"{R.loc[t, 'set_size_gold_crossfit']:.2f}" for t in TASKS))
    keys.append(f"source {src.name}, alpha {ALPHA}, target coverage {TARGET_COVERAGE}, card sets {CARD_SETS}, gold splits {GOLD_CAL_SPLITS}, "
                f"runtime {(time.time() - t0) / 60:.1f} min")
    verdict(6, checks, keys, OUT)
    return R


if __name__ == "__main__":
    run_safely(6, main)
