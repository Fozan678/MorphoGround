"""MorphoGround - PHASE 7 (v2): evidence-grounded morphology explanations with a vision-language model (Qwen2.5-VL).
Conditions (same embryos, same evidence card = calibrated grades, conformal plausible grades, evidence regions, review flag):
  prediction_only                 - the VLM only gets the grades;
  evidence_grounded               - free text from the image + evidence maps + card (v1 prompt);
  evidence_grounded+verify_edit   - v2: deterministic sentence-level repair of that text (contradicting / forbidden sentences are
                                    deleted, missing card facts and the review statement are appended); no extra VLM call;
  structured                      - v2: the card facts are written by a template, the VLM only adds one visual description per
                                    criterion, and each description is kept only if the checker finds nothing wrong with it.
Every text is scored by the same rule-based factuality checker (now sentence-level; self-test 10 cases).
v2 also uses a larger VLM by default and takes the evidence maps from the same model as the grades (the Phase 6 source).
Self-contained: paste into ONE Jupyter cell, or run  python phase7_explanations.py"""
from pathlib import Path

# ====================================== USER SETTINGS ========================================
PROJECT_DIR = Path.home() / "Embryo_Updated Project"   # same folder as Phases 1-3
PHASE1_DIR  = PROJECT_DIR / "phase1_outputs"
CACHE_DIR   = PROJECT_DIR / "cache"                     # image cache + adapted encoders from earlier phases
IMG_SIZE, NUM_WORKERS = 224, 4
QUICK_TEST  = False   # True = fast smoke test first

# ---- PHASE 7 settings
VLM_ID         = "Qwen/Qwen2.5-VL-7B-Instruct"   # v2 default (v1 used Qwen2-VL-2B); smaller GPU: "Qwen/Qwen2.5-VL-3B-Instruct"
LOAD_4BIT      = False    # True = 4-bit weights (needs bitsandbytes) for GPUs with < 20 GB
N_EXPLAIN      = 100      # gold-test embryos to explain (300 = all; half are review-required cases when available)
MAX_NEW_TOKENS = 200
CONDITIONS     = ["prediction_only", "evidence_grounded", "structured"]   # evidence_grounded also gives +verify_edit (no extra call)
REPAIR_LLM     = False    # v1 LLM rewrite round (left 94% of texts unchanged with the 2B model); True = also report it
EVIDENCE_FILE  = "auto"   # auto = the prediction file Phase 6 used, so grades and evidence maps come from the same model
RELIABILITY    = "phase6_outputs/reliability_test.csv"
OUT            = PROJECT_DIR / "phase7_outputs"
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


def log(msg): print(f"{time.strftime('%H:%M:%S')} | {msg}", flush=True)


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


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


# ##################################### PHASE 7 #####################################
import io, re, textwrap  # noqa: E402
if QUICK_TEST: N_EXPLAIN = 8
GARDNER = ("Gardner criteria - Expansion: 1 early blastocyst (blastocoel < half), 2 blastocoel >= half, 3 full, 4 expanded "
           "(zona thinning), 5 hatching, 6 hatched. ICM: A many cells tightly packed; B several cells loosely grouped; C very few cells. "
           "TE: A many cells forming a cohesive epithelium; B few cells forming a loose epithelium; C very few large cells. "
           "ICM and TE are not defined at expansion 1-2.")
NAMES = {"EXP": "expansion", "ICM": "inner_cell_mass", "TE": "trophectoderm"}
READ = {"EXP": "Expansion", "ICM": "Inner cell mass (ICM)", "TE": "Trophectoderm (TE)"}
KW = {"EXP": r"\b(?:expansion|expanded|blastocoel\w*|expan\w*)\b", "ICM": r"\b(?:inner cell mass|icm)\b", "TE": r"\b(?:trophectoderm|te)\b"}
GRADE_RE = {"EXP": KW["EXP"] + r"[^.;]{0,40}?\b([1-6])\b",
            "ICM": r"\b(?i:" + KW["ICM"][2:-2] + r")\b[^.;]{0,60}?\b\(?([ABC])\)?(?![a-z])",
            "TE": r"\b(?i:" + KW["TE"][2:-2] + r")\b[^.;]{0,60}?\b\(?([ABC])\)?(?![a-z])"}
LEX = {"ICM": {"A": [r"tightly packed", r"many cells"], "B": [r"loosely grouped", r"several cells"], "C": [r"very few cells"]},
       "TE": {"A": [r"cohesive", r"many cells"], "B": [r"loose epithelium", r"(?<!very )few cells"], "C": [r"very few large cells", r"few large cells"]}}
FORBIDDEN = r"(pregnan|implant|live birth|ploid|genetic|chromosom|transfer|viab|success rate|prognos|pgt|miscarr|fragment)"
DEFER = r"(embryologist|expert|human|manual)?\s*(review|verif|confirm|double-check)"
REGION = r"(upper|lower|left|right|central|centre|center|middle|periph|rim|edge|top|bottom|region|area)"
NEG = r"\b(not|no|lack\w*|without)\b[^.]{0,15}$"


def conf_word(p): return "high" if p >= .8 else "moderate" if p >= .6 else "low"


def region_of(emap):
    g = math.isqrt(len(emap)); e = emap.reshape(g, g); top = e >= np.quantile(e, .85); ys, xs = np.where(top)
    v, h = ["upper", "central", "lower"][min(2, int(ys.mean() / g * 3))], ["left", "central", "right"][min(2, int(xs.mean() / g * 3))]
    where = "central" if v == h == "central" else v if h == "central" else h if v == "central" else f"{v}-{h}"
    return f"{where} region, {'concentrated' if e[top].sum() / (e.sum() + 1e-9) > .45 else 'diffuse'}"


def build_card(r, emaps):
    """Evidence card = the only facts the VLM may use (predictions, calibrated confidence, plausible set, evidence region)."""
    card = {}
    for c, t in enumerate(TASKS):
        if r[f"{t}_pred"] == "ND": card[NAMES[t]] = {"grade": "not defined (early blastocyst)"}; continue
        card[NAMES[t]] = {"grade": r[f"{t}_pred"], "confidence": f"{conf_word(r[f'{t}_conf'])} ({r[f'{t}_conf']:.2f})",
                          "plausible_grades": r[f"{t}_set"].split("|"),
                          **({"evidence_region": region_of(emaps[c])} if emaps is not None else {})}
    card["review_required"] = bool(r["review_required"])
    if card["review_required"]: card["review_reasons"] = r["reasons"]
    return card


def split_sents(text): return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text).strip()) if s.strip()]


def sentence_issues(s, card):
    """v2: contradictions and forbidden claims of ONE sentence (the unit of the deterministic verify-and-edit step)."""
    iss, hall, stated = [], [], {}
    crit_in_s = [u for u in TASKS if re.search(KW[u], s, re.I)]
    for t in crit_in_s:
        c = card[NAMES[t]]; defined = "confidence" in c
        mm = re.search(GRADE_RE[t], s, re.I if t == "EXP" else 0); g = mm.group(1) if mm else None
        if g is not None:
            stated[t] = g
            if not defined and t != "EXP": iss.append(f"{t} graded although not defined at this expansion")
            elif defined and g not in c["plausible_grades"]: iss.append(f"{t} stated {g}, card {c['grade']} (plausible {c['plausible_grades']})")
        if defined and crit_in_s == [t]:
            lvl = c["confidence"].split()[0]
            w = re.search(r"\b(high|moderate|low)(?:ly)?\s+(?:level of\s+)?confiden", s, re.I)
            if w and w.group(1).lower() != lvl: iss.append(f"{t} confidence '{w.group(1)}' vs card '{lvl}'")
            for grade, pats in LEX.get(t, {}).items():
                for p in pats:
                    m2 = re.search(p, s, re.I)
                    if m2 and not re.search(NEG, s[:m2.start()], re.I) and grade not in c["plausible_grades"]:
                        iss.append(f"{t} described as '{m2.group(0)}' (implies {grade})")
    compact = re.search(r"\b([1-6])([ABC])([ABC])\b", s)                           # e.g. "4AB"
    if compact:
        for t, gi in zip(TASKS, (1, 2, 3)):
            g, c = compact.group(gi), card[NAMES[t]]
            if t in stated: continue
            stated[t] = g
            if "confidence" not in c and t != "EXP": iss.append(f"{t} graded although not defined at this expansion")
            elif "confidence" in c and g not in c["plausible_grades"]: iss.append(f"{t} stated {g}, card {c['grade']} (plausible {c['plausible_grades']})")
    hall += [mm.group(0) for mm in re.finditer(FORBIDDEN, s.lower())]
    if re.search(r"hatch", s, re.I) and card["expansion"].get("grade", "0")[0] in "1234": hall.append("hatching (EXP < 5)")
    return iss, hall, stated


def check_text(text, card):
    """Rule-based factuality check of one explanation against its evidence card (sentence by sentence)."""
    sents = split_sents(text); low = str(text).lower()
    out = {"contradictions": [], "hallucinations": [], "stated": {t: None for t in TASKS}, "attributed": {}}
    for s in sents:
        iss, hall, st = sentence_issues(s, card); out["contradictions"] += iss; out["hallucinations"] += hall
        for t, g in st.items():
            if out["stated"][t] is None: out["stated"][t] = g
    graded = [t for t in TASKS if "confidence" in card[NAMES[t]]]
    for t in graded: out["attributed"][t] = any(re.search(KW[t], s, re.I) and re.search(REGION, s, re.I) for s in sents)
    defers = bool(re.search(DEFER, low))
    out["deferral_ok"] = defers if card["review_required"] else True
    out["overcautious"] = defers and not card["review_required"]
    out["complete"] = all(out["stated"][t] is not None for t in graded)
    out["attribution"] = float(np.mean([out["attributed"][t] for t in graded])) if graded else 1.0
    out["n_words"] = len(str(text).split()); out["ok"] = not out["contradictions"] and not out["hallucinations"] and out["deferral_ok"]
    return out


def fact_sentence(t, c):
    """Template sentence carrying one criterion's card facts (grade, calibrated confidence, region, alternatives)."""
    w, p = c["confidence"].split(" ", 1)
    s = f"{READ[t]}: grade {c['grade']} ({w} confidence, {p.strip('()')})"
    if "evidence_region" in c: s += f", evidence in the {c['evidence_region']}"
    alt = [g for g in c["plausible_grades"] if g != c["grade"]]
    return s + (f"; it could also be grade {' or '.join(alt)}." if alt else ".")


def review_sentence(card):
    unc = [{"EXP": "expansion", "ICM": "inner cell mass", "TE": "trophectoderm"}[t] for t in TASKS if f"{t} uncertain" in card.get("review_reasons", "")]
    return "Embryologist review is required because the grading is uncertain" + (f" for the {' and '.join(unc)}." if unc else ".")


def undefined_sentence(card):
    return f"Inner cell mass and trophectoderm are not graded at expansion {card['expansion']['grade']} (early blastocyst)."


def verify_edit(text, card):
    """v2 deterministic verify-and-edit: delete every sentence with a contradiction or a forbidden claim, then append the
    card facts that are still missing (grade/confidence/region of a graded criterion, the review statement)."""
    sents = split_sents(text)
    if sents and not sents[-1].endswith((".", "!", "?", ")")): sents = sents[:-1] + [""]   # generation cut off mid-sentence
    keep = [s for s in sents if s and not any(sentence_issues(s, card)[:2])]
    if not card["review_required"]: keep = [s for s in keep if not re.search(DEFER, s, re.I)]
    out = " ".join(keep); chk = check_text(out, card); add = []
    for t in TASKS:
        c = card[NAMES[t]]
        if "confidence" in c and (chk["stated"][t] is None or not chk["attributed"].get(t, True)): add.append(fact_sentence(t, c))
    if not chk["deferral_ok"]: add.append(review_sentence(card))
    return " ".join([out] + add).strip(), len(sents) - len(keep)


def parse_notes(text):
    """VLM output for the structured condition -> {criterion: visual note}; tolerant to code fences and loose JSON."""
    notes, t = {}, re.sub(r"```(?:json)?", "", str(text))
    mm = re.search(r"\{.*\}", t, re.S)
    if mm:
        try: d = json.loads(mm.group(0)); notes = {k: str(v) for k, v in d.items() if isinstance(v, (str, int, float))}
        except Exception: pass  # noqa: BLE001
    if not notes:
        for k in NAMES.values():
            m2 = re.search(rf'"?{k}"?\s*[:=]\s*"?([^"\n]+)', t, re.I)
            if m2: notes[k] = m2.group(1).strip()
    return notes


def structured_text(notes, card):
    """v2 structured explanation: the template states every card fact; the VLM contributes one visual note per criterion,
    kept only if it states no grade, no confidence contradiction, no forbidden claim and no unrequested review advice."""
    out, used, rejected = [], 0, 0
    for t in TASKS:
        c = card[NAMES[t]]
        if "confidence" not in c:
            if t == "ICM": out.append(undefined_sentence(card))
            if t != "EXP": continue
        else: out.append(fact_sentence(t, c))
        for s in split_sents(notes.get(NAMES[t], ""))[:2]:
            iss, hall, st = sentence_issues(s, card)
            bad = iss or hall or st or re.search(DEFER, s, re.I) or re.search(r"\bgrade\b|confiden", s, re.I)
            if bad: rejected += 1
            else: out.append(s if s.endswith((".", "!", "?")) else s + "."); used += 1
    if card["review_required"]: out.append(review_sentence(card))
    return " ".join(out), used, rejected


def checker_selftest():
    card = {"expansion": {"grade": "4", "confidence": "high (0.91)", "plausible_grades": ["4"], "evidence_region": "upper-left region"},
            "inner_cell_mass": {"grade": "A", "confidence": "moderate (0.70)", "plausible_grades": ["A", "B"], "evidence_region": "central region"},
            "trophectoderm": {"grade": "B", "confidence": "low (0.52)", "plausible_grades": ["B", "C"], "evidence_region": "lower-right region"},
            "review_required": True, "review_reasons": "TE uncertain (conf 0.52, plausible B|C)"}
    good = ("Expansion is grade 4 with high confidence, based on the upper-left region. The ICM is grade A with moderate confidence, "
            "centred in the central region, and it could also be grade B. The TE is grade B with low confidence in the lower-right rim, "
            "and could also be C. The embryo requires embryologist review because the TE grade is uncertain.")
    bad = good.replace("ICM is grade A", "ICM is grade C") + " It has high implantation potential."
    notes = {"expansion": "The blastocoel fills the embryo and the zona looks thin.", "inner_cell_mass": "It is grade C with very few cells.",
             "trophectoderm": "Cells along the rim look irregular."}
    cases = [(good, lambda o: o["ok"] and o["complete"] and o["attribution"] == 1),
             (good.replace("ICM is grade A", "ICM is grade C"), lambda o: any("ICM stated C" in x for x in o["contradictions"])),
             (good + " It has high implantation potential.", lambda o: bool(o["hallucinations"])),
             (good.replace(" The embryo requires embryologist review because the TE grade is uncertain.", ""), lambda o: not o["deferral_ok"]),
             (good.replace("with moderate confidence", "with very few cells"), lambda o: any("implies C" in x for x in o["contradictions"])),
             (good.replace("grade 4 with high confidence", "grade 4 with low confidence"), lambda o: any("confidence" in x for x in o["contradictions"])),
             (good.replace("The ICM is grade A", "The ICM is not tightly packed and is grade A"), lambda o: not any("tightly" in x for x in o["contradictions"])),
             (verify_edit(bad, card)[0], lambda o: o["ok"] and o["complete"] and o["attribution"] == 1),                      # v2
             (structured_text(notes, card)[0], lambda o: o["ok"] and o["complete"] and o["attribution"] == 1 and not o["overcautious"]),
             (structured_text({}, {**card, "inner_cell_mass": {"grade": "not defined (early blastocyst)"}, "trophectoderm": {"grade": "not defined (early blastocyst)"},
                                   "expansion": {**card["expansion"], "grade": "2", "plausible_grades": ["2"]}})[0], lambda o: o["ok"] and o["complete"])]
    fails = [i for i, (txt, ok) in enumerate(cases) if not ok(check_text(txt, card if i < 9 else
                                                                    {**card, "inner_cell_mass": {"grade": "not defined (early blastocyst)"},
                                                                     "trophectoderm": {"grade": "not defined (early blastocyst)"},
                                                                     "expansion": {**card["expansion"], "grade": "2", "plausible_grades": ["2"]}}))]
    return len(cases) - len(fails), len(cases), fails


def composite(img, emaps, title):
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.6)); axes[0].imshow(img); axes[0].set_title(title, fontsize=8); axes[0].axis("off")
    for ax, c, t in zip(axes[1:], range(3), ("expansion", "inner cell mass", "trophectoderm")):
        if emaps is None: ax.axis("off"); continue
        g = math.isqrt(emaps.shape[1]); e = emaps[c].reshape(g, g); e = (e - e.min()) / (e.max() - e.min() + 1e-9)
        ax.imshow(img); ax.imshow(np.asarray(Image.fromarray((e * 255).astype(np.uint8)).resize(img.shape[:2][::-1], Image.BICUBIC)), cmap="jet", alpha=.45)
        ax.set_title(f"{t} evidence", fontsize=8); ax.axis("off")
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=110); plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGB")


def load_vlm():
    import transformers
    from transformers import AutoProcessor
    names = ["Qwen2_5_VLForConditionalGeneration" if "2.5" in VLM_ID else "Qwen2VLForConditionalGeneration",
             "AutoModelForImageTextToText", "AutoModelForVision2Seq"]
    cls = next(c for c in (getattr(transformers, n, None) for n in names) if c is not None)
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if torch.cuda.is_available() else torch.float32
    kw = {"torch_dtype": dtype}
    if LOAD_4BIT:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4")
    try: model = cls.from_pretrained(VLM_ID, device_map="auto", **kw)
    except (ImportError, ValueError): model = cls.from_pretrained(VLM_ID, **kw).to("cuda" if torch.cuda.is_available() else "cpu")
    log(f"VLM loaded: {VLM_ID} ({next(model.parameters()).dtype}{', 4-bit' if LOAD_4BIT else ''})")
    return model.eval(), AutoProcessor.from_pretrained(VLM_ID, min_pixels=128 * 28 * 28, max_pixels=640 * 28 * 28)


@torch.no_grad()
def generate(model, proc, img, turns):
    """turns = [(role, text), ...]; the image is attached to the first user turn."""
    msgs, first = [], True
    for role, text in turns:
        content = [{"type": "text", "text": text}]
        if role == "user" and first: content, first = [{"type": "image"}] + content, False
        msgs.append({"role": role, "content": content})
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[prompt], images=[img], return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def prompts(card):
    grades = {t: str(card[NAMES[t]]["grade"]) for t in TASKS}
    p_only = [("system", "You are an assistant for embryologists."),
              ("user", f"An AI model graded this blastocyst as expansion {grades['EXP']}, ICM {grades['ICM']}, TE {grades['TE']}. "
                       "Write a short explanation (at most 5 sentences) of this assessment for an embryologist.")]
    e_grounded = [("system", "You write short, factual, evidence-grounded morphology explanations of human blastocyst images for "
                             "embryologists. You follow the rules exactly."),
                  ("user", "The image shows the blastocyst (left) and heat maps of the image regions the grading model used for "
                           f"expansion, inner cell mass (ICM) and trophectoderm (TE).\n{GARDNER}\n"
                           f"EVIDENCE CARD (the only source of grades):\n{json.dumps(card, indent=1, default=str)}\n"
                           "Rules: 1) State each criterion's grade exactly as in the card; never change a grade. 2) For each graded criterion "
                           "give its confidence word and its evidence region from the card. 3) If a criterion has more than one plausible grade, "
                           "say it could also be the other grade(s). 4) If review_required is true, say clearly that the embryo requires "
                           "embryologist review and why. 5) Do not mention pregnancy, implantation, transfer, viability, ploidy, genetics or "
                           "prognosis. 6) At most 5 sentences.")]
    graded = [NAMES[t] for t in TASKS if "confidence" in card[NAMES[t]]]
    region = {NAMES[t]: card[NAMES[t]].get("evidence_region", "whole embryo") for t in TASKS}
    structured = [("system", "You describe what is visible in human blastocyst images. You never assign grades."),
                  ("user", "The image shows the blastocyst (left) and heat maps of the regions a grading model used. "
                           f"For each of {', '.join(graded)}, write ONE short sentence (at most 20 words) describing what is visible in its "
                           f"highlighted region ({json.dumps({k: region[k] for k in graded})}): cell number, packing, epithelium, cavity size "
                           "or zona. Do NOT state grades, letters, numbers, confidence, or any clinical outcome, and do not recommend review. "
                           'Answer ONLY with JSON like {"expansion": "...", "inner_cell_mass": "...", "trophectoderm": "..."} using only '
                           f"these keys: {graded}.")]
    return {"prediction_only": p_only, "evidence_grounded": e_grounded, "structured": structured}


def summarise(E):
    rows = []
    for cond, d in E.groupby("condition", sort=False):
        rv = d[d["review_required"]]
        rows.append({"condition": cond, "n": len(d), "contradiction_rate": (d["n_contradictions"] > 0).mean(),
                     "hallucination_rate": (d["n_hallucinations"] > 0).mean(),
                     "deferral_compliance": rv["deferral_ok"].mean() if len(rv) else np.nan, "overcautious_rate": d["overcautious"].mean(),
                     "completeness": d["complete"].mean(), "attribution": d["attribution"].mean(), "all_checks_ok": d["ok"].mean(),
                     "mean_words": d["n_words"].mean(), "vlm_sentences_removed": d["vlm_removed"].mean(), "vlm_notes_used": d["vlm_used"].mean()})
    return pd.DataFrame(rows).set_index("condition")


def figures(S, E, comp_imgs):
    d = S[["contradiction_rate", "hallucination_rate", "deferral_compliance", "completeness", "attribution", "all_checks_ok"]].reset_index() \
        .melt(id_vars="condition", var_name="metric")
    fig, ax = plt.subplots(figsize=(14, 4.2))
    grouped_bars(ax, d, "metric", "condition", "value", None, None, "Factuality of generated explanations (lower is better for the first two)", "rate")
    ax.legend(fontsize=8); save(fig, OUT / "factuality_comparison.png")
    keys = list(comp_imgs)[:4]
    if not keys: return
    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 6.2 * len(keys)), squeeze=False)
    for ax, k in zip(axes[:, 0], keys):
        ax.imshow(comp_imgs[k]); ax.axis("off"); txt = []
        for cond in [c for c in ("prediction_only", "evidence_grounded", "structured") if (E["condition"] == c).any()]:
            r = E[(E["image_key"] == k) & (E["condition"] == cond)].iloc[0]
            flag = "OK" if r["ok"] else f"issues: {r['issues']}"
            txt.append(f"[{cond}] ({flag})\n" + "\n".join(textwrap.wrap(r["text"], 150)))
        ax.set_title("\n\n".join(txt), fontsize=7, loc="left", y=-0.02, va="top")
    save(fig, OUT / "example_explanations.png")


def row(k, cond, card, text, chk, removed=0, used=0):
    return {"image_key": k, "condition": cond, "review_required": card["review_required"], "text": text,
            "n_contradictions": len(chk["contradictions"]), "n_hallucinations": len(chk["hallucinations"]), "deferral_ok": chk["deferral_ok"],
            "overcautious": chk["overcautious"], "complete": chk["complete"], "attribution": chk["attribution"], "ok": chk["ok"],
            "n_words": chk["n_words"], "vlm_removed": removed, "vlm_used": used,
            "issues": "; ".join(chk["contradictions"] + chk["hallucinations"] + ([] if chk["deferral_ok"] else ["missing review statement"]))}


def main():
    t0 = time.time(); seed_all(); OUT.mkdir(parents=True, exist_ok=True)
    passed, total, fails = checker_selftest(); log(f"Factuality checker self-test: {passed}/{total}")
    m = load_manifest(); imgs = image_cache(m); te = np.where(m["fold"] == -1)[0]; keys_te = m["image_key"].astype(str).to_numpy()[te]
    rel_f, rel_npz = PROJECT_DIR / RELIABILITY, PROJECT_DIR / RELIABILITY.replace(".csv", ".npz")
    if EVIDENCE_FILE == "auto":                                                    # grades and evidence maps from the SAME model
        ev_f = Path(str(np.load(rel_npz, allow_pickle=True)["source"])) if rel_npz.exists() else PROJECT_DIR / "phase5_outputs/preds/morphoground.npz"
    else: ev_f = PROJECT_DIR / EVIDENCE_FILE
    if not ev_f.exists(): raise FileNotFoundError(f"{ev_f} not found - run Phases 5 and 6 first")
    z = np.load(ev_f, allow_pickle=True); emaps = z["evidence"] if "evidence" in z.files else None
    if rel_f.exists():
        card_df, rel_src = pd.read_csv(rel_f, keep_default_na=False, dtype=str), "Phase 6 (calibrated, conformal sets, review flags)"
        card_df["review_required"] = card_df["review_required"].astype(str).eq("True")
        for t in TASKS: card_df[f"{t}_conf"] = card_df[f"{t}_conf"].astype(float)
    else:                                                                          # fallback: raw Phase 5 probabilities
        rel_src = "Phase 5 only (Phase 6 not found: no calibration or conformal sets)"; card_df = pd.DataFrame({"image_key": keys_te})
        pe = z["test_EXP"].argmax(1)
        for t in TASKS:
            P = z[f"test_{t}"]; defined = np.ones(len(te), bool) if t == "EXP" else pe >= 2
            card_df[f"{t}_pred"] = np.where(defined, [CLASS_NAMES[t][i] for i in P.argmax(1)], "ND")
            card_df[f"{t}_conf"], card_df[f"{t}_set"] = P.max(1), [CLASS_NAMES[t][i] for i in P.argmax(1)]
            card_df[f"{t}_uncertain"] = defined & (P.max(1) < .6)
        card_df["review_required"] = card_df[[f"{t}_uncertain" for t in TASKS]].any(axis=1)
        card_df["reasons"] = card_df.apply(lambda r: "; ".join(f"{t} uncertain" for t in TASKS if r[f"{t}_uncertain"]), axis=1)
    if (card_df["image_key"].astype(str).to_numpy() != keys_te).any(): raise RuntimeError("Phase 5/6 test rows do not match the manifest")

    rng = np.random.default_rng(0); rv = np.where(card_df["review_required"])[0]; ok = np.where(~card_df["review_required"])[0]
    n_rv = min(len(rv), N_EXPLAIN // 2); sel = np.concatenate([rng.choice(rv, n_rv, replace=False), rng.choice(ok, min(len(ok), N_EXPLAIN - n_rv), replace=False)])
    log(f"Loading {VLM_ID}"); model, proc = load_vlm()
    rows, cards, comp_imgs = [], [], {}
    for n, i in enumerate(sel):
        r = card_df.iloc[i]; card = build_card(r, None if emaps is None else emaps[i]); k = r["image_key"]
        img = np.array(imgs[te[i]]); comp = composite(img, None if emaps is None else emaps[i], k); comp_imgs[k] = comp
        cards.append({"image_key": k, **card}); P_ = prompts(card)
        for cond in CONDITIONS:
            if cond == "prediction_only":
                text = generate(model, proc, Image.fromarray(img), P_[cond]); rows.append(row(k, cond, card, text, check_text(text, card)))
            elif cond == "evidence_grounded":
                text = generate(model, proc, comp, P_[cond]); chk = check_text(text, card); rows.append(row(k, cond, card, text, chk))
                ed, removed = verify_edit(text, card)                                        # v2: deterministic verify-and-edit
                rows.append(row(k, "evidence_grounded+verify_edit", card, ed, check_text(ed, card), removed=removed))
                if REPAIR_LLM and not chk["ok"]:                                              # v1-style LLM rewrite (optional)
                    probs = chk["contradictions"] + [f"forbidden topic: {h}" for h in chk["hallucinations"]] + \
                        ([] if chk["deferral_ok"] else ["you did not say that embryologist review is required"])
                    t2 = generate(model, proc, comp, P_[cond] + [("assistant", text), ("user", "Your explanation has these problems: " + "; ".join(probs) +
                                                                                       ". Rewrite it following all rules and the evidence card exactly.")])
                    rows.append(row(k, "evidence_grounded+llm_repair", card, t2, check_text(t2, card)))
                elif REPAIR_LLM: rows.append(row(k, "evidence_grounded+llm_repair", card, text, chk))
            elif cond == "structured":
                raw = generate(model, proc, comp, P_[cond]); notes = parse_notes(raw); text, used, rej = structured_text(notes, card)
                rows.append({**row(k, cond, card, text, check_text(text, card), removed=rej, used=used), "vlm_raw": raw})
        if n % 10 == 0: log(f"   explained {n + 1}/{len(sel)} embryos")
    E = pd.DataFrame(rows); E.to_csv(OUT / "explanations.csv", index=False)
    with open(OUT / "evidence_cards.jsonl", "w") as fh:
        for c in cards: fh.write(json.dumps(c, default=str) + "\n")
    S = summarise(E); S.to_csv(OUT / "factuality_summary.csv"); figures(S, E, comp_imgs)

    # ---------------- verdict
    s5, s6 = phase_status("phase5_outputs", "PHASE5_VERDICT.txt"), phase_status("phase6_outputs", "PHASE6_VERDICT.txt")
    fin = next(c for c in ("structured", "evidence_grounded+verify_edit", "evidence_grounded") if c in S.index)
    g, p = S.loc[fin], S.loc["prediction_only"] if "prediction_only" in S.index else S.iloc[0]
    nrv = int(E[E["condition"] == fin]["review_required"].sum())
    checks = [("PASS" if s5 and "FAIL" not in s5 else "FAIL" if s5 else "WARN", "Phase 5 verdict", s5 or "not found"),
              ("PASS" if s6 and "FAIL" not in s6 else "WARN", "Phase 6 reliability inputs", rel_src if s6 else f"{rel_src}; Phase 6 verdict not found"),
              ("PASS" if emaps is not None else "WARN", "Evidence maps from the same model as the grades",
               f"{ev_f.name}" if emaps is not None else f"{ev_f.name} has no evidence maps (text-only card) - use a token model in Phase 6"),
              ("PASS" if passed == total else "FAIL", "Factuality checker self-test", f"{passed}/{total} cases" + (f", failed {fails}" if fails else "")),
              ("PASS", "Explanations generated", f"{len(sel)} gold-test embryos x {E['condition'].nunique()} conditions with {VLM_ID}"),
              ("PASS" if g["contradiction_rate"] < max(p["contradiction_rate"], 1e-9) or g["contradiction_rate"] == 0 else "WARN",
               f"Contradictions: {fin} vs prediction-only", f"{p['contradiction_rate']:.1%} -> {g['contradiction_rate']:.1%}"),
              ("PASS" if g["hallucination_rate"] <= .05 else "WARN", "Hallucinated clinical claims <= 5%", f"{g['hallucination_rate']:.1%} (prediction-only {p['hallucination_rate']:.1%})"),
              ("PASS" if nrv == 0 or g["deferral_compliance"] >= .9 else "WARN", "Uncertain embryos are sent to review in the text (>= 90%)",
               f"{g['deferral_compliance']:.1%} of {nrv}" if nrv else "no review-required embryos in the sample"),
              ("PASS" if g["completeness"] >= .9 else "WARN", "Every graded criterion is stated (>= 90%)", f"{g['completeness']:.1%}"),
              ("PASS" if g["attribution"] >= .8 else "WARN", "Statements linked to an evidence region (>= 80%)", f"{g['attribution']:.1%}")]
    if "evidence_grounded+verify_edit" in S.index:
        a, b = S.loc["evidence_grounded", "all_checks_ok"], S.loc["evidence_grounded+verify_edit", "all_checks_ok"]
        checks.append(("PASS" if b > a or a == 1 else "WARN", "Deterministic verify-and-edit repairs free text (all checks ok)",
                       f"{a:.1%} -> {b:.1%}; {S.loc['evidence_grounded+verify_edit', 'vlm_sentences_removed']:.2f} sentences removed per text"))
    if "structured" in S.index:
        u, rj = S.loc["structured", "vlm_notes_used"], S.loc["structured", "vlm_sentences_removed"]
        checks.append(("PASS" if u >= 1 else "WARN", "Structured mode: VLM visual notes kept per embryo (>= 1)",
                       f"{u:.2f} kept, {rj:.2f} rejected by the checker (facts come from the card template)"))
    checks.append(("INFO", "How to report", "structured / verify_edit guarantee card-consistent grades by construction; the VLM's own "
                                            "descriptive accuracy needs the embryologist ratings (Phase 9)"))
    keys = ["Condition                         contradiction  hallucination  review-deferral  complete  attribution  all-ok  words  VLM kept/removed"]
    keys += [f"  {c:<32} {r.contradiction_rate:6.1%}        {r.hallucination_rate:6.1%}         {r.deferral_compliance:6.1%}         "
             f"{r.completeness:6.1%}    {r.attribution:6.1%}    {r.all_checks_ok:6.1%}  {r.mean_words:4.0f}   {r.vlm_notes_used:.2f}/{r.vlm_sentences_removed:.2f}"
             for c, r in S.iterrows()]
    keys.append(f"VLM {VLM_ID} (4-bit {LOAD_4BIT}), embryos {len(sel)} ({n_rv} review-required), conditions {CONDITIONS}, LLM repair {REPAIR_LLM}, "
                f"device {DEVICE}, runtime {(time.time() - t0) / 60:.1f} min")
    verdict(7, checks, keys, OUT)
    return S


if __name__ == "__main__":
    run_safely(7, main)
