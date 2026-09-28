# MorphoGround

**Grounded, ordinal and reliable human blastocyst grading under noisy and imbalanced labels**

MorphoGround grades day-5 human blastocysts with the Gardner system: expansion (EXP 1–5), inner cell mass (ICM A–C) and trophectoderm (TE A–C).
It is evaluated on the public Kromp et al. (2023) benchmark with a leakage-aware, reliability-centred protocol.

This repository contains:
- **The nine-phase pipeline.** Each phase is one self-contained Python script that ends with a PASS / WARN / FAIL verdict.
- **The complete results of the reported run.** Verdicts, summaries and predictions; no images.
- **The CVPR-style paper.** Every number in it is generated from those results by a script.

## Method in one paragraph
A DINOv2 ViT-B/14 encodes the image into patch tokens. Six learned morphology queries read the patches, and a relational transformer mixes them with the global token. One single-head query per Gardner criterion then reads the tokens. It feeds a CORN ordinal head and yields an intrinsic evidence map.

Five label-free losses ground the tokens and the criterion evidence: diversity, compactness, photometric consistency, criterion-attention entropy and criterion distinctness. A collapse guard rejects training epochs whose ordinal predictions degenerate.

Post hoc, three further layers are added:
- **A reliability layer:** temperature scaling, abstention, and conformal sets calibrated on silver labels or on consensus labels.
- **An evidence-grounded explanation step:** an evidence card, plus Qwen2.5-VL-7B with template-constrained generation.
- **A rule-based factuality checker** that verifies every generated sentence.

## Key results (locked 300-embryo gold test)

| System | QWK EXP | QWK ICM | QWK TE | Mean QWK |
|---|---|---|---|---|
| ResNet-50, linear probe | 0.839 | 0.194 | 0.336 | 0.456 |
| DINOv2, fine-tuned (best baseline) | 0.893 | 0.475 | 0.660 | 0.676 |
| Global token + CE | 0.899 | 0.429 | 0.682 | 0.670 |
| Global token + CORN | 0.895 | 0.508 | 0.639 | 0.680 |
| + morphology tokens + relational | 0.895 | 0.558 | 0.646 | 0.700 |
| **MorphoGround** (+ grounding) | **0.901** | 0.547 | 0.653 | **0.700** |

No pairwise difference survives Holm correction. 95% bootstrap CIs and paired tests are in `results/phase9_outputs/tables/`.

Other findings reported in the paper:
- **Patient structure:** file-name prefixes behave as patient/cycle IDs. Embryo numbers run 1..n in 99.9% of 851 prefixes, and contrast clusters by prefix (ICC 0.57, p = 0.001). Folds are therefore grouped by patient/cycle.
- **Ordinal collapse:** a CORAL ordinal head can collapse to the extreme grades (EXP accuracy 0.094 at QWK 0.60). CORN heads plus a collapse guard remove this failure.
- **Grounding:**
  - ICM evidence is faithful: occluding it lowers the ICM probability by 0.41, against 0.04 for random patches.
  - TE reuses the ICM evidence, which is a shortcut.
- **Reliability:**
  - Uncertainty detects errors, with AUROC 0.67–0.73.
  - Keeping the 80% most certain predictions raises ICM accuracy from 0.765 to 0.831.
  - Conformal sets calibrated on consensus labels reach about 90% coverage.
- **Rare grades:** a silver→gold label shift explains missed ICM-C embryos (median predicted P(C) 0.64 in validation vs 0.03 on gold). A 20% review budget still finds 59% of them.
- **Explanations:** template-constrained explanations cut hallucinated clinical claims from 80% to 0%. The VLM itself adds little visual description.

## Repository layout
```
pipeline/   phase1_data_audit.py … phase9_final_validation.py, run_all.py, make_project_zip.py
results/    outputs of the reported run: verdicts, summaries, prediction files (.npz), tables (no images, no model weights)
paper/      CVPR-style LaTeX source (main.tex), tables/, figures/, numbers.tex, analysis/make_assets.py, compiled PDF
docs/       PIPELINE_GUIDE.md (run order, settings, what each verdict checks)
```

## Installation
```bash
git clone https://github.com/<user>/MorphoGround.git
cd MorphoGround
conda create -n morphoground python=3.12 -y && conda activate morphoground
pip install -r requirements.txt       # install the PyTorch build that matches your CUDA first: https://pytorch.org
```
Tested with Python 3.12.7, PyTorch 2.11.0 (CUDA 12.8), transformers 5.14.1 and timm 1.0.28 on one 32 GB GPU. The phases also run with transformers 4.x.

## Data
The dataset is **not** included. Download it from the authors' figshare record
([doi:10.6084/m9.figshare.20123153](https://doi.org/10.6084/m9.figshare.20123153)) and check its licence there.
Then set the three paths at the top of `pipeline/phase1_data_audit.py`:
- `IMAGES_DIR`
- `SILVER_FILE`
- `GOLD_FILE`

All outputs go to `~/Embryo_Updated Project/`; change `PROJECT_DIR` at the top of each phase to use another folder.

The optional FEMI encoder is gated on Hugging Face. Request access, then set `HF_TOKEN` in Phase 2 or 3.

## Running the pipeline
Run everything in the correct order (1 → 2 → 3 → 5 → 4 → 6 → 7 → 8 → 9). Do a quick smoke test first by setting `QUICK_TEST = True` in `run_all.py`:
```bash
python pipeline/run_all.py
```
Or run a single phase with `python pipeline/phaseN_*.py`, or by pasting it into one Jupyter cell.

Each phase prints a verdict block and saves it in its output folder. The run stops at the first FAIL. Approximate runtimes on one GPU:

| Phase | Time |
|---|---|
| 1 | 2 min |
| 2 | 24 min |
| 3 | 15 min |
| 5 | 29 min |
| 4 | 1 min |
| 6 | < 1 min |
| 7 | 7 min |
| 8 | 6 min |
| 9 | < 1 min |

| Phase | What it does |
|---|---|
| 1 | Data audit: label coding, QC, duplicates, patient/cycle grouping from file names, leakage-free 5-fold split, locked gold test |
| 2 | Baseline ladder: ResNet-50 / ViT-B/16 / DINOv2 (+ FEMI), frozen → partial → full fine-tuning |
| 3 | Self-supervised representations: domain-adaptive MAE, linear/kNN probes, label efficiency |
| 4 | Grounding tests: deletion, criterion specificity, token function vs. attention (optional masks) |
| 5 | MorphoGround ablation: global CE → CORN → tokens + relational → grounding; collapse guard |
| 6 | Reliability: temperature scaling, uncertainty, selective prediction, conformal sets (silver / gold-calibrated) |
| 7 | Evidence-grounded explanations with a VLM, rule-based factuality checker, verify-and-edit, structured mode |
| 8 | Rare grades: loss / sampling recipes, tuned decision offsets, rare-grade screening |
| 9 | Final validation: bootstrap CIs, paired tests, sibling-stratified results, expert-rating kit |

## Reproducing the paper
To rebuild every number, table and data figure from the included results, or from your own run:
```bash
python paper/analysis/make_assets.py results                           # or: "~/Embryo_Updated Project"
cd paper && latexmk -pdf main.tex
```
The compiled paper is at `paper/MorphoGround_paper.pdf`.

## Citation

## Licence

