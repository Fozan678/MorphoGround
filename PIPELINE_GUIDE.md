# MorphoGround v2 — rerun guide

All nine scripts are replaced. Each one is still self-contained: paste it into one Jupyter cell, or run it with `python`.

## Before you start

1. **Rename the old result folders.** In `~/Embryo_Updated Project`, rename `phase1_outputs` … `phase9_outputs` to `phase1_outputs_v1` … `phase9_outputs_v1`. This keeps the v1 results for the paper's "as-run" comparison and stops old prediction files from mixing with new ones. Phase 9 would flag them as misaligned.
2. **Keep `cache/`.** The image cache, frozen features, frozen tokens and adapted MAE encoders are now checked against the manifest rows by image name. Anything stale is rebuilt automatically.
3. **FEMI (optional).** Open huggingface.co/ihlab/FEMI and press "Agree and access repository". Wait for the owners' approval, then paste a read token into `HF_TOKEN` at the top of Phase 2 (or Phase 3). The token is stored, so later phases reuse it. Without it, FEMI is skipped with a WARN, as before.

## Easiest way: `run_all.py`

1. Put the nine v2 `phase*.py` files and `run_all.py` in one folder, e.g. `~/Embryo_Updated Project/code_v2` (that is the `SCRIPTS_DIR` default).
2. Run `run_all.py` once with `QUICK_TEST = True`, then again with `QUICK_TEST = False`. Use a single Jupyter cell or `python run_all.py`.

The runner always runs Phase 5 before Phase 4. Each phase runs in its own process, so GPU memory is freed between phases. Output streams live and is saved in `code_v2/run_logs/`. The run stops at the first phase that fails, and a summary is printed at the end. To continue after fixing something, set `START_AT` to that phase.

## Run order

Run in this order: **1 → 2 → 3 → 5 → 4 → 6 → 7 → 8 → 9**. Phase 4 needs the new Phase 5 checkpoints. If Phase 4 runs first, it now stops with a message listing what is missing instead of a traceback.

Phases 4, 5, 7 and 8 contain new PyTorch code that could not be executed in my environment. Run each of them once with `QUICK_TEST = True`, then again with `False`.

| Phase | Quick test first | Approx. time (v1 runs) | What must be true in the verdict |
|---|---|---|---|
| 1 | no | 1–2 min | `File-name prefix behaves like a patient/cycle ID` PASS; `CV folds leakage-free … patient/cycle keys across folds = 0` PASS |
| 2 | optional | ~37 min | as before; `All backbones ran` (FEMI only if access was granted) |
| 3 | optional | ~32 min (+MAE retraining only if the SSL image set changed) | `mae_in1k` and `mae_in1k_dapt` are both evaluated (no `encoder` crash); `Domain adaptation improves mae_in1k` line present |
| 5 | **yes** | ~30 min | `No ordinal collapse (validation, every variant)` PASS; `Criteria read few tokens`; `EXP / ICM / TE evidence maps differ` |
| 4 | **yes** | ~5–15 min (estimate) | faithful, criterion-specific (strict margin), grounded vs unconstrained comparison |
| 6 | no | <1 min | source = `morphoground.npz`; `Input predictions are not collapsed` PASS; `Gold-calibrated conformal coverage` |
| 7 | **yes** | ~15–25 min with the 7B VLM (estimate) | `structured` row: contradictions and hallucinations ≈ 0%, complete ≥ 90% |
| 8 | **yes** | ~8 min | `No variant collapsed`; ICM-C screening lines |
| 9 | no | <1 min | `Main models are not collapsed` PASS; sibling-status line |

## Where each issue is fixed and checked

| Issue | Fix | Where to see it |
|---|---|---|
| CORAL threshold collapse | CORN ordinal heads; collapse check every epoch, used in model selection; FAIL item | Phase 5 verdict, `collapse_check.json`, `grade_distribution_check.png`; Phases 6/8/9 re-check every input |
| Patient-level leakage | Folds grouped by file-name prefix (tested on the data: embryo numbers run 1..n in 99.9% of prefixes; contrast ICC 0.57 and sharpness ICC 0.53 within prefixes, permutation p < 0.005) | Phase 1 verdict and report; Phase 9 `tables/sibling_stratified.csv` |
| Phase 4 never ran | Rebuilt to load the v2 checkpoints; also tests the unconstrained token model for comparison | Phase 4 verdict, `grounded_vs_unconstrained.png` |
| Phase 3 MAE crash / FEMI | Version-robust block lookup and token reading; `HF_TOKEN` setting | Phase 3 verdict |
| Phases 6–7 on broken predictions | Phase 6 picks a healthy source automatically; Phase 7 takes grades and evidence maps from that same model | Phase 6 `Source of predictions`; Phase 7 `Evidence maps from the same model` |
| Conformal coverage below 90% | Adds gold-calibrated sets: each gold embryo calibrated on the other gold embryos, grouped by patient | Phase 6 verdict. On your current global-CE predictions this already gives EXP 0.909, ICM 0.916, TE 0.897 |
| Duplicate-label bug ("4.0" vs "4") | Canonical label comparison | Phase 1 `tables/duplicate_pairs.csv` should list 7 disagreeing pairs, not 15 |
| Uniform criterion attention | Single-head criterion attention with a learned temperature, plus two losses for sharp and criterion-specific evidence; optional mask supervision (`MASKS_DIR`) | Phase 5 token statistics; Phase 4 specificity margins |
| Weak explanations / ineffective repair | Qwen2.5-VL-7B; deterministic sentence-level verify-and-edit; `structured` mode (template facts + checked VLM notes) | Phase 7 verdict. Replaying your v1 texts through the new editor: 100% pass the checker (was 39%) |
| Phase 8 CORAL artifact; ICM-C recall 0.06 | CORN; collapsed variants cannot be recommended; nested validation-tuned offsets; per-grade AUROC; ICM-C screening with a 20% review budget | Phase 8 verdict, `rare_screening.csv` |

## A finding to expect for ICM-C

On your v1 predictions, the models rank ICM-C embryos well (AUROC ≈ 0.8), but gold-consensus ICM-C embryos get a much lower P(C) than silver ICM-C embryos in validation (median 0.16 vs 0.39). The consensus panel appears to call "C" on embryos a single annotator grades B. So a threshold learned on silver labels does not transfer. Flagging the 20% most C-like embryos for review caught 65% of gold ICM-C cases. This is a label-shift result worth reporting, not only a model weakness.

## Settings you may want to change

- **Phase 1** `TEST_SIBLINGS = "exclude"`: sibling-free protocol. It removes 627 of 2,040 training images, so use it as a sensitivity run.
- **Phase 5** `GROUND_W`: weights of the grounding losses. `CRIT_GLOBAL`: whether criteria may read the global token. `MASKS_DIR`: compartment masks for some training images, if you have them.
- **Phase 6** `CARD_SETS`: which conformal sets appear on the Phase 7 card (`classwise`, `marginal` or `gold`).
- **Phase 7** `VLM_ID` / `LOAD_4BIT`: for GPUs under ~20 GB, use `LOAD_4BIT = True` or the 3B model.
- **Phase 8** `SCREEN_BUDGET`: share of embryos flagged for rare-grade review.

## What was and was not tested before delivery

- **Run on data:**
  - Phase 1, end to end on a synthetic dataset with NNNN_EE names (both sibling policies, and the old no-prefix naming).
  - Phase 6 and Phase 9, on your real v1 outputs.
  - The Phase 7 checker and editor, on your real v1 explanations (self-test 10/10).
  - Phase 8 offset tuning and screening, on your real v1 predictions.
  - The prefix evidence test, on your real manifest.
- **Not executable in my environment (no PyTorch):** the training/inference parts of Phases 2–5, 7 and 8. They pass syntax and undefined-name checks. Please use `QUICK_TEST` and send me the verdict blocks, or the traceback if anything fails.

After the reruns, the paper's analysis scripts (`posthoc_analysis.py`, `make_assets.py`) need the new model names and results. Send the nine verdict blocks and I'll update them and the paper.
