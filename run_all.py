"""MorphoGround v2 - run the whole pipeline in the correct order:  1 -> 2 -> 3 -> 5 -> 4 -> 6 -> 7 -> 8 -> 9
(Phase 4 analyses the checkpoints that Phase 5 saves, so Phase 5 must run before Phase 4.)
Each phase runs as its own Python process (fresh GPU memory each time). Its output is shown live and saved in run_logs/.
The run stops at the first phase that ends in FAIL, DID NOT RUN or NOT RUN YET, and prints a summary of all phases.
Put the nine phase*.py files (v2) in SCRIPTS_DIR.   Jupyter: paste into ONE cell.   Terminal: python run_all.py"""
from pathlib import Path

# ====================================== USER SETTINGS ========================================
# folder with phase1_data_audit.py ... phase9_final_validation.py: this file's folder (cloned repo), else ~/Embryo_Updated Project/code_v2
SCRIPTS_DIR  = Path(__file__).resolve().parent if "__file__" in globals() else Path.home() / "Embryo_Updated Project" / "code_v2"
ORDER        = [1, 2, 3, 5, 4, 6, 7, 8, 9]   # do not put 4 before 5
START_AT     = 1       # e.g. 5 = continue from Phase 5 when Phases 1-3 are already done
STOP_AFTER   = 9       # e.g. 5 = stop after Phase 5
SKIP         = []      # e.g. [7] to skip the vision-language phase
QUICK_TEST   = False   # True = smoke test of every phase (switched on in temporary copies; your files are not changed)
STOP_ON_WARN = False   # True = also stop at PASS WITH WARNINGS
# ============================================================================================

import re, subprocess, sys, time

LOG_DIR = SCRIPTS_DIR / "run_logs"
STOP = ("FAIL", "DID NOT RUN", "NOT RUN", "NO VERDICT")


def find_script(n):
    hits = sorted(p for p in SCRIPTS_DIR.glob(f"phase{n}_*.py") if not p.name.startswith("_"))
    if not hits: raise FileNotFoundError(f"No phase{n}_*.py in {SCRIPTS_DIR} - copy the nine v2 scripts there (or edit SCRIPTS_DIR)")
    if len(hits) > 1: raise RuntimeError(f"Several Phase {n} scripts in {SCRIPTS_DIR}: {[h.name for h in hits]} - keep only the v2 one")
    return hits[0]


def prepare(n, path):
    """QUICK_TEST: run a temporary copy with QUICK_TEST = True (the original file stays untouched)."""
    if not QUICK_TEST: return path
    txt, k = re.subn(r"^QUICK_TEST(\s*)=\s*False", r"QUICK_TEST\1= True", path.read_text(), count=1, flags=re.M)
    tmp = LOG_DIR / f"_quick_{path.name}"; tmp.write_text(txt)
    if not k: print(f"   (Phase {n} has no QUICK_TEST switch - it runs normally; it is fast)")
    return tmp


def run_phase(n, path):
    """Run one phase script, stream its output, save the log, and read its verdict line."""
    t0, lines, log = time.time(), [], LOG_DIR / f"phase{n}.log"
    proc = subprocess.Popen([sys.executable, "-u", str(path)], cwd=str(SCRIPTS_DIR), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    with open(log, "w", encoding="utf-8") as fh:
        for line in proc.stdout:
            print(line, end="", flush=True); fh.write(line); lines.append(line)
    rc = proc.wait()
    found = [re.search(r"VERDICT:\s+(.+?)(?:\s{2,}\(|\s*$)", l) for l in lines if "VERDICT:" in l]
    status = found[-1].group(1).strip() if found and found[-1] else ("DID NOT RUN" if rc else "NO VERDICT")
    if rc and not status.startswith(STOP): status = f"DID NOT RUN (exit code {rc})"
    return status, (time.time() - t0) / 60, log


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if ORDER.index(4) < ORDER.index(5): raise ValueError("Phase 4 must come after Phase 5 in ORDER")
    todo = [n for n in ORDER[ORDER.index(START_AT):ORDER.index(STOP_AFTER) + 1] if n not in SKIP]
    scripts = {n: find_script(n) for n in todo}                                 # fail early if a script is missing
    print(f"MorphoGround v2 pipeline: phases {todo} | quick test {QUICK_TEST} | python {sys.executable}\nLogs: {LOG_DIR}\n", flush=True)
    summary, stopped = [], None
    for n in todo:
        print("#" * 100 + f"\n# PHASE {n}: {scripts[n].name}\n" + "#" * 100, flush=True)
        status, mins, log = run_phase(n, prepare(n, scripts[n]))
        summary.append((n, status, mins, log))
        if status.startswith(STOP) or (STOP_ON_WARN and "WARN" in status):
            stopped = n; break
    bar = "=" * 100
    L = [bar, " PIPELINE SUMMARY" + (" (quick test)" if QUICK_TEST else ""), bar]
    L += [f"  Phase {n}: {s:<45} {m:6.1f} min   log: {lg.name}" for n, s, m, lg in summary]
    if stopped is not None:
        L += ["-" * 100, f"  Stopped at Phase {stopped}. Fix it (see its log and verdict above), then set START_AT = {stopped} and run again."]
    else:
        L += ["-" * 100, "  All requested phases finished." + (" Now set QUICK_TEST = False and run the full pipeline." if QUICK_TEST else "")]
    L += [" Send the PHASE VERDICT blocks (or the logs in run_logs/) back for review.", bar]
    txt = "\n".join(L); print("\n" + txt, flush=True); (LOG_DIR / "SUMMARY.txt").write_text(txt + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
