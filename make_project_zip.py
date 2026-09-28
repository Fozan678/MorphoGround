"""Zip the whole 'Embryo_Updated Project' folder so you can download it from Jupyter.
Paste into ONE Jupyter cell and run it. Download links appear under the cell. Files are saved next to your notebook.
Big projects are split into several independent zip parts, each opening on its own, because very large single
downloads through the browser often break.
MODE = "results": code, verdicts, tables, figures, logs and predictions (small; enough for the paper and for review).
MODE = "everything": also the cache (resized images, features, tokens), model checkpoints and adapted encoders (big)."""
from pathlib import Path
import fnmatch, os, shutil, time, zipfile

# ====================================== USER SETTINGS ========================================
PROJECT_DIR = Path.home() / "Embryo_Updated Project"
MODE        = "results"   # "results" (small) or "everything" (includes cache and model weights)
SPLIT_MB    = 1900        # largest size of one zip part in MB (None = one single zip, however big)
EXCLUDE     = []          # extra patterns to leave out, e.g. ["*.npy", "phase3_outputs/*"]
OUT_DIR     = Path.cwd()  # where the zip(s) are written; must be inside the Jupyter folder for the download link to work
DRY_RUN     = False       # True = only show what would be zipped and how big it is
# ============================================================================================

STAMP = time.strftime("%Y%m%d_%H%M")
ROOT = Path("Embryo_Updated Project")
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git"}
OWN_ZIPS = "Embryo_Updated_Project_*.zip"
WEIGHTS = ("*.pt", "*.pth", "*.safetensors", "*.bin", "*.ckpt")          # model weights (left out in "results" mode)
STORE = {".png", ".jpg", ".jpeg", ".npz", ".pt", ".pth", ".bin", ".safetensors", ".zip", ".gz", ".pdf"}   # already compressed


def mb(n): return f"{n / 1e6:,.1f} MB"


def reason_to_skip(rel, size):
    """Why a file is left out (None = keep)."""
    s = rel.as_posix()
    if any(fnmatch.fnmatch(s, pat) or fnmatch.fnmatch(rel.name, pat) for pat in EXCLUDE): return "EXCLUDE patterns"
    if MODE == "results":
        if rel.parts[0] == "cache": return "cache folder (re-created by rerunning the phases)"
        if any(fnmatch.fnmatch(rel.name, w) for w in WEIGHTS): return "model weights"
    return None


def collect():
    keep, skipped, empty = [], {}, []
    for dp, dn, fn in os.walk(PROJECT_DIR):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        rel_dir = Path(dp).relative_to(PROJECT_DIR)
        if not dn and not fn and rel_dir.parts and reason_to_skip(rel_dir / "_", 0) is None: empty.append(rel_dir)   # keep empty folders
        for f in fn:
            p = Path(dp) / f
            if fnmatch.fnmatch(f, OWN_ZIPS) or not p.is_file(): continue
            rel, size = p.relative_to(PROJECT_DIR), p.stat().st_size
            why = reason_to_skip(rel, size)
            if why: skipped[why] = (skipped.get(why, (0, 0))[0] + 1, skipped.get(why, (0, 0))[1] + size)
            else: keep.append((p, rel, size))
    return sorted(keep, key=lambda x: x[1].as_posix()), skipped, empty


def plan_parts(files):
    """Group files into parts of at most SPLIT_MB (a single larger file gets its own part)."""
    if not SPLIT_MB: return [files]
    parts, cur, cur_size, lim = [], [], 0, SPLIT_MB * 1e6
    for f in files:
        if cur and cur_size + f[2] > lim: parts.append(cur); cur, cur_size = [], 0
        cur.append(f); cur_size += f[2]
    return parts + ([cur] if cur else [])


def summary_by_folder(files):
    top = {}
    for _, rel, size in files:
        k = rel.parts[0] if len(rel.parts) > 1 else "(top level files)"; top[k] = top.get(k, 0) + size
    return top


def main():
    if not PROJECT_DIR.exists(): raise FileNotFoundError(f"{PROJECT_DIR} not found - check PROJECT_DIR at the top of this cell")
    if MODE not in ("results", "everything"): raise ValueError('MODE must be "results" or "everything"')
    files, skipped, empty = collect(); total = sum(f[2] for f in files); parts = plan_parts(files)
    print(f"Project: {PROJECT_DIR}\nMode: {MODE} | {len(files)} files, {mb(total)} before compression | {len(parts)} zip part(s)\n")
    print("Included, by folder:")
    for k, v in sorted(summary_by_folder(files).items(), key=lambda kv: -kv[1]): print(f"  {mb(v):>12}  {k}")
    if skipped:
        print("Left out:")
        for k, (n, v) in skipped.items(): print(f"  {mb(v):>12}  {n} files - {k}")
        if MODE == "results": print('  (set MODE = "everything" to include them)')
    free = shutil.disk_usage(OUT_DIR).free
    if total > free * .95: raise SystemExit(f"\nNot enough free disk space in {OUT_DIR}: need up to {mb(total)}, free {mb(free)}. "
                                            "Use MODE = 'results', add EXCLUDE patterns, or set OUT_DIR to a bigger disk.")
    if DRY_RUN: print("\nDRY_RUN = True: nothing written. Set DRY_RUN = False to make the zip."); return []

    out, t0 = [], time.time()
    for i, part in enumerate(parts, 1):
        name = f"Embryo_Updated_Project_{STAMP}" + (f"_part{i}of{len(parts)}" if len(parts) > 1 else "") + ".zip"
        zp = OUT_DIR / name; listing = [f"{name}: part {i} of {len(parts)}, made {time.strftime('%Y-%m-%d %H:%M')} from {PROJECT_DIR}", ""]
        print(f"\nWriting {name} ({len(part)} files, {mb(sum(f[2] for f in part))}) ...", flush=True)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
            for n, (p, rel, size) in enumerate(part, 1):
                z.write(p, (ROOT / rel).as_posix(), compress_type=zipfile.ZIP_STORED if p.suffix.lower() in STORE else zipfile.ZIP_DEFLATED)
                listing.append(f"{size / 1e6:10.2f} MB  {rel.as_posix()}")
                if n % 500 == 0: print(f"  {n}/{len(part)} files", flush=True)
            if i == 1:
                for d in empty: z.writestr((ROOT / d).as_posix() + "/", "")                  # empty folders too
            if i == 1 and skipped:
                listing += ["", "Left out:"] + [f"  {k}: {n} files, {mb(v)}" for k, (n, v) in skipped.items()]
            z.writestr((ROOT / f"ZIP_CONTENTS_part{i}.txt").as_posix(), "\n".join(listing))
        with zipfile.ZipFile(zp) as z: bad = z.testzip()                         # verify every part
        print(f"  saved {zp} - {mb(zp.stat().st_size)} - integrity {'OK' if bad is None else 'FAILED at ' + bad}", flush=True)
        out.append(zp)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min: {len(out)} file(s), {mb(sum(z.stat().st_size for z in out))} in total.")
    if len(out) > 1: print("Each part is a normal zip; unzip all parts into the same place to get the whole project folder back.")
    try:                                                                            # clickable download links in Jupyter
        from IPython.display import FileLink, display
        for zp in out: display(FileLink(os.path.relpath(zp, Path.cwd()), result_html_prefix="Download: "))
        print("If a link does not start the download: in the Jupyter file browser, right-click the zip -> Download.")
    except Exception:  # noqa: BLE001
        print("Not in Jupyter: copy the zip file(s) above from the paths shown.")
    return out


if __name__ == "__main__":
    main()
