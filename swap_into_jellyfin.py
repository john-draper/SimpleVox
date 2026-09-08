#!/usr/bin/env python
"""
Swap freshly censored episodes into the live Jellyfin library.

Maps each censored output (by SxxEyy code) onto the live library file with the
same code, using the LIVE library's naming convention:

    Censored\\Season 15\\American Dad! - S15E09 - Title - [HDTV-1080p]h265 AAC.mkv
      -> live\\S15\\American Dad! S15E09 Title.mkv

Behavior:
  * Old live file is MOVED to the backup dir (nothing is deleted).
  * New file is MOVED into the live dir under the live filename.
  * Episodes only in the new set (e.g. S16E21/E22) are ADDED under live naming.
  * Episodes only in the live set (S17E23/24, S21E13-22) are left untouched.
  * Refuses to swap a file whose output is missing or suspiciously small.

Usage:
    python swap_into_jellyfin.py            # dry run (prints the plan)
    python swap_into_jellyfin.py --apply    # actually move files
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

CENSORED_ROOT = Path(r"E:\Videos\Unedited\Censored")
LIVE_ROOT = Path(r"E:\Videos\TVShows\American Dad! 2005.S01-S21.720p.DSNP.H264.x264.AAC-Zero00")
BACKUP_ROOT = Path(r"E:\Videos\Unedited\Replaced_Old_Censored")
MIN_OUTPUT_BYTES = 50 * 1024 * 1024  # a real 1080p episode is 150MB+


def code_of(name: str) -> str | None:
    m = re.search(r"(S\d+E\d+)", name)
    return m.group(1) if m else None


def title_from_live(name: str) -> str:
    m = re.match(r"American Dad! S\d+E\d+ (.+)\.mkv$", name)
    return m.group(1) if m else name


def ffprobe_ok(path: Path) -> bool:
    """True if the file probes cleanly with a plausible episode duration."""
    import subprocess
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30)
        dur = float(r.stdout.strip())
        return 10 * 60 < dur < 90 * 60
    except Exception:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Actually move files (default: dry run)")
    ap.add_argument("--no-backup", action="store_true",
                    help="Overwrite live files instead of moving them to the backup dir")
    ap.add_argument("--validate", action="store_true", default=True,
                    help="ffprobe each output (duration 10-90 min) before swapping")
    args = ap.parse_args(argv)

    # index live library by episode code
    live_by_code = {}
    for f in sorted(LIVE_ROOT.rglob("*.mkv")):
        c = code_of(f.name)
        if c:
            if c in live_by_code:
                print(f"[warn] duplicate code {c} in live library: {f}")
            live_by_code[c] = f

    # index censored outputs by code (skip _intermediate)
    new_by_code = {}
    for f in sorted(CENSORED_ROOT.rglob("*.mkv")):
        if "_intermediate" in f.parts:
            continue
        c = code_of(f.name)
        if c:
            if c in new_by_code:
                print(f"[warn] duplicate code {c} in censored set: {f}")
            new_by_code[c] = f

    live_codes = set(live_by_code)
    common = sorted(set(new_by_code) & live_codes,
                    key=lambda c: (int(c[1:3]), int(c[4:])))
    added = sorted(set(new_by_code) - live_codes,
                   key=lambda c: (int(c[1:3]), int(c[4:])))
    untouched = sorted(set(live_by_code) - set(new_by_code),
                       key=lambda c: (int(c[1:3]), int(c[4:])))

    print(f"censored outputs: {len(new_by_code)} | live episodes: {len(live_by_code)}")
    print(f"  replace: {len(common)}   add-new: {len(added)}   "
          f"live-kept-as-is: {len(untouched)}")

    problems = []
    plan = []  # (new_path, live_path, backup_path or None for overwrite)
    for c in common:
        new_f = new_by_code[c]
        live_f = live_by_code[c]
        size = new_f.stat().st_size
        if size < MIN_OUTPUT_BYTES:
            problems.append(f"{c}: output too small ({size / 1e6:.1f} MB) — {new_f.name}")
            continue
        if args.validate and not ffprobe_ok(new_f):
            problems.append(f"{c}: ffprobe validation failed — {new_f.name}")
            continue
        backup_f = None if args.no_backup else BACKUP_ROOT / live_f.parent.name / live_f.name
        plan.append((new_f, live_f, backup_f))
    for c in added:
        new_f = new_by_code[c]
        size = new_f.stat().st_size
        if size < MIN_OUTPUT_BYTES:
            problems.append(f"{c}: NEW output too small — skipped")
            continue
        if args.validate and not ffprobe_ok(new_f):
            problems.append(f"{c}: NEW ffprobe validation failed — skipped")
            continue
        m = re.search(r"S\d+E\d+ - (.+?)(?: - \[.*)?\.mkv$", new_f.name)
        title = (m.group(1) if m else "Episode").strip()
        live_f = LIVE_ROOT / f"S{int(c[1:3]):02d}" / f"American Dad! {c} {title}.mkv"
        plan.append((new_f, live_f, None))

    if untouched:
        print(f"episodes staying as old versions (not in new set): {untouched}")

    for new_f, live_f, backup_f in plan:
        if live_f.exists():
            if backup_f is not None:
                print(f"REPLACE {live_f.relative_to(LIVE_ROOT)}  "
                      f"(old -> {backup_f.relative_to(BACKUP_ROOT)})")
            else:
                print(f"OVERWRITE {live_f.relative_to(LIVE_ROOT)}")
        else:
            print(f"ADD     {live_f.relative_to(LIVE_ROOT)}")
        print(f"         from {new_f.relative_to(CENSORED_ROOT)}")

    if problems:
        print("\nPROBLEMS (not swapped):")
        for p in problems:
            print(f"  {p}")

    if not args.apply:
        print("\n[DRY RUN] nothing moved. Re-run with --apply to execute.")
        return 0

    print(f"\n[APPLY] moving {len(plan)} file(s)...")
    errors = 0
    for new_f, live_f, backup_f in plan:
        try:
            live_f.parent.mkdir(parents=True, exist_ok=True)
            if backup_f is not None:
                backup_f.parent.mkdir(parents=True, exist_ok=True)
                if backup_f.exists():
                    print(f"  [skip] backup already exists: {backup_f.name}")
                else:
                    shutil.move(str(live_f), str(backup_f))
                shutil.move(str(new_f), str(live_f))
            else:
                # overwrite mode: atomically replace the live file
                os.replace(str(new_f), str(live_f))
        except Exception as exc:
            errors += 1
            print(f"  [ERROR] {live_f.name}: {type(exc).__name__}: {exc}")
    print(f"[APPLY] done: {len(plan) - errors} swapped, {errors} errors")
    print(f"[APPLY] old files preserved under: {BACKUP_ROOT}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
