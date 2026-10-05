#!/usr/bin/env python
r"""
Splice audit extras (<stem>_extras.json from audit_gaps.py) onto the CURRENT
live files. Only the newly found words are replaced - everything already
censored in earlier passes stays as-is, so no re-TTS of ~900 words and no
timestamp drift.

  python censor_extras.py "E:\Videos\Unedited\RnM_pass5\_intermediate" \
      "E:\Videos\Unedited\RnM_pass5" --live-root "E:\Videos\TVShows\Rick and Morty (2013)"

Outputs mirror the live layout under OUT_ROOT (Season N\<name>.<ext>).
Swapping into the live library is a separate step (see swap pattern in
handoff section 8); this script never touches the live files.
"""

import argparse
import json
import sys
from pathlib import Path

from batch_videos import generate_wavs
from splice_audio import splice_audio_ffmpeg

# Prompted clip transcription can hallucinate rapid profanity repeats into
# music/silence gaps (observed: five 'fuck' tokens inside 0.9s in a gap where
# the original full pass heard nothing at all). Genuine rapid-fire profanity
# exists but a >=3-token burst inside CLUSTER_WINDOW seconds in a previously
# wordless gap is far more likely hallucination: drop it and report, so a
# human (or a second listen) can overrule.
CLUSTER_MIN = 3          # tokens ...
CLUSTER_WINDOW = 2.0     # ... within this many seconds -> suspect


def split_suspect_clusters(extras: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (trusted, suspect) splitting hallucination-shaped clusters."""
    trusted, suspect, i = [], [], 0
    by_start = sorted(extras, key=lambda e: e["start"])
    while i < len(by_start):
        j = i
        while (j + 1 < len(by_start)
               and by_start[j + 1]["start"] - by_start[i]["start"] <= CLUSTER_WINDOW):
            j += 1
        group = by_start[i:j + 1]
        (suspect if len(group) >= CLUSTER_MIN else trusted).extend(group)
        i = j + 1
    return trusted, suspect


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("extras_root", help="dir of Season N/<stem>_extras.json")
    ap.add_argument("out_root")
    ap.add_argument("--live-root", default=r"E:\Videos\TVShows\Rick and Morty (2013)")
    ap.add_argument("--voice", default="en-US-GuyNeural")
    args = ap.parse_args(argv)

    extras_root = Path(args.extras_root)
    out_root = Path(args.out_root)
    live_root = Path(args.live_root)

    ok = failed = skipped = 0
    for extras_json in sorted(extras_root.rglob("*_extras.json")):
        season_dir = extras_json.parent.name          # "Season N"
        stem = extras_json.name[: -len("_extras.json")]
        live = None
        for ext in (".mp4", ".mkv"):
            cand = live_root / season_dir / f"{stem}{ext}"
            if cand.is_file():
                live = cand
                break
        if live is None:
            print(f"[skip] {season_dir}/{stem}: live media not found")
            skipped += 1
            continue

        extras = json.loads(extras_json.read_text(encoding="utf-8"))
        if not extras:
            continue
        trusted, suspect = split_suspect_clusters(extras)
        for e in suspect:
            print(f"    [suspect-dropped] {season_dir}/{stem} "
                  f"{int(e['start']//60):02d}:{e['start']%60:06.3f} {e['word']!r}")
        if not trusted:
            print(f"[skip] {season_dir}/{stem}: only suspect clusters, nothing trusted")
            skipped += 1
            continue
        extras = trusted
        audio_dir = out_root / "_intermediate" / season_dir / "generated_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        final = out_root / season_dir / live.name
        final.parent.mkdir(parents=True, exist_ok=True)

        n_ok, n_fail = generate_wavs(extras, audio_dir, args.voice)
        if n_fail:
            print(f"[FAIL] {season_dir}/{stem}: TTS failed for {n_fail} word(s)")
            failed += 1
            continue
        success = splice_audio_ffmpeg(
            audio_path=str(live), replacements=extras, audio_dir=audio_dir,
            output_path=str(final), lead_in_ms=None,
        )
        if success and final.is_file():
            print(f"[ok] {season_dir}/{live.name}: {len(extras)} extra(s) spliced")
            ok += 1
        else:
            print(f"[FAIL] {season_dir}/{stem}: splice/mux failed")
            failed += 1

    print(f"\n[extras done] spliced {ok}, failed {failed}, skipped {skipped}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
