#!/usr/bin/env python
r"""
Gap auditor: find profanity Whisper deleted from transcripts.

Whisper's politeness bias has two failure modes on the AUDIO side:
  1. DELETION - a short profane utterance is removed entirely, leaving a
     suspicious hole in the word timeline (S02E07 09:35 "What the fuck?"
     sits in a 4.1s gap between 'Wow.' and 'Yeah,').
  2. EUPHEMISM SWAP - the slot stays filled but with a clean word
     ("heck" where the audio may say the F-word).

This tool re-checks both against the audio with faster-whisper driven
DIRECTLY (whisperx's wrapper drops initial_prompt) using a verbatim prompt:

  * gaps: every interior silence in the word timeline between MIN_GAP and
    MAX_GAP seconds gets its own clip transcription; profanity tokens found
    there become "extra" replacement entries with absolute timestamps.
  * euphemism recheck: existing tokens that are themselves euphemisms
    (heck/darn/gosh...) are re-listened to; if the clip transcription
    clearly contains a dictionary profanity at that spot, it is reported
    (and included in extras only with --include-swaps, since real "heck"s
    exist in dialogue).

Extras JSONs are written next to the source transcripts as
<stem>_extras.json and are meant to be spliced onto the CURRENT live files
(already-censored audio) by censor_extras.py - only the newly found words
are replaced.

Usage:
  python audit_gaps.py "D:\Coding\SimpleVox\output\_intermediate\Rick and Morty (pass 2)" \
      "E:\Videos\Unedited\RnM_pass5\_intermediate" --min-gap 2.0 --max-gap 8.0
  python audit_gaps.py SRC DST --seasons 2          # one season
  python audit_gaps.py SRC DST --dry                # report only, no files
"""

import argparse
import json
import sys
from pathlib import Path

from batch_videos import VERBATIM_PROMPT
from replacements import find_matches, REPLACEMENTS

# Tokens worth re-listening to: our own replacement vocabulary appearing in a
# transcript transcribed from UNCENSORED audio is a swap tell (pass-2
# transcripts predate most censoring, so a "heck" there can be a swapped
# "hell"/"fuck").
EUPHEMISM_RECHECK = {"heck", "darn", "gosh", "dang", "freak", "freaking"}

MIN_CLIP_PROB = 0.4   # ignore clip words below this faster-whisper probability


def find_gaps(words: list[dict], min_gap: float, max_gap: float,
              regions: str = "interior", media_duration: float | None = None) -> list[tuple[float, float]]:
    """Silences to re-check, in (start, end) seconds.

    interior: between consecutive words, gap within [min_gap, max_gap].
    tail:     from the last word's end to the media end (post-credit scenes;
              S02E10's end-tag chant "get the fuck off" x4 lived here, past
              both the interior-gap logic and its 8s cap).
    head:     from 0 to the first word's start.
    """
    wanted = {r.strip() for r in regions.split(",")}
    gaps: list[tuple[float, float]] = []
    if "interior" in wanted:
        for a, b in zip(words, words[1:]):
            gap = b["start"] - a["end"]
            if min_gap <= gap <= max_gap:
                gaps.append((a["end"], b["start"]))
    if words:
        if "head" in wanted and words[0]["start"] >= min_gap:
            gaps.append((0.0, words[0]["start"]))
        if "tail" in wanted and media_duration is not None:
            tail = media_duration - words[-1]["end"]
            if tail >= min_gap:
                gaps.append((words[-1]["end"], media_duration))
    return gaps


def media_duration(path: Path) -> float | None:
    import subprocess
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=60)
        return float(r.stdout.strip())
    except Exception:
        return None


def clip_words(fw_model, media: Path, t0: float, t1: float) -> list[dict]:
    """Transcribe [t0, t1] of media with the verbatim prompt; absolute times."""
    import subprocess, tempfile
    import numpy as np
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        clip = Path(tf.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0, t0 - 0.4):.3f}",
             "-to", f"{t1 + 0.4:.3f}", "-i", str(media),
             "-vn", "-ac", "1", "-ar", "16000", str(clip)],
            check=True)
        from faster_whisper import WhisperModel  # noqa: F401  (already loaded instance passed in)
        segments, _ = fw_model.transcribe(str(clip), language="en", beam_size=5,
                                          initial_prompt=VERBATIM_PROMPT,
                                          word_timestamps=True, vad_filter=False)
        out = []
        offset = max(0, t0 - 0.4)
        for seg in segments:
            for w in seg.words or []:
                if w.probability is not None and w.probability < MIN_CLIP_PROB:
                    continue
                tok = w.word.strip()
                if tok:
                    out.append({"word": tok, "start": offset + w.start,
                                "end": offset + w.end})
        return out
    finally:
        clip.unlink(missing_ok=True)


def profane(word: str) -> bool:
    return bool(find_matches([{"word": word, "start": 0.0, "end": 1.0}]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("transcript_root", help="dir of Season N/<stem>.json word transcripts")
    ap.add_argument("out_root", help="destination for <stem>_extras.json")
    ap.add_argument("--media-root", default=r"E:\Videos\TVShows\Rick and Morty (2013)",
                    help="live library the transcripts' AUDIO came from... but note "
                         "clips must come from a file whose audio still HAS the "
                         "profanity (pass-2 era live files are gone); pass --media-root "
                         "explicitly when auditing against censored audio is not wanted")
    ap.add_argument("--min-gap", type=float, default=2.0)
    ap.add_argument("--max-gap", type=float, default=8.0)
    ap.add_argument("--seasons", default="all", help="comma list of season numbers")
    ap.add_argument("--regions", default="interior",
                    help="comma list: interior,tail,head (default interior)")
    ap.add_argument("--include-swaps", action="store_true",
                    help="also write euphemism-swap finds into extras")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--model", default="medium")
    args = ap.parse_args(argv)

    src = Path(args.transcript_root)
    out_root = Path(args.out_root)
    media_root = Path(args.media_root)
    seasons = None if args.seasons == "all" else {f"Season {s}" for s in args.seasons.split(",")}

    from faster_whisper import WhisperModel
    fw = WhisperModel(args.model, device="cuda", compute_type="float16")

    total_extras = total_swaps = total_gaps = 0
    for season_dir in sorted(d for d in src.iterdir() if d.is_dir()):
        if seasons and season_dir.name not in seasons:
            continue
        for j in sorted(season_dir.glob("*.json")):
            if j.name.endswith(("_replacements.json", "_extras.json")):
                continue
            words = json.loads(j.read_text(encoding="utf-8"))
            stem = j.stem
            # media file next to the live library layout
            media = None
            for ext in (".mp4", ".mkv"):
                cand = media_root / season_dir.name / f"{stem}{ext}"
                if cand.is_file():
                    media = cand
                    break
            if media is None:
                print(f"[skip] {season_dir.name}/{stem}: no media file")
                continue

            extras, swap_reports = [], []
            gaps = find_gaps(words, args.min_gap, args.max_gap,
                             regions=args.regions,
                             media_duration=media_duration(media))
            total_gaps += len(gaps)
            # long tails/heads: check in consecutive <=30s chunks
            chunked = []
            for t0, t1 in gaps:
                while t1 - t0 > 30.0:
                    chunked.append((t0, t0 + 30.0))
                    t0 = t0 + 30.0
                chunked.append((t0, t1))
            for t0, t1 in chunked:
                for w in clip_words(fw, media, t0, t1):
                    if profane(w["word"]):
                        m = find_matches([w])[0]
                        extras.append({**m, "source": "gap"})
            # euphemism swap recheck (report-only unless --include-swaps)
            for w in words:
                key = w["word"].strip().strip(".,!?\"'").lower()
                if key in EUPHEMISM_RECHECK:
                    for cw in clip_words(fw, media, w["start"] - 0.2, w["end"] + 0.2):
                        if profane(cw["word"]) and abs(cw["start"] - w["start"]) < 0.6:
                            swap_reports.append(
                                f"{int(w['start']//60):02d}:{w['start']%60:05.1f} "
                                f"transcript={w['word']!r} clip-heard={cw['word']!r}")
                            if args.include_swaps:
                                m = find_matches([cw])[0]
                                extras.append({**m, "source": "swap"})
                            break

            tag = f"{season_dir.name}/{stem}"
            if extras:
                print(f"[{tag}] {len(gaps)} gaps -> {len(extras)} EXTRA(S):")
                for e in extras:
                    print(f"    + {int(e['start']//60):02d}:{e['start']%60:06.3f} "
                          f"{e['word']!r} -> {e['replacement']!r} ({e['source']})")
                if not args.dry:
                    dst = out_root / season_dir.name
                    dst.mkdir(parents=True, exist_ok=True)
                    (dst / f"{stem}_extras.json").write_text(
                        json.dumps(extras, ensure_ascii=False, indent=2), encoding="utf-8")
                total_extras += len(extras)
            for line in swap_reports:
                print(f"    [swap?] {tag} {line}")
                total_swaps += 1

    print(f"\n[audit done] gaps checked: {total_gaps}, extras: {total_extras}, "
          f"swap suspects (report-only): {total_swaps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
