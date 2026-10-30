#!/usr/bin/env python
"""Repair leaks found by the coverage + audio-diff audit.

The audit lists every profane word in the episode (subtitles + raw transcript)
and measures how much of its span is still bit-identical to the raw audio in
the censored file. Anything with a substantial untouched fraction is a leak:
the window ended early / started late and part of the original word was copied
through.

Each round:
  1. widen the responsible window to the occurrence (or add a new one),
  2. regenerate TTS for the changed windows,
  3. re-splice from the raw file,
  4. re-audit.

Usage:
    python repair_leaks.py <raw> <censored> <replacements.json> <words.json> \
        --audio-dir <generated_audio> [--max-rounds 2]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
import _nowindow  # noqa: F401  (no console windows for child processes)

import numpy as np

from audit_coverage import (SR, covered, subtitle_occurrences,
                            transcript_occurrences)
from generate_audio import main as generate_main
from oracle import extract_cues
from splice_audio import main as splice_main
from transcribe import decode_audio_16k, load_model

UNTOUCHED_DIFF = 0.03
MIN_ENERGY = 0.003
LEAK_FRACTION = 0.20
PAD_S = 0.18


def decode_pcm(path: str) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0",
           "-ac", "1", "-ar", str(SR), "-f", "s16le",
           "-acodec", "pcm_s16le", "pipe:1"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True,
                                        check=True).stdout,
                         dtype=np.int16).astype(np.float32) / 32768.0


def untouched_fraction(raw: np.ndarray, cen: np.ndarray,
                       st: float, en: float) -> float:
    a, b = int(st * SR), int(en * SR)
    a = max(0, min(a, len(raw) - 1))
    b = max(a + 1, min(b, len(raw), len(cen)))
    r, c = raw[a:b], cen[a:b]
    k = 160
    m = (len(r) // k) * k
    if m < k:
        return 0.0
    rr = r[:m].reshape(-1, k)
    cc = c[:m].reshape(-1, k)
    # Untouched audio survives the lossy re-encode with the same waveform
    # shape (correlation ~1). Silence or TTS in the same slot has near-zero
    # correlation even when both signals are quiet, so compare shape rather
    # than absolute sample difference.
    num = (rr * cc).sum(axis=1)
    den = np.sqrt((rr ** 2).sum(axis=1) * (cc ** 2).sum(axis=1)) + 1e-9
    corr = num / den
    e = np.sqrt((rr ** 2).mean(axis=1))
    untouched = (corr > 0.90) & (e > MIN_ENERGY)
    return float(untouched.mean())


def _merge_occurrences(occ: list[dict]) -> list[dict]:
    occ = sorted(occ, key=lambda x: x["start"])
    merged: list[dict] = []
    for o in occ:
        if (merged and abs(o["start"] - merged[-1]["start"]) < 0.20
                and o["replacement"] == merged[-1]["replacement"]):
            merged[-1]["end"] = max(merged[-1]["end"], o["end"])
            continue
        merged.append(dict(o))
    return merged


def widen(matches: list[dict], occ: dict) -> bool:
    """Widen the window overlapping `occ`, or add a new one. Returns changed."""
    st, en = occ["start"] - PAD_S, occ["end"] + PAD_S
    best, best_ov = None, 0.0
    for m in matches:
        ov = min(en, m["end"]) - max(st, m["start"])
        if ov > best_ov:
            best, best_ov = m, ov
    if best is not None and best_ov > 0.05:
        new_start = min(best["start"], st)
        new_end = max(best["end"], en)
        if new_start < best["start"] - 0.001 or new_end > best["end"] + 0.001:
            best["start"] = round(max(0.0, new_start), 3)
            best["end"] = round(new_end, 3)
            best.pop("lead_in", None)
            return True
        return False
    matches.append({"word": occ["word"], "start": round(max(0.0, st), 3),
                    "end": round(en, 3), "replacement": occ["replacement"]})
    return True


def clamp(matches: list[dict]) -> None:
    matches.sort(key=lambda m: float(m["start"]))
    prev = 0.0
    for m in matches:
        if m["start"] < prev:
            m["start"] = round(prev, 3)
        m["end"] = round(max(m["end"], m["start"] + 0.06), 3)
        prev = m["end"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("censored")
    ap.add_argument("replacements")
    ap.add_argument("words")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--voice", default="en-US-GuyNeural")
    args = ap.parse_args(argv)

    model = load_model("medium", "cuda")
    raw = decode_pcm(args.raw)
    audio = decode_audio_16k(args.raw)
    cues = extract_cues(args.raw)
    words = json.load(open(args.words, encoding="utf-8"))
    matches = json.load(open(args.replacements, encoding="utf-8"))
    occ = _merge_occurrences(
        subtitle_occurrences(cues, audio, model) + transcript_occurrences(words))

    for rnd in range(1, args.max_rounds + 1):
        cen = decode_pcm(args.censored)
        leaks = []
        for o in occ:
            frac = untouched_fraction(raw, cen, o["start"], o["end"])
            # A word is a leak if a meaningful part of it is untouched, OR if
            # its span is not fully inside a window at all (a short onset/tail
            # can be <20% of the word yet still audible).
            if frac >= LEAK_FRACTION or not covered(o, matches):
                o = dict(o)
                o["untouched"] = round(frac, 2)
                leaks.append(o)
        print(f"[round {rnd}] {len(occ)} occurrences, {len(leaks)} leak(s)")
        for o in leaks:
            print(f"  {o['start']:8.2f}-{o['end']:8.2f} {o['word']!r:16} "
                  f"untouched {o['untouched']:.0%}")
        if not leaks:
            print("CLEAN")
            return 0

        changed = False
        for o in leaks:
            changed |= widen(matches, o)
        if not changed:
            print("no windows changed; stopping")
            return 1

        clamp(matches)
        with open(args.replacements, "w", encoding="utf-8") as f:
            json.dump(matches, f, ensure_ascii=False, indent=1)
        rc = generate_main([args.replacements, "--output-dir", args.audio_dir,
                            "--voice", args.voice, "--skip-existing"])
        if rc != 0:
            print("TTS generation failed")
            return 2
        rc = splice_main([args.raw, "--replacements-json", args.replacements,
                          "--audio-dir", args.audio_dir,
                          "--output", args.censored])
        if rc != 0:
            print("splice failed")
            return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
