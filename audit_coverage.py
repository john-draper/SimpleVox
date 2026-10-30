#!/usr/bin/env python
"""Coverage audit: is every profane word covered by a censored window?

This does not listen to the censored output (TTS euphemisms are often
mis-transcribed back into profanity, which makes output-ASR audits noisy).
Instead it builds an independent list of spoken profanity from two sources:

  * the embedded subtitle track (human-authored; catches words Whisper drops)
  * the raw full-file transcript (catches profanity missing from subtitles)

and checks each occurrence lies inside a replacement window. A word whose
span pokes outside a window is a real leak: that audio was copied through
untouched.

Usage:
    python audit_coverage.py <raw.mkv> <replacements.json> <words.json> [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from oracle import _profane_tokens, batch_local_words, extract_cues
from refine import expected_ms
from replacements import REPLACEMENTS, clean_word, _resolve_key
from transcribe import VERBATIM_PROMPT, decode_audio_16k

SR = 16000


def _local_words(audio, model, t0: float, t1: float) -> list[dict]:
    clip = audio[int(t0 * SR):int(t1 * SR)]
    out: list[dict] = []
    try:
        segs, _ = model.transcribe(
            clip, language="en", vad_filter=False, beam_size=5,
            word_timestamps=True, condition_on_previous_text=False,
            initial_prompt=VERBATIM_PROMPT,
        )
        for s in segs:
            for w in (s.words or []):
                tok = (w.word or "").strip()
                if tok:
                    out.append({"word": tok, "start": t0 + float(w.start),
                                "end": t0 + float(w.end)})
    except Exception:
        pass
    return out


def subtitle_occurrences(cues, audio, model) -> list[dict]:
    occ: list[dict] = []
    profane = []
    for cue in cues:
        hits = _profane_tokens(cue["text"])
        if hits:
            profane.append((cue, hits))
    if not profane:
        return occ
    local_sets = batch_local_words(
        audio, model, [(c["start"], c["end"]) for c, _ in profane])
    for (cue, hits), local in zip(profane, local_sets):
        used: set[int] = set()
        for pos, tok, rep in hits:
            best_i, best_d = -1, 1.0
            for i, lw in enumerate(local):
                if i in used:
                    continue
                lkey = clean_word(lw["word"])
                lrep = _resolve_key(lkey)
                if lrep is None and lkey.endswith("'s"):
                    lrep = _resolve_key(lkey[:-2])
                if lrep != rep:
                    continue
                d = abs((lw["start"] + lw["end"]) / 2 - (cue["start"] + cue["end"]) / 2)
                if d < best_d:
                    best_i, best_d = i, d
            if best_i >= 0:
                lw = local[best_i]
                used.add(best_i)
                st, en = float(lw["start"]), float(lw["end"])
                if en <= st:
                    en = st + 0.30
                en = min(en, st + expected_ms(tok) / 1000.0 + 0.45)
            else:
                span = max(0.2, cue["end"] - cue["start"])
                st = cue["start"] + (pos / max(1, len(cue["text"]))) * span
                en = min(cue["end"], st + 0.45)
            occ.append({"word": tok, "start": st, "end": en,
                        "replacement": rep, "source": "subtitle"})
    return occ


def transcript_occurrences(words) -> list[dict]:
    occ = []
    for w in words:
        key = clean_word(w["word"])
        rep = _resolve_key(key)
        if rep is None and key.endswith("'s"):
            rep = _resolve_key(key[:-2])
        if rep is not None:
            occ.append({"word": w["word"], "start": float(w["start"]),
                        "end": float(w["end"]), "replacement": rep,
                        "source": "asr"})
    return occ


def covered(occ: dict, windows: list[dict], tol: float = 0.10) -> bool:
    """True if the occurrence lies inside the UNION of the windows.

    Two adjacent windows (e.g. "God" then "damn") each cover part of the same
    utterance; the word is still fully replaced because the union is
    continuous. Checking a single window would flag that as a leak.
    """
    spans = sorted((float(w["start"]) - tol, float(w["end"]) + tol)
                   for w in windows)
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    st, en = occ["start"], occ["end"]
    return any(a <= st and b >= en for a, b in merged)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("replacements")
    ap.add_argument("words")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    from transcribe import load_model
    model = load_model("medium", "cuda")
    audio = decode_audio_16k(args.raw)
    windows = json.load(open(args.replacements, encoding="utf-8"))
    words = json.load(open(args.words, encoding="utf-8"))

    cues = extract_cues(args.raw)
    occ = subtitle_occurrences(cues, audio, model) + transcript_occurrences(words)
    # Deduplicate occurrences from both sources (same word/time).
    occ.sort(key=lambda x: x["start"])
    merged = []
    for o in occ:
        if merged and abs(o["start"] - merged[-1]["start"]) < 0.20 \
                and o["replacement"] == merged[-1]["replacement"]:
            merged[-1]["end"] = max(merged[-1]["end"], o["end"])
            continue
        merged.append(o)

    leaks = [o for o in merged if not covered(o, windows)]
    print(f"{len(cues)} cues, {len(merged)} profane occurrences, "
          f"{len(windows)} windows, {len(leaks)} uncovered")
    for o in leaks:
        print(f"  LEAK {o['start']:8.2f}-{o['end']:8.2f} "
              f"{o['word']!r:16} ({o['source']}) -> {o['replacement']}")
    if args.json:
        json.dump({"occurrences": merged, "leaks": leaks},
                  open(args.json, "w", encoding="utf-8"), indent=1)
    return 0 if not leaks else 1


if __name__ == "__main__":
    sys.exit(main())
