#!/usr/bin/env python
"""Re-time profanity candidates with a short local ASR pass.

Full-file faster-whisper word timestamps are unreliable for individual words:
a one-syllable "Fuck" can be reported as 1.4 s (starting a second before the
actual word), which makes the euphemism play early and leaves the real word
exposed. The same word transcribed from a 3-5 s clip is accurate (verified:
full pass 715.59-716.99 vs clip 716.30-716.90 for the same "Fuck").

This module takes the candidates found in the full transcript, re-listens to a
short window around each one, and replaces the candidate's time with the local
ASR time when the local pass hears the same profane word. Candidates whose
local pass does not hear the word keep their original time (the acoustic
refiner and the output verifier then cover them).

Usage:
    from retime import retime_matches
    matches = retime_matches(matches, audio, model)
"""
from __future__ import annotations

import numpy as np

from replacements import REPLACEMENTS, clean_word
from transcribe import VERBATIM_PROMPT

SR = 16000
CONTEXT_S = 3.0     # listen this far either side of the candidate
BUCKET_S = 4.0      # cache clips on a coarse grid


def _bucket_bounds(t: float, dur: float) -> tuple[float, float]:
    b = int(t // BUCKET_S)
    return max(0.0, b * BUCKET_S - 1.0), min(dur, (b + 1) * BUCKET_S + 1.0)


def retime_matches(matches: list[dict], audio: np.ndarray, model,
                   sr: int = SR) -> list[dict]:
    """Return matches with start/end replaced by local-ASR times when heard."""
    if not matches:
        return matches

    dur = len(audio) / sr
    cache: dict[tuple[int, int], list[dict]] = {}

    def local_words_for(t: float) -> list[dict]:
        key = (int(t // BUCKET_S),)
        if key in cache:
            return cache[key]
        t0, t1 = _bucket_bounds(t, dur)
        clip = audio[int(t0 * sr):int(t1 * sr)]
        words: list[dict] = []
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
                        words.append({"word": tok,
                                      "start": t0 + float(w.start),
                                      "end": t0 + float(w.end)})
        except Exception:
            pass
        cache[key] = (t0, t1, words)
        return cache[key]

    out: list[dict] = []
    for m in matches:
        mid = (float(m["start"]) + float(m["end"])) / 2.0
        t0, t1, words = local_words_for(mid)
        # Pick the profane token that OVERLAPS the candidate, not merely the
        # nearest one. A nearest-word search within 1.6 s jumped onto a
        # different occurrence of the same word ("shit!" at 1265.84 retimed to
        # a "shit" 1.5 s earlier) and the real word then had no window at all.
        c_start, c_end = float(m["start"]), float(m["end"])
        c_dur = max(0.05, c_end - c_start)
        best = None
        best_score = 0.0
        orig_key = clean_word(str(m.get("word", "")))
        for w in words:
            key = clean_word(w["word"])
            if key not in REPLACEMENTS and not (
                    key.endswith("'s") and key[:-2] in REPLACEMENTS):
                continue
            overlap = min(c_end, float(w["end"])) - max(c_start, float(w["start"]))
            if overlap <= 0.0:
                continue
            score = overlap / c_dur
            if score > best_score:
                best, best_score = w, score
        if best is not None and best_score >= 0.40 and best["end"] > best["start"]:
            b_start = float(best["start"])
            b_end = float(best["end"])
            # A clip pass can occasionally over-extend (e.g. merge two words).
            # Keep it under 1.6 s; the acoustic refiner trims/extends from the
            # waveform afterwards. A token far longer than the word can be said
            # is a merge ("I... OH SHIT" came back as one 1.76 s "shit,"), so
            # trust the full-file position instead of the merged span.
            from refine import expected_ms
            exp = expected_ms(best["word"]) / 1000.0
            cap = 1.60
            if b_end - b_start > min(cap, exp * 1.8 + 0.30):
                out.append(dict(m))
                continue
            if b_end - b_start > cap:
                b_end = b_start + cap
            out.append({**m,
                        "start": round(b_start, 3),
                        "end": round(max(b_end, b_start + 0.06), 3),
                        "word": best["word"]})
        else:
            out.append(dict(m))

    # Two candidates can collapse onto the same local token; keep one window so
    # the splicer does not speak the euphemism twice in place.
    out.sort(key=lambda x: float(x["start"]))
    dedup: list[dict] = []
    for m in out:
        if (dedup and m["replacement"] == dedup[-1]["replacement"]
                and abs(float(m["start"]) - float(dedup[-1]["start"])) < 0.05
                and abs(float(m["end"]) - float(dedup[-1]["end"])) < 0.05):
            continue
        dedup.append(m)
    return dedup
