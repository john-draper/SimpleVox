#!/usr/bin/env python
"""Acoustic refinement of profane-word windows before TTS generation.

Whisper's word timestamps are good for *finding* a word but unreliable for its
edges. A drawn-out "mother-fuckah!" can be reported as the first 280 ms only,
and a "Je-" onset can sit 250 ms before the reported start. Splicing the
euphemism into the reported span then leaves part of the profanity audible
("scoundrel ... fucker", "eum ... geez").

This pass expands each censored window to the speech it belongs to:

  * END   grows forward until 250 ms of quiet, or the next transcript word.
    This is where a drawn-out word's leaked tail lives.
  * START grows backward at most MAX_LEFT_S (150 ms) and never past the end of
    the preceding transcript word: any more would mute real dialogue. The
    euphemism itself is always spoken from the reported word onset.

The window never crosses the previous/next *replacement*, so windows cannot
overlap and the concat timeline stays exact. A `lead_in` marker is attached so
the splicer can keep the euphemism at the reported word start instead of
speaking it at the widened left edge.

Usage:
    from refine import refine_matches
    refine_matches(matches, words, audio)      # in-memory (run.py stage 2)
"""
from __future__ import annotations

import numpy as np

SR = 16000
FRAME = 0.01          # 10 ms analysis frames
QUIET_FRAMES = 25     # 250 ms of quiet ends a speech island
MAX_GROW_S = 1.5      # hard cap when the transcript has no next word
LEAD_IN_S = 0.12      # keep TTS at the word start while the window grows left
MAX_LEFT_S = 0.15     # window may grow at most this far left (into a pause)
END_GUARD_S = 0.10    # silence kept after the acoustic end of the word


def syllables(word: str) -> int:
    w = "".join(c for c in word.lower() if c.isalpha())
    if not w:
        return 1
    n, prev = 0, False
    for c in w:
        v = c in "aeiouy"
        if v and not prev:
            n += 1
        prev = v
    return max(1, n)


def expected_ms(word: str) -> int:
    """Expected spoken length for `word`, in milliseconds (diagnostics only)."""
    return min(1500, 250 + 160 * (syllables(word) - 1))


def _envelope(audio: np.ndarray) -> np.ndarray:
    n = int(SR * FRAME)
    usable = (len(audio) // n) * n
    if usable < n:
        return np.full(1, -120.0, dtype=np.float32)
    frames = audio[:usable].reshape(-1, n).astype(np.float32)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-6))


def _speech_end(env: np.ndarray, start_f: int, limit_f: int,
                thr: float) -> int:
    """Last frame before `limit_f` that is followed by <QUIET_FRAMES quiet."""
    j = start_f
    quiet = 0
    last = start_f
    while j < limit_f:
        if env[min(j, len(env) - 1)] < thr:
            quiet += 1
            if quiet >= QUIET_FRAMES:
                break
        else:
            quiet = 0
            last = j
        j += 1
    return last


def refine_matches(
    matches: list[dict],
    words: list[dict],
    audio: np.ndarray,
    sr: int = SR,
) -> list[dict]:
    """Return `matches` with start/end grown to the spoken speech island."""
    if not matches or audio.size == 0:
        return matches

    env = _envelope(audio)
    n_frames = len(env)
    total_s = len(audio) / sr
    word_spans = sorted((float(w["start"]), float(w["end"])) for w in words)

    out: list[dict] = []
    prev_end = 0.0
    for i, m in enumerate(sorted(matches, key=lambda x: float(x["start"]))):
        start = float(m["start"])
        end = float(m["end"])
        word = str(m.get("word", ""))

        # Neighbouring transcript words bound the growth. The immediately
        # preceding word usually ends exactly at (or a few ms after) `start`
        # because Whisper's words are contiguous, so the test has to be
        # "started before us", not "ended before us" -- an epsilon test let a
        # window grow back over a whole word ("We want A's," was erased by the
        # "bitches!" window).
        prev_word_end = 0.0
        for s, e in word_spans:
            if s <= start - 0.02 and e <= start + 0.06:
                prev_word_end = max(prev_word_end, e)
        nxt_word_start = total_s
        for s, e in word_spans:
            if s >= end - 0.01:
                nxt_word_start = min(nxt_word_start, s)
                break

        # Replacements never overlap: keep a small guard between windows.
        start_floor = max(prev_word_end, prev_end)
        end_ceil = min(nxt_word_start, start + MAX_GROW_S)
        if i + 1 < len(matches):
            end_ceil = min(end_ceil, float(matches[i + 1]["start"]))

        e0 = max(0, min(n_frames - 1, int(end / FRAME)))
        lo = max(0, int(start_floor / FRAME))
        hi = min(n_frames, int(end_ceil / FRAME))

        local = env[max(0, lo - 100):min(n_frames, hi + 100)]
        floor = float(np.percentile(local, 15)) if local.size else -120.0
        thr = floor + 7.0

        new_end_f = _speech_end(env, e0, hi, thr)

        # The window may only grow LEFT into an actual pause. Bounded by the
        # preceding word's end and by MAX_LEFT_S, so it can never eat dialogue:
        # when the neighbour is contiguous (its end == our start) nothing moves.
        new_start = max(start_floor, start - MAX_LEFT_S)

        # The END grows to the acoustic end of the speech island (that is where
        # the leaked tail of a drawn-out word lives), plus a small guard.
        acoustic_end = min(end_ceil, (new_end_f + 1) * FRAME)
        acoustic_end = max(end, acoustic_end)
        new_end = min(end_ceil, acoustic_end + END_GUARD_S)
        if new_end - new_start < 0.08:
            new_end = min(total_s, new_start + 0.15)

        # The euphemism is spoken over [start, speech_end]; the window's extra
        # lead/trail is silence. Keeping speech_end at the acoustic end (not at
        # the guarded window end) stops the euphemism from being stretched
        # through the following pause.
        speech_end = min(new_end, acoustic_end)

        lead = max(0.0, min(LEAD_IN_S, start - new_start))
        out.append({**m,
                    "start": round(new_start, 3),
                    "end": round(new_end, 3),
                    # Spoken span: starts at the (local-ASR) word onset and
                    # runs to the acoustic end found above. The splicer
                    # stretches the euphemism across this full span, so a
                    # drawn-out word does not get a short "freak" followed by
                    # silence, and pads only the window's extra lead with
                    # silence.
                    "speech_start": round(start, 3),
                    "speech_end": round(speech_end, 3),
                    "lead_in": round(lead, 3)})
        prev_end = new_end

    return out
