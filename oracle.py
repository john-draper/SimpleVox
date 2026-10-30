#!/usr/bin/env python
"""Subtitle oracle: recover profanity that Whisper's transcript dropped.

Whisper occasionally omits a whole profane word (measured: "We want A's,
bitches!" -> transcript has no "bitches"; "the eating of ass" -> "a va -").
The episode's embedded English subtitle track is human-authored and still
lists the word, so it is used as a second, independent detector:

  1. Parse the first text subtitle track.
  2. Keep cues whose text contains a dictionary word.
  3. Re-listen to each such cue with a short VAD-off ASR pass (accurate local
     timings) and add any profane word that is not already in a censored
     window.
  4. If the local pass still cannot hear it, place a window from the word's
     proportional position in the cue (last-resort recall).

The output is merged with the ASR candidates and then acoustically refined.

Usage:
    from oracle import oracle_matches, extract_cues
    cues = extract_cues(video)
    matches = oracle_matches(matches, cues, audio, model)
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from refine import expected_ms
from replacements import REPLACEMENTS, clean_word, _resolve_key
from transcribe import VERBATIM_PROMPT

SR = 16000
TAG_RE = re.compile(r"<[^>]+>|\{[^}]*\}")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-*]*")
TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*"
                     r"(\d+):(\d+):(\d+)[,.](\d+)")


def _t(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000.0


def extract_cues(video: str) -> list[dict]:
    """Extract the first text subtitle track as [{start,end,text}]."""
    with tempfile.TemporaryDirectory(prefix="oracle_") as tmp:
        srt = Path(tmp) / "subs.srt"
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", video,
               "-map", "0:s:0", "-f", "srt", str(srt)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not srt.is_file():
            return []
        raw = srt.read_text(encoding="utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n")
    cues: list[dict] = []
    for block in re.split(r"\n\n+", raw.strip()):
        lines = block.split("\n")
        if len(lines) < 2:
            continue
        m = TIME_RE.search(lines[1] if TIME_RE.search(lines[1]) else lines[0])
        if not m:
            continue
        start = _t(*m.groups()[:4])
        end = _t(*m.groups()[4:])
        text = " ".join(lines[2:])
        text = TAG_RE.sub(" ", text).strip()
        if text and end > start:
            cues.append({"start": start, "end": end, "text": text})
    return cues


def _profane_tokens(text: str) -> list[tuple[int, str, str]]:
    """Return [(char_pos, original_token, replacement)] for dictionary hits."""
    out = []
    for m in TOKEN_RE.finditer(text):
        tok = m.group(0)
        key = clean_word(tok)
        rep = _resolve_key(key)
        if rep is None and key.endswith("'s"):
            rep = _resolve_key(key[:-2])
        if rep is not None:
            out.append((m.start(), tok, rep))
    return out


def batch_local_words(audio: np.ndarray, model, spans: list[tuple[float, float]],
                      sr: int = SR, pad: float = 0.35,
                      gap: float = 0.25) -> list[list[dict]]:
    """Transcribe many short cue windows in ONE model call.

    Per-cue calls spend most of their time in model/decoder setup; a typical
    episode has 40-100 profane cues, which made the audit take minutes. The
    clips are concatenated with short silences, transcribed once, and the
    words are mapped back to their source windows.
    """
    if not spans:
        return []
    total = len(audio) / sr
    parts: list[np.ndarray] = []
    meta: list[tuple[int, int, float]] = []
    cursor = 0
    gap_n = int(gap * sr)
    for st, en in spans:
        t0 = max(0.0, st - pad)
        t1 = min(total, en + pad)
        a, b = int(t0 * sr), int(t1 * sr)
        clip = audio[a:b]
        parts.append(clip)
        parts.append(np.zeros(gap_n, dtype=np.float32))
        meta.append((cursor, cursor + len(clip), t0))
        cursor += len(clip) + gap_n
    concat = np.concatenate(parts)
    all_words: list[tuple[float, float, str]] = []
    try:
        segs, _ = model.transcribe(
            concat, language="en", vad_filter=False, beam_size=5,
            word_timestamps=True, condition_on_previous_text=False,
            initial_prompt=VERBATIM_PROMPT,
        )
        for s in segs:
            for w in (s.words or []):
                tok = (w.word or "").strip()
                if tok:
                    all_words.append((float(w.start), float(w.end), tok))
    except Exception:
        pass
    out: list[list[dict]] = []
    for c0, c1, t0 in meta:
        words = []
        for a, b, tok in all_words:
            mid = (a + b) / 2.0
            if c0 <= mid < c1:
                words.append({"word": tok,
                              "start": t0 + (a - c0) / sr,
                              "end": t0 + (b - c0) / sr})
        out.append(words)
    return out


def _add_or_skip(matches: list[dict], cand: dict, near_s: float = 0.45) -> None:
    """Append `cand` unless a match already covers the same word occurrence."""
    mid = (cand["start"] + cand["end"]) / 2.0
    for m in matches:
        m_mid = (m["start"] + m["end"]) / 2.0
        if abs(m_mid - mid) < near_s:
            return
        if cand["start"] < m["end"] and cand["end"] > m["start"]:
            return
    matches.append(cand)


def oracle_matches(matches: list[dict], cues: list[dict], audio: np.ndarray,
                   model, sr: int = SR) -> list[dict]:
    """Merge subtitle-detected profanity into `matches` (in place copy)."""
    if not cues:
        return matches

    audio_dur = len(audio) / sr
    profane_cues = []
    for cue in cues:
        hits = _profane_tokens(cue["text"])
        if hits:
            profane_cues.append((cue, hits))
    if not profane_cues:
        return matches

    out = list(matches)
    added = 0
    cue_spans = [(c["start"], c["end"]) for c, _ in profane_cues]
    local_sets = batch_local_words(audio, model, cue_spans, sr=sr)
    for (cue, hits), local in zip(profane_cues, local_sets):
        used: set[int] = set()
        for pos, tok, rep in hits:
            # Match a local ASR word with the same dictionary key/replacement.
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
                dmid = abs((lw["start"] + lw["end"]) / 2.0
                           - (cue["start"] + cue["end"]) / 2.0)
                if dmid < best_d:
                    best_i, best_d = i, dmid
            if best_i >= 0:
                lw = local[best_i]
                used.add(best_i)
                start = float(lw["start"])
                end = float(lw["end"])
                if end <= start:
                    end = min(audio_dur, start + 0.30)
                # Clip passes can over/under-shoot; keep a plausible span.
                cap = expected_ms(tok) / 1000.0 + 0.30
                end = min(end, start + cap)
                _add_or_skip(out, {"word": tok, "start": round(start, 3),
                                   "end": round(max(end, start + 0.08), 3),
                                   "replacement": rep})
                added += 1
            else:
                # Last resort: proportional position in the cue text.
                span = max(0.2, cue["end"] - cue["start"])
                ratio = pos / max(1, len(cue["text"]))
                start = cue["start"] + ratio * span
                end = min(cue["end"], start + 0.45)
                _add_or_skip(out, {"word": tok, "start": round(start, 3),
                                   "end": round(max(end, start + 0.20), 3),
                                   "replacement": rep})
                added += 1

    out.sort(key=lambda m: float(m["start"]))
    return out
