#!/usr/bin/env python
"""Rewrite profanity in embedded subtitles with the same dictionary the audio uses.

The video's audio is already censored ("fucking" -> "stinking"), but ffmpeg's
mux step copies the original subtitle track unchanged, so a viewer with
subtitles on still reads every censored word. This module closes that gap: it
extracts the text subtitle track, replaces each dictionary match with the same
euphemism, and returns an .srt for the mux step to embed.

Design notes:
  * Matching reuses replacements.clean_word/_resolve_key so subtitle and audio
    censoring can never disagree (hyphen compounds, possessives, whisper's
    "f***ing" spellings included).
  * Only the token is replaced; surrounding punctuation, tags and line breaks
    stay byte-identical, so cue timing and formatting are preserved.
  * Capitalisation is mirrored ("God" -> "Gosh", "FUCK" -> "FREAK").
  * If a file has more than one subtitle track it is left alone (copy) rather
    than risk dropping a forced/foreign track.
"""

import re
import string
import subprocess
import _nowindow  # noqa: F401  (no console windows for child processes)

from replacements import REPLACEMENTS, clean_word, _resolve_key

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-*]*")
TIME_LINE = "-->"  # SRT cue timestamp separator
TEXT_CODECS = {"subrip", "srt", "mov_text", "ass", "ssa", "webvtt"}


def resolve(tok: str) -> str | None:
    """Replacement text for a subtitle token, or None if it is clean."""
    key = clean_word(tok)
    rep = _resolve_key(key)
    if rep is None and key.endswith("'s"):
        stem = _resolve_key(key[:-2])
        if stem is not None:
            rep = stem + "'s"
    if rep is None and "*" in tok:
        star = tok.strip().strip(
            "".join(c for c in string.punctuation if c != "*")).lower()
        rep = REPLACEMENTS.get(star)
    return rep


def _match_case(orig: str, rep: str) -> str:
    letters = [c for c in orig if c.isalpha()]
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return rep.upper()
    if letters and letters[0].isupper():
        return rep[:1].upper() + rep[1:]
    return rep


def rewrite_text(text: str) -> tuple[str, int]:
    """Replace dictionary words in one line of subtitle text."""
    hits = 0

    def _sub(m: re.Match) -> str:
        nonlocal hits
        tok = m.group(0)
        rep = resolve(tok)
        if rep is None:
            return tok
        hits += 1
        return _match_case(tok, rep)

    return WORD_RE.sub(_sub, text), hits


def rewrite_srt(srt: str) -> tuple[str, int]:
    """Rewrite all text lines of an SRT document; timestamps and indices intact."""
    out, hits = [], 0
    for line in srt.splitlines():
        stripped = line.strip()
        if not stripped or TIME_LINE in line or stripped.isdigit():
            out.append(line)
            continue
        new, h = rewrite_text(line)
        hits += h
        out.append(new)
    return "\n".join(out) + "\n", hits


def subtitle_streams(video: str) -> list[dict]:
    """All subtitle streams in order: absolute index, subtitle-relative index,
    codec, language, and whether the codec is text."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s",
         "-show_entries", "stream=index,codec_name:stream_tags=language",
         "-of", "compact=p=0", video],
        capture_output=True, text=True)
    out = []
    for rel, line in enumerate(r.stdout.splitlines()):
        fields = dict(p.split("=", 1) for p in line.split("|") if "=" in p)
        codec = fields.get("codec_name", "")
        out.append({"index": int(fields["index"]), "rel": rel, "codec": codec,
                    "lang": fields.get("tag:language", ""),
                    "text": codec in TEXT_CODECS})
    return out


def extract_srt(video: str, stream_index: int) -> str:
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", video, "-map", f"0:{stream_index}",
         "-f", "srt", "-"], capture_output=True)
    return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else ""


def prepare_subtitles(video: str, workdir: str) -> tuple[list[dict], int, int]:
    """Rewrite every English (or und) text subtitle track.

    Returns (tracks, hits, total_subtitle_streams) where each track is
    {"path": srt, "rel": subtitle-relative index, "lang": language}. Files with
    bitmap subtitles are left untouched (they cannot round-trip through SRT).
    """
    import os

    streams = subtitle_streams(video)
    if not streams:
        return [], 0, 0
    if not all(s["text"] for s in streams):
        print(f"[subs] {os.path.basename(video)}: bitmap subtitle track; "
              f"leaving subtitles untouched", flush=True)
        return [], 0, len(streams)
    targets = [s for s in streams if s["lang"] in ("eng", "en", "und")]
    if not targets and len(streams) == 1:
        targets = streams
    tracks, hits = [], 0
    for s in targets:
        srt = extract_srt(video, s["index"])
        if not srt.strip():
            continue
        new_srt, h = rewrite_srt(srt)
        if not h:
            continue
        path = os.path.join(workdir, f"censored_subtitles_{s['rel']}.srt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_srt)
        tracks.append({"path": path, "rel": s["rel"], "lang": s["lang"] or "eng"})
        hits += h
    return tracks, hits, len(streams)
