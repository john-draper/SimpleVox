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

from replacements import REPLACEMENTS, clean_word, _resolve_key

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-*]*")
TIME_LINE = "-->"  # SRT cue timestamp separator


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
    """Text subtitle streams of a file (index, codec, language)."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s",
         "-show_entries", "stream=index,codec_name:stream_tags=language",
         "-of", "compact=p=0", video],
        capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        fields = dict(p.split("=", 1) for p in line.split("|") if "=" in p)
        if fields.get("codec_name") in ("subrip", "srt", "mov_text", "ass", "ssa", "webvtt"):
            out.append({"index": int(fields["index"]),
                        "codec": fields["codec_name"],
                        "lang": fields.get("tag:language", "")})
    return out


def extract_srt(video: str, stream_index: int) -> str:
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", video, "-map", f"0:{stream_index}",
         "-f", "srt", "-"], capture_output=True)
    return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else ""


def prepare_subtitles(video: str, workdir: str) -> tuple[str | None, int]:
    """Return (path to rewritten .srt, hits) or (None, 0) when nothing to do."""
    import os

    streams = subtitle_streams(video)
    if len(streams) != 1:
        if streams:
            print(f"[subs] {os.path.basename(video)}: {len(streams)} subtitle "
                  f"tracks; leaving them untouched", flush=True)
        return None, 0
    srt = extract_srt(video, streams[0]["index"])
    if not srt.strip():
        return None, 0
    new_srt, hits = rewrite_srt(srt)
    if not hits:
        return None, 0
    path = os.path.join(workdir, "censored_subtitles.srt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_srt)
    return path, hits
