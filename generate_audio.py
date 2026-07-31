#!/usr/bin/env python
"""
Generate replacement audio for each profane word using a SINGLE, generic,
high-quality male TTS voice for ALL replacements.

SimpleVox intentionally does NOT do voice cloning or per-speaker voices. Every
replacement word is spoken by the same voice (Microsoft Edge neural TTS,
en-US-GuyNeural by default), which keeps the pipeline simple, free, and
dependency-light.

The output WAV files are named so the splice stage can match them back to the
original timestamps:
    <start_ms padded to 8 digits>_<end_ms padded to 8 digits>_<replacement>.wav

Example:
    python generate_audio.py output/movie_replacements.json --output-dir output/generated_audio
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# The single generic male voice used for every replacement word.
# Microsoft Edge neural TTS voice "Guy" (en-US) — a neutral, quality male voice.
DEFAULT_VOICE = "en-US-GuyNeural"


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Input loading
# --------------------------------------------------------------------------- #

def load_replacements(path: str) -> list[dict]:
    in_path = Path(path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list of replacement objects, got {type(data).__name__}."
        )
    return data


def format_filename(start: float, end: float, replacement: str) -> str:
    """Format the output .wav filename based on start/end times and replacement text."""
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    safe_replacement = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in replacement
    ).strip("_")
    return f"{start_ms:08d}_{end_ms:08d}_{safe_replacement}.wav"


# --------------------------------------------------------------------------- #
# edge-tts (Microsoft Edge neural TTS — free, no API key, reliable)
# --------------------------------------------------------------------------- #

_EDGE_TTS = None  # lazily-loaded edge_tts module


def _get_edge_tts():
    global _EDGE_TTS
    if _EDGE_TTS is None:
        import edge_tts
        _EDGE_TTS = edge_tts
    return _EDGE_TTS


async def _synthesize_to_mp3(text: str, voice: str, mp3_path: str) -> None:
    """Synthesize text to an MP3 file via edge-tts."""
    edge_tts = _get_edge_tts()
    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(mp3_path)
    size = os.path.getsize(mp3_path)
    if size < 200:
        raise RuntimeError(f"edge-tts produced only {size} bytes for {text!r}")


def generate_wav(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """Generate WAV bytes (44.1kHz mono PCM) for the given text using edge-tts.

    edge-tts returns MP3, so we transcode to WAV via ffmpeg to match the format
    expected by the splice stage.
    """
    mp3_path = None
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_tmp:
            mp3_path = mp3_tmp.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
            wav_path = wav_tmp.name

        asyncio.run(_synthesize_to_mp3(text, voice, mp3_path))

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path,
             "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", wav_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not os.path.exists(wav_path):
            tail = result.stderr[-300:] if result.stderr else "(no stderr)"
            raise RuntimeError(f"ffmpeg mp3->wav failed: {tail}")

        with open(wav_path, "rb") as f:
            return f.read()
    finally:
        for p in (mp3_path, wav_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


# --------------------------------------------------------------------------- #
# Per-word generation
# --------------------------------------------------------------------------- #

def save_wav(audio_bytes: bytes, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(audio_bytes)


def process_replacement(
    entry: dict,
    output_dir: Path,
    voice: str,
) -> str | None:
    """Generate audio for a single replacement entry using the generic male voice."""
    replacement = entry.get("replacement", "")
    start = entry.get("start", 0.0)
    end = entry.get("end", 0.0)

    if not replacement:
        log(f"  [skip] Empty replacement text for entry: {entry}")
        return None

    filename = format_filename(start, end, replacement)
    output_path = output_dir / filename

    try:
        log(f"  [generate] '{replacement}' -> {filename}")
        audio_bytes = generate_wav(text=replacement, voice=voice)
        save_wav(audio_bytes, output_path)
        log(f"    [saved] {output_path} ({len(audio_bytes) / 1024:.1f} KB)")
        return str(output_path)
    except Exception as exc:
        exc_name = type(exc).__name__
        log(f"    [error] {exc_name}: {exc}")
        return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate replacement audio with a single generic male "
                    "voice (Microsoft Edge neural TTS).",
    )
    p.add_argument(
        "input",
        help="Path to the filtered replacements JSON from find_replacements.py",
    )
    p.add_argument(
        "-o", "--output-dir",
        default="output/generated_audio",
        help="Directory to save generated .wav files (default: output/generated_audio).",
    )
    p.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"edge-tts voice to use (default: {DEFAULT_VOICE}).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip generating audio for files that already exist.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Process the JSON without calling the API.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        replacements = load_replacements(args.input)
    except FileNotFoundError as exc:
        log(f"[error] {exc}")
        return 2
    except (ValueError, json.JSONDecodeError) as exc:
        log(f"[error] {type(exc).__name__}: {exc}")
        return 1

    log(f"[input] Loaded {len(replacements)} replacement(s) from {args.input}")
    log(f"[voice] Using single generic male voice: {args.voice}")
    log(f"[output] Saving .wav files to {Path(args.output_dir)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        log("[dry-run] Skipping synthesis. Showing planned filenames only.")
        for entry in replacements:
            replacement = entry.get("replacement", "")
            start = entry.get("start", 0.0)
            end = entry.get("end", 0.0)
            filename = format_filename(start, end, replacement)
            log(f"  [dry-run] '{replacement}' -> {output_dir / filename}")
        return 0

    # Warm up the edge-tts module (lazy import) before the loop.
    _get_edge_tts()

    success_count = 0
    failed_count = 0
    skipped_count = 0

    for i, entry in enumerate(replacements):
        replacement_text = entry.get("replacement", "")
        start_t = entry.get("start", 0.0)
        end_t = entry.get("end", 0.0)

        if args.skip_existing and replacement_text:
            expected_filename = format_filename(start_t, end_t, replacement_text)
            expected_path = output_dir / expected_filename
            if expected_path.is_file():
                log(f"[{i + 1}/{len(replacements)}] [skip-existing] {expected_filename}")
                skipped_count += 1
                success_count += 1
                continue

        log(f"[{i + 1}/{len(replacements)}] Processing replacement...")
        result = process_replacement(
            entry=entry,
            output_dir=output_dir,
            voice=args.voice,
        )
        if result is not None:
            success_count += 1
        else:
            failed_count += 1

    log(f"[done] Generated {success_count} audio file(s) in {output_dir}.")
    if skipped_count:
        log(f"[done] {skipped_count} file(s) skipped (already existed).")
    if failed_count:
        log(f"[done] {failed_count} replacement(s) failed. See errors above.")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())