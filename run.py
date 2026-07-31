#!/usr/bin/env python
"""
SimpleVox — automatically detect profanity in a video file and replace each
profane word with a clean euphemism spoken by a single, generic male voice.

The video stream is preserved losslessly; only the audio track is edited.

Pipeline (4 stages, no voice cloning, no diarization, no vocal separation):
    1. Transcribe    (transcribe.py)       — WhisperX word-level timestamps
    2. Filter        (replacements.py)     — match profanity -> clean words
    3. Generate      (generate_audio.py)   — single generic male TTS voice
    4. Splice + Mux  (splice_audio.py)     — censored audio + ORIGINAL video

Usage:
    python run.py "input/movie.mkv"
    python run.py "input/movie.mp4" --output-dir output
    python run.py "input/movie.mkv" --voice en-US-DavisNeural
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# .env loader (no external dependency)
# --------------------------------------------------------------------------- #

def load_dotenv(path: str = ".env") -> None:
    """Load environment variables from a .env file if it exists.

    Does NOT override variables already set in the OS environment.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = val


load_dotenv()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

BANNER = "=" * 78


def log(msg: str = "") -> None:
    print(msg, flush=True)


def banner(title: str) -> None:
    log()
    log(BANNER)
    log(f"  {title}")
    log(BANNER)


def check_tool(name: str) -> bool:
    """Return True if a CLI tool is on PATH."""
    return shutil.which(name) is not None


# --------------------------------------------------------------------------- #
# Pipeline stages (call into modules directly, not subprocesses)
# --------------------------------------------------------------------------- #

def stage1_transcribe(video: str, words_json: str, model: str, device: str | None) -> bool:
    banner("[STAGE 1/4] TRANSCRIPTION")
    log("  WhisperX word-level transcription (no speaker diarization)")
    log(f"  Input:   {video}")
    log(f"  Output:  {words_json}")
    try:
        from transcribe import transcribe_audio
        transcribe_audio(
            audio_path=video,
            language="en",
            model_name=model,
            batch_size=8,
            compute_type="auto",
            device=device,
            output_path=words_json,
        )
    except SystemExit as exc:
        log(str(exc))
        return False
    except Exception as exc:
        log(f"[FAILED] Stage 1 error: {type(exc).__name__}: {exc}")
        return False

    if not Path(words_json).is_file():
        log(f"[FAILED] Output JSON not found: {words_json}")
        return False
    log(f"\n  [OK] Stage 1 complete -> {words_json}")
    return True


def stage2_filter(words_json: str, replacements_json: str) -> tuple[bool, int]:
    banner("[STAGE 2/4] PROFANITY FILTERING")
    log(f"  Input:   {words_json}")
    log(f"  Output:  {replacements_json}")
    try:
        from replacements import find_matches
        with open(words_json, encoding="utf-8") as f:
            words = json.load(f)
        matches = find_matches(words)
    except Exception as exc:
        log(f"[FAILED] Stage 2 error: {type(exc).__name__}: {exc}")
        return False, 0

    Path(replacements_json).parent.mkdir(parents=True, exist_ok=True)
    with open(replacements_json, "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)

    log(f"\n  [OK] Stage 2 complete -> {replacements_json} ({len(matches)} replacements)")
    for m in matches[:5]:
        log(f"    {m['start']:7.2f}-{m['end']:7.2f}  "
            f"{m['word']!r} -> {m['replacement']!r}")
    if len(matches) > 5:
        log(f"    ... and {len(matches) - 5} more.")
    return True, len(matches)


def stage3_generate(replacements_json: str, audio_dir: str, voice: str) -> bool:
    banner("[STAGE 3/4] AUDIO GENERATION")
    log(f"  Input:     {replacements_json}")
    log(f"  Output:    {audio_dir}")
    log(f"  Voice:     {voice} (single generic voice for ALL words)")
    try:
        from generate_audio import main as generate_main
        rc = generate_main([
            replacements_json,
            "--output-dir", audio_dir,
            "--voice", voice,
            "--skip-existing",
        ])
    except SystemExit as exc:
        rc = int(exc.code) if exc.code is not None else 1
    except Exception as exc:
        log(f"[FAILED] Stage 3 error: {type(exc).__name__}: {exc}")
        return False
    if rc != 0:
        log(f"\n  [FAILED] Stage 3 exited with code {rc}")
        return False
    log(f"\n  [OK] Stage 3 complete -> {audio_dir}")
    return True


def stage4_splice(video: str, replacements_json: str, audio_dir: str, final_output: str) -> bool:
    banner("[STAGE 4/4] AUDIO SPLICING / VIDEO MUX")
    log(f"  Original:     {video}")
    log(f"  Replacements: {replacements_json}")
    log(f"  WAV dir:      {audio_dir}")
    log(f"  Output:       {final_output}")
    log(f"  (Video stream is copied losslessly; only audio is censored.)")
    try:
        from splice_audio import main as splice_main
        rc = splice_main([
            video,
            "--replacements-json", replacements_json,
            "--audio-dir", audio_dir,
            "--output", final_output,
        ])
    except SystemExit as exc:
        rc = int(exc.code) if exc.code is not None else 1
    except Exception as exc:
        log(f"[FAILED] Stage 4 error: {type(exc).__name__}: {exc}")
        return False
    if rc != 0:
        log(f"\n  [FAILED] Stage 4 exited with code {rc}")
        return False
    if not Path(final_output).is_file():
        log(f"\n  [FAILED] Output file not found: {final_output}")
        return False
    log(f"\n  [OK] Stage 4 complete -> {final_output}")
    return True


# --------------------------------------------------------------------------- #
# Pre-flight checks
# --------------------------------------------------------------------------- #

def preflight(video: str) -> bool:
    banner("PRE-FLIGHT CHECKS")

    log("  [check] Python...")
    if not check_tool("python"):
        log("          [FAILED] Python not found on PATH.")
        return False
    log("          [OK]")

    log("  [check] ffmpeg...")
    if not check_tool("ffmpeg"):
        log("          [FAILED] ffmpeg not found on PATH.")
        log("                 Install: winget install Gyan.FFmpeg (Windows)")
        log("                          brew install ffmpeg (macOS)")
        log("                          sudo apt install ffmpeg (Linux)")
        return False
    log("          [OK]")

    log("  [check] Input file...")
    if not Path(video).is_file():
        log(f"          [FAILED] Input file not found: {video}")
        return False
    log(f"          [OK] {video}")

    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="SimpleVox: censor profanity in a video with a single generic voice.",
    )
    p.add_argument(
        "video",
        help="Path to the input video file (e.g. input/movie.mkv)",
    )
    p.add_argument(
        "--output-dir",
        default="output",
        help="Directory for all outputs (default: output)",
    )
    p.add_argument(
        "--voice",
        default="en-US-GuyNeural",
        help="edge-tts voice to use for all replacements (default: en-US-GuyNeural). "
             "Alternatives: en-US-DavisNeural, en-GB-RyanNeural.",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("WHISPER_MODEL", "small"),
        help="Whisper model size (default: small). e.g. medium, large-v3.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device for WhisperX (cuda/cpu). Default: auto-detect.",
    )
    args = p.parse_args(argv)

    video = args.video
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    basename = Path(video).stem

    # Keep the same video container extension for the output.
    input_ext = Path(video).suffix.lower()
    video_exts = {".mkv", ".mp4", ".m4v", ".mov", ".avi", ".webm", ".wmv", ".flv"}
    out_ext = input_ext if input_ext in video_exts else ".mkv"

    words_json = str(output_dir / f"{basename}.json")
    replacements_json = str(output_dir / f"{basename}_replacements.json")
    audio_dir = str(output_dir / "generated_audio")
    final_output = str(output_dir / f"{basename}_censored{out_ext}")

    if not preflight(video):
        banner("[FAILED] PRE-FLIGHT CHECKS FAILED")
        return 1

    # --- Config summary ---
    banner("PIPELINE CONFIGURATION")
    log(f"  Input file:    {video}")
    log(f"  Base name:     {basename}")
    log(f"  Output folder: {output_dir}")
    log(f"  Voice:         {args.voice}")
    log(f"  Whisper model: {args.model}")
    log(f"  Output file:   {final_output}")
    log()
    log(f"  Stage 1 -> {words_json}")
    log(f"  Stage 2 -> {replacements_json}")
    log(f"  Stage 3 -> {audio_dir}/*.wav")
    log(f"  Stage 4 -> {final_output}")

    # --- Stage 1: Transcription ---
    if not stage1_transcribe(video, words_json, args.model, args.device):
        return 1

    # --- Stage 2: Profanity filtering ---
    ok, count = stage2_filter(words_json, replacements_json)
    if not ok:
        return 1

    if count == 0:
        banner("[INFO] NO PROFANITY DETECTED")
        log("  No replacements needed. Copying original to output.")
        shutil.copy2(video, final_output)
        banner("[SUCCESS] PIPELINE COMPLETE")
        log(f"  Final output: {final_output}")
        return 0

    # --- Stage 3: Audio generation (single generic voice) ---
    if not stage3_generate(replacements_json, audio_dir, args.voice):
        return 1

    # --- Stage 4: Splicing (into the ORIGINAL video) ---
    if not stage4_splice(video, replacements_json, audio_dir, final_output):
        return 1

    # --- Success ---
    banner("[SUCCESS] PIPELINE COMPLETE")
    log(f"  Final censored video: {final_output}")
    log()
    log("  Intermediate files:")
    log(f"    Transcription JSON:  {words_json}")
    log(f"    Replacements JSON:   {replacements_json}")
    log(f"    Generated .wav dir:  {audio_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())