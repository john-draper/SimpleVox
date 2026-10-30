#!/usr/bin/env python
"""
SimpleVox — automatically detect profanity in a video file and replace each
profane word with a clean euphemism spoken by a single, generic male voice.

The video stream is preserved losslessly; only the audio track is edited.

Pipeline (4 stages, no voice cloning, no diarization, no vocal separation):
    1. Transcribe    (transcribe.py)       — faster-whisper word timestamps
    2. Filter        (replacements.py)     — match profanity -> clean words
    3. Generate      (generate_audio.py)   — single generic male TTS voice
    4. Splice + Mux  (splice_audio.py)     — censored audio + ORIGINAL video

Usage:
    python run.py                              # process input/ recursively
    python run.py "input/movie.mkv"            # single file
    python run.py "input/Season 1"             # whole folder (recursive)
    python run.py "input/movie.mp4" --output-dir output
    python run.py "input/movie.mkv" --voice en-US-DavisNeural

When a directory is given (or no argument), every video file under it is
processed recursively. The input folder structure is mirrored into the
output directory and each output keeps its ORIGINAL filename (no suffix is
appended); intermediate JSON/WAV files live under output/_intermediate/.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
import _nowindow  # noqa: F401  (no console windows for child processes)


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


# Lead-in pad is now per-family (auto) by default — see splice_audio.py's
# get_word_lead_in_ms(). No import needed; we pass None to use auto mode.


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

BANNER = "=" * 78

# Seconds of extra audio censored around a second-ear hit. The second model's
# timestamps are frame-level and can sit a few hundred ms late; widening the
# window costs a little neighbour audio (acceptable) and never moves the
# timeline (the replacement fills exactly the window it is given).
SECOND_EAR_PAD_S = 0.30


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

def stage1_transcribe(video: str, words_json: str, model: str, device: str | None,
                      word_pad_ms: int = 50) -> bool:
    banner("[STAGE 1/4] TRANSCRIPTION")
    log("  faster-whisper word-level transcription (no alignment, no diarization)")
    log(f"  Input:   {video}")
    log(f"  Output:  {words_json}")
    try:
        from transcribe import transcribe_audio
        transcribe_audio(
            audio_path=video,
            language="en",
            model_name=model,
            compute_type="auto",
            device=device,
            output_path=words_json,
            word_pad_ms=word_pad_ms,
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


def stage1b_second_ear(video: str, words_json: str) -> bool:
    """Optional second detector pass (one extra model, independent bias).

    The primary Whisper pass is biased toward clean text: it deletes or
    euphemises profanity. A second, architecturally different model catches a
    different subset of misses. Hits that do not overlap an existing word are
    appended to the transcription JSON, so every later stage sees them.

    Non-fatal by design: if the second ear cannot load, the pipeline continues
    with the primary pass alone.
    """
    banner("[STAGE 1b] SECOND EAR (independent detector)")
    try:
        from ensemble_audit import detect_media, load_second_ear
        ear = load_second_ear()
        hits = detect_media(Path(video), [ear])

        with open(words_json, encoding="utf-8") as f:
            words = json.load(f)
        known = list(words)
        added = []
        for h in hits:
            # Overlap test on the raw hit: if the primary pass already heard
            # this occurrence, there is nothing to add.
            if any(h["start"] < k["end"] - 0.05 and h["end"] > k["start"] + 0.05
                   for k in known):
                continue
            # The second ear's timestamps are coarser than the primary's
            # (frame-level, can be ~300ms late). Widen the window: over-
            # censoring a neighbour is acceptable, a leaked onset is not.
            added.append({"word": h["word"],
                          "start": max(0.0, float(h["start"]) - SECOND_EAR_PAD_S),
                          "end": float(h["end"]) + SECOND_EAR_PAD_S})

        if added:
            words = sorted(words + added, key=lambda w: w["start"])
            with open(words_json, "w", encoding="utf-8") as f:
                json.dump(words, f, ensure_ascii=False, indent=2)
        log(f"\n  [OK] Second ear: {len(hits)} profane hit(s), "
            f"{len(added)} not already in the primary transcript")
    except Exception as exc:
        log(f"\n  [WARN] Second ear unavailable ({type(exc).__name__}: {exc}); "
            f"continuing with the primary pass only.")
    return True


def stage2_filter(words_json: str, replacements_json: str,
                  video: str | None = None,
                  model_name: str = "large-v3",
                  device: str | None = None) -> tuple[bool, int]:
    banner("[STAGE 2/4] PROFANITY FILTERING")
    log(f"  Input:   {words_json}")
    log(f"  Output:  {replacements_json}")
    try:
        from replacements import find_matches
        with open(words_json, encoding="utf-8") as f:
            words = json.load(f)
        matches = find_matches(words)
        if matches and video:
            from transcribe import decode_audio_16k
            audio = decode_audio_16k(video)
            try:
                from transcribe import load_model
                # Full-file discovery may use a faster model (medium); the
                # per-candidate re-listen stays large-v3 for accurate times.
                rt_model = load_model("medium", device or "cuda")
            except Exception as exc:
                rt_model = None
                log(f"  [model] WARNING: could not reload for re-timing "
                    f"({type(exc).__name__}: {exc})")
            # Full-file word times are loose; re-listen to a short clip around
            # each candidate. A clip-sized pass puts the word where it is
            # actually spoken (full pass: "Fuck" 715.59-716.99; clip:
            # 716.30-716.90), which stops the euphemism from playing early.
            if rt_model is not None:
                try:
                    from retime import retime_matches
                    matches = retime_matches(matches, audio, rt_model)
                    log(f"  [retime] Re-listened to {len(matches)} candidate(s)")
                except Exception as exc:
                    log(f"  [retime] WARNING: skipped ({type(exc).__name__}: {exc})")
                # The subtitle track lists words Whisper sometimes drops.
                try:
                    from oracle import extract_cues, oracle_matches
                    cues = extract_cues(video)
                    before = len(matches)
                    matches = oracle_matches(matches, cues, audio, rt_model)
                    if len(matches) > before:
                        log(f"  [oracle] Added {len(matches) - before} "
                            f"window(s) from {len(cues)} subtitle cues")
                    elif cues:
                        log(f"  [oracle] {len(cues)} subtitle cue(s); nothing new")
                except Exception as exc:
                    log(f"  [oracle] WARNING: skipped ({type(exc).__name__}: {exc})")
            # Word timestamps often end a long word early. Grow each censored
            # window to the word's acoustic end before TTS generation, so the
            # euphemism fully covers the original instead of leaving its tail
            # in the following gap ("scoundrel ... fucker").
            try:
                from refine import refine_matches
                before = [(m["start"], m["end"]) for m in matches]
                matches = refine_matches(matches, words, audio)
                grown = sum(1 for (a, b), m in zip(before, matches)
                            if m["end"] - m["start"] > b - a + 0.02)
                if grown:
                    log(f"  [refine] Extended {grown} window(s) to the spoken "
                        f"word end")
            except Exception as exc:
                log(f"  [refine] WARNING: skipped ({type(exc).__name__}: {exc})")
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


def stage4_splice(video: str, replacements_json: str, audio_dir: str, final_output: str,
                  lead_in_ms: int | None = None) -> bool:
    banner("[STAGE 4/4] AUDIO SPLICING / VIDEO MUX")
    log(f"  Original:     {video}")
    log(f"  Replacements: {replacements_json}")
    log(f"  WAV dir:      {audio_dir}")
    log(f"  Output:       {final_output}")
    from splice_audio import lead_in_desc
    lead_desc = (f"{lead_in_ms}ms (all words)" if lead_in_ms is not None
                 else lead_in_desc())
    log(f"  Lead-in pad:  {lead_desc}")
    log(f"  (Only the audio is censored; video stream, if any, is copied losslessly.)")
    try:
        from splice_audio import main as splice_main
        cli_args = [
            video,
            "--replacements-json", replacements_json,
            "--audio-dir", audio_dir,
            "--output", final_output,
        ]
        if lead_in_ms is not None:
            cli_args += ["--lead-in-ms", str(lead_in_ms)]
        rc = splice_main(cli_args)
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
# Discovery + path helpers
# --------------------------------------------------------------------------- #

VIDEO_EXTS = {".mkv", ".mp4", ".m4v", ".mov", ".avi", ".webm", ".wmv", ".flv"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".m4b", ".flac", ".ogg", ".aac"}
MEDIA_EXTS = VIDEO_EXTS | AUDIO_EXTS


def discover_videos(path: Path) -> list[Path]:
    """Return a sorted list of media files (video or audio) at/under `path`.

    - If `path` is a file, return [path] (regardless of extension).
    - If `path` is a directory, return all video/audio files found recursively.
    - Otherwise return [].
    """
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in MEDIA_EXTS
        )
    return []


def relative_to_root(video_path: Path, input_root: Path) -> Path:
    """Path of `video_path` relative to `input_root`.

    Falls back to just the filename if `video_path` is not under `input_root`
    (e.g. when the user passes a file from outside the input directory).
    """
    try:
        return video_path.resolve().relative_to(input_root.resolve())
    except ValueError:
        return Path(video_path.name)


# --------------------------------------------------------------------------- #
# Pre-flight checks
# --------------------------------------------------------------------------- #

def preflight_global() -> bool:
    """One-time environment checks (Python, ffmpeg)."""
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
    return True


# --------------------------------------------------------------------------- #
# Per-file pipeline
# --------------------------------------------------------------------------- #

def process_file(
    video_path: Path,
    input_root: Path,
    output_dir: Path,
    voice: str,
    model: str,
    device: str | None,
    skip_existing: bool,
    lead_in_ms: int | None = None,
    word_pad_ms: int = 50,
    second_ear: bool = False,
    words_json_src: str | None = None,
) -> bool:
    """Run the full 4-stage pipeline on a single video file.

    Folder structure under `input_root` is mirrored into `output_dir`, and the
    output video keeps its ORIGINAL filename (no `_censored` suffix). All
    intermediate JSON/WAV files are isolated under
    `output_dir/_intermediate/<mirrored subpath>/`.
    """
    rel = relative_to_root(video_path, input_root)
    # Output video mirrors the input subpath and keeps the same filename.
    final_output = output_dir / rel
    # Intermediates live under a dedicated, clearly-named subfolder.
    intermediate_dir = output_dir / "_intermediate" / rel.parent

    words_json = str(intermediate_dir / f"{video_path.stem}.json")
    replacements_json = str(intermediate_dir / f"{video_path.stem}_replacements.json")
    audio_dir = str(intermediate_dir / "generated_audio")

    final_output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and final_output.is_file():
        log(f"  [skip-existing] {final_output} already exists; skipping.")
        return True

    if not video_path.is_file():
        log(f"  [FAILED] Input file not found: {video_path}")
        return False

    # --- Config summary ---
    banner(f"PROCESSING: {rel}")
    log(f"  Input file:    {video_path}")
    log(f"  Output folder: {final_output.parent}")
    log(f"  Voice:         {voice}")
    log(f"  Whisper model: {model}")
    log(f"  Output file:   {final_output}")
    log()
    log(f"  Stage 1 -> {words_json}")
    log(f"  Stage 2 -> {replacements_json}")
    log(f"  Stage 3 -> {audio_dir}/*.wav")
    log(f"  Stage 4 -> {final_output}")

    # --- Stage 1: Transcription ---
    if words_json_src and Path(words_json_src).is_file():
        # Fast path for re-censoring: reuse an existing (VAD-era) transcript as
        # the candidate list. retime.py re-listens to every candidate and
        # oracle.py adds the words it dropped, so the old timings are only a
        # locator, not the splice point. Saves the full-file ASR pass.
        shutil.copy2(words_json_src, words_json)
        log(f"  [words] Reused existing transcript: {words_json_src}")
    else:
        if not stage1_transcribe(str(video_path), words_json, model, device,
                                 word_pad_ms=word_pad_ms):
            return False

        # --- Stage 1b: optional second ear ---
        if second_ear:
            stage1b_second_ear(str(video_path), words_json)

    # --- Stage 2: Profanity filtering ---
    ok, count = stage2_filter(words_json, replacements_json,
                              video=str(video_path), model_name=model,
                              device=device)
    if not ok:
        return False

    if count == 0:
        banner("[INFO] NO PROFANITY DETECTED")
        log("  No replacements needed. Copying original to output.")
        shutil.copy2(video_path, final_output)
        log(f"  [OK] Final output: {final_output}")
        return True

    # --- Stage 3: Audio generation (single generic voice) ---
    if not stage3_generate(replacements_json, audio_dir, voice):
        return False

    # --- Stage 4: Splicing (into the ORIGINAL video) ---
    if not stage4_splice(str(video_path), replacements_json, audio_dir, str(final_output),
                         lead_in_ms=lead_in_ms):
        return False

    # --- Success ---
    log(f"\n  [OK] Final output: {final_output}")
    log("  Intermediate files:")
    log(f"    Transcription JSON:  {words_json}")
    log(f"    Replacements JSON:   {replacements_json}")
    log(f"    Generated .wav dir:  {audio_dir}")
    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="SimpleVox: censor profanity in videos/audio with a single generic voice. "
                    "Accepts a file, a directory (processed recursively), or nothing "
                    "(defaults to the input/ folder, processed recursively).",
    )
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to a single video/audio file OR a directory to process recursively. "
             "If omitted, the --input-dir folder (default: input) is used.",
    )
    p.add_argument(
        "--input-dir",
        default="input",
        help="Root input folder (default: input). Used to mirror the folder "
             "structure into the output directory.",
    )
    p.add_argument(
        "--output-dir",
        default="output",
        help="Directory for all outputs (default: output).",
    )
    p.add_argument(
        "--voice",
        default="en-US-GuyNeural",
        help="edge-tts voice to use for all replacements (default: en-US-GuyNeural). "
             "Alternatives: en-US-DavisNeural, en-GB-RyanNeural.",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("WHISPER_MODEL", "large-v3"),
        help="faster-whisper model (default: large-v3). 'small' is much faster "
             "but misses profanity.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device for faster-whisper (cuda/cpu). Default: auto-detect.",
    )
    p.add_argument(
        "--words-json",
        default=None,
        help="Reuse an existing transcript JSON as the candidate list instead "
             "of running the full-file ASR pass (retime/oracle still run).",
    )
    p.add_argument(
        "--word-pad-ms",
        type=int,
        default=50,
        help="Widen each censored word span by this much (default: 50ms) to "
             "cover native word-timestamp imprecision. 0 disables.",
    )
    p.add_argument(
        "--second-ear",
        action="store_true",
        help="Run a second, independent detector (NVIDIA Parakeet) after the "
             "primary pass and add any profanity it hears that Whisper missed. "
             "Costs a few seconds per file; catches a different subset of misses.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos whose output file already exists.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining videos if one fails (batch mode).",
    )
    p.add_argument(
        "--lead-in-ms",
        type=int,
        default=None,
        help=("Silence (ms) inserted before each replacement to mute the leaked "
              "onset of the original word. Default: per-family auto "
              "(see splice_audio.WORD_FAMILY_LEAD_IN_MS). Pass 0 to disable "
              "all lead-in, or a specific value to override all words."),
    )
    args = p.parse_args(argv)

    if not preflight_global():
        banner("[FAILED] PRE-FLIGHT CHECKS FAILED")
        return 1

    input_root = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Decide what to process: explicit arg, else the whole input folder.
    target = Path(args.input).resolve() if args.input else input_root
    if args.input and not target.exists():
        log(f"[FAILED] Input path does not exist: {target}")
        return 1
    if not args.input and not input_root.exists():
        log(f"[FAILED] Input folder does not exist: {input_root}")
        return 1

    videos = discover_videos(target)
    if not videos:
        log(f"[FAILED] No video files found at/under: {target}")
        return 1

    banner("PIPELINE BATCH")
    log(f"  Input root:  {input_root}")
    log(f"  Output dir:  {output_dir}")
    log(f"  Videos:      {len(videos)}")
    for v in videos:
        log(f"    - {relative_to_root(v, input_root)}")
    log(f"  Voice:       {args.voice}")
    log(f"  Whisper:     {args.model}")
    log(f"  Second ear:  {'on (Parakeet)' if args.second_ear else 'off'}")
    log(f"  Word pad:    {args.word_pad_ms}ms")
    log(f"  Skip existing: {'yes' if args.skip_existing else 'no'}")
    log(f"  Continue on error: {'yes' if args.continue_on_error else 'no'}")
    lead_desc = (f"{args.lead_in_ms}ms (all words)" if args.lead_in_ms is not None
                 else "per-family auto")
    log(f"  Lead-in pad:   {lead_desc}")

    failures = 0
    for idx, video_path in enumerate(videos, start=1):
        banner(f"[{idx}/{len(videos)}] {relative_to_root(video_path, input_root)}")
        try:
            ok = process_file(
                video_path=video_path,
                input_root=input_root,
                output_dir=output_dir,
                voice=args.voice,
                model=args.model,
                device=args.device,
                skip_existing=args.skip_existing,
                lead_in_ms=args.lead_in_ms,
                word_pad_ms=args.word_pad_ms,
                second_ear=args.second_ear,
                words_json_src=args.words_json,
            )
        except Exception as exc:
            log(f"[FAILED] Unexpected error on {video_path}: "
                f"{type(exc).__name__}: {exc}")
            ok = False

        if not ok:
            failures += 1
            if not args.continue_on_error and len(videos) > 1:
                log("[STOP] A video failed and --continue-on-error was not set; "
                    "aborting remaining files.")
                break

    banner("BATCH SUMMARY")
    log(f"  Processed: {len(videos) - failures} succeeded, {failures} failed "
        f"(of {len(videos)} total)")
    log(f"  Outputs written under: {output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
