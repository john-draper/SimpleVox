#!/usr/bin/env python
"""
Batch profanity censor for a folder of audio clips (e.g. dialogue extracted
from a game's audio archives).

This is the RDR2-scale companion to run.py. run.py reloads the WhisperX model
on EVERY file, which is fine for a handful of videos but prohibitive for the
thousands of short clips that come out of a game's audio archives (each model
load is ~10-20s). batch_censor.py loads the ASR model + alignment model ONCE
and reuses them for every clip, giving roughly a 10-100x speedup at scale.

Pipeline per clip (reuses the proven SimpleVox stages):
    1. Transcribe  (WhisperX, model loaded ONCE for the whole batch)
    2. Filter      (replacements.find_matches -> profane words)
    3. Skip-clean  (no profanity? -> skip, do NOT copy; output holds only
                    the clips that actually changed, ready for selective
                    re-import). Use --copy-clean to mirror every clip instead.
    4. Generate    (edge-tts single generic voice, one wav per replacement)
    5. Splice      (build_censored_audio + encode_audio_only -> censored wav)

Resume: --skip-existing skips clips whose output already exists.
Audit:  writes <output>/_censor_manifest.jsonl with one record per clip
        (censored / clean / error / skipped) so you can review exactly what
        changed BEFORE re-importing anything into the game.

Usage:
    python batch_censor.py "E:\\rdr2_clips\\PEDS_00" "E:\\rdr2_censored\\PEDS_00"
    python batch_censor.py clips out --skip-existing --continue-on-error
    python batch_censor.py clips out --model medium --voice en-US-DavisNeural

Designed for clips exported by OpenIV (AWC -> WAV). Output WAVs are PCM 16-bit
and can be re-imported into the matching AWC streams via OpenIV.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Reuse the proven SimpleVox stages instead of re-implementing them.
from transcribe import select_device
from replacements import find_matches
from generate_audio import process_replacement
from splice_audio import (
    build_censored_audio,
    encode_audio_only,
    get_audio_properties,
    CROSSFADE_MS,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".m4b", ".flac", ".ogg", ".aac"}


def log(msg: str) -> None:
    print(msg, flush=True)


def discover_clips(path: Path) -> list[Path]:
    """Return a sorted list of audio files at/under `path`."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        )
    return []


def relative_to(clip: Path, root: Path) -> Path:
    try:
        return clip.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(clip.name)


# --------------------------------------------------------------------------- #
# Model loading (ONCE per batch)
# --------------------------------------------------------------------------- #

def load_models(model_name: str, device: str | None, language: str):
    """Load the WhisperX ASR model and alignment model once.

    Returns (asr_model, align_model, align_metadata, device, language).
    """
    import whisperx

    if device is None:
        device = select_device()

    compute_type = "float16" if device == "cuda" else "int8"
    lang = None if (language and language.lower() == "auto") else (language or "en")

    t0 = time.time()
    log(f"[model] Loading WhisperX ASR model '{model_name}' on {device} "
        f"(compute_type={compute_type})...")
    asr_model = whisperx.load_model(
        model_name, device, compute_type=compute_type, language=lang,
    )
    log(f"[model] ASR model loaded in {time.time() - t0:.1f}s")

    align_lang = lang or "en"
    t0 = time.time()
    log(f"[model] Loading alignment model for '{align_lang}'...")
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=align_lang, device=device,
        )
        log(f"[model] Alignment model loaded in {time.time() - t0:.1f}s")
    except Exception as exc:
        log(f"[model] [warning] Could not load alignment model ({exc}); "
            "word timestamps may be unavailable.")
        align_model, align_metadata = None, None

    return asr_model, align_model, align_metadata, device, lang


def transcribe_clip(
    audio_path: str,
    asr_model,
    align_model,
    align_metadata,
    device: str,
    language: str | None,
) -> list[dict]:
    """Transcribe one clip -> list of {"word","start","end"} using loaded models.

    Mirrors transcribe.transcribe_audio but reuses pre-loaded models.
    """
    import whisperx

    audio = whisperx.load_audio(audio_path)
    result = asr_model.transcribe(audio, batch_size=8, language=language)

    if align_model is not None:
        try:
            result = whisperx.align(
                result["segments"], align_model, align_metadata,
                audio, device, return_char_alignments=False,
            )
        except Exception as exc:
            log(f"  [align] [warning] alignment failed ({exc}); "
                "falling back to segment-level words")

    words: list[dict] = []
    skipped = 0
    for seg in result.get("segments", []):
        for w in seg.get("words", []) or []:
            start = w.get("start")
            end = w.get("end")
            if start is None or end is None:
                skipped += 1
                continue
            words.append({
                "word": (w.get("word") or "").strip(),
                "start": float(start),
                "end": float(end),
            })
    if skipped:
        log(f"  [words] Skipped {skipped} token(s) with missing timestamps.")
    return words


# --------------------------------------------------------------------------- #
# Per-clip censoring
# --------------------------------------------------------------------------- #

def censor_clip(
    clip_path: Path,
    output_path: Path,
    asr_model,
    align_model,
    align_metadata,
    device: str,
    language: str | None,
    voice: str,
    lead_in_ms: int | None = None,
) -> dict:
    """Censor a single clip. Returns a manifest record dict."""
    record = {
        "clip": str(clip_path),
        "status": "error",
        "words_found": [],
        "replacements": [],
    }

    # 1) Transcribe
    try:
        words = transcribe_clip(
            str(clip_path), asr_model, align_model, align_metadata,
            device, language,
        )
    except Exception as exc:
        record["error"] = f"transcribe: {type(exc).__name__}: {exc}"
        return record

    # 2) Filter
    matches = find_matches(words)
    record["words_found"] = [m["word"] for m in matches]
    record["replacements"] = [
        {"word": m["word"], "replacement": m["replacement"],
         "start": m["start"], "end": m["end"]}
        for m in matches
    ]

    if not matches:
        record["status"] = "clean"
        return record

    # 3+4+5) Generate replacements, splice, encode
    import tempfile
    work_dir = Path(tempfile.mkdtemp(prefix="batchcensor_"))
    try:
        gen_dir = work_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)

        generated = 0
        for entry in matches:
            # process_replacement takes a dict with replacement/start/end.
            r = process_replacement(entry, gen_dir, voice)
            if r is not None:
                generated += 1
        if generated == 0:
            record["status"] = "error"
            record["error"] = "all replacement audio generations failed"
            return record

        props = get_audio_properties(str(clip_path))
        sample_rate = props["sample_rate"]
        channels = props["channels"]

        censored_wav = str(work_dir / "censored.wav")
        ok, succ, skip = build_censored_audio(
            audio_path=str(clip_path),
            replacements=matches,
            audio_dir=gen_dir,
            output_wav=censored_wav,
            sample_rate=sample_rate,
            channels=channels,
            crossfade_ms=CROSSFADE_MS,
            lead_in_ms=lead_in_ms,
        )
        if not ok:
            record["status"] = "error"
            record["error"] = "build_censored_audio failed"
            return record

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not encode_audio_only(censored_wav, str(output_path)):
            record["status"] = "error"
            record["error"] = "encode_audio_only failed"
            return record

        record["status"] = "censored"
        record["spliced"] = succ
        record["skipped_splice"] = skip
        return record
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Batch-censor profanity in a folder of audio clips. "
                    "Loads the WhisperX model ONCE for the whole batch "
                    "(use this, not run.py, for large sets of short clips).",
    )
    p.add_argument("input", help="Input clip file or folder (recursive).")
    p.add_argument("output", help="Output folder (censored clips written here).")
    p.add_argument("--voice", default="en-US-GuyNeural",
                   help="edge-tts voice for replacements (default en-US-GuyNeural).")
    p.add_argument("--model", default="small", help="Whisper model size (default small).")
    p.add_argument("--device", default=None, help="cuda/cpu (default auto).")
    p.add_argument("--language", default="en",
                   help="Language code or 'auto' (default en).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip clips whose output file already exists.")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Keep going if a clip fails (batch mode).")
    p.add_argument("--copy-clean", action="store_true",
                   help="Also copy clean (no-profanity) clips to output. "
                        "Default: output holds ONLY censored clips, so you can "
                        "selectively re-import just the changed ones.")
    p.add_argument("--lead-in-ms", type=int, default=None,
                   help=("Silence (ms) inserted before each replacement to mute "
                         "the leaked onset of the original word. Default: "
                         "per-family auto (50ms for ass/shit, 0ms for god/fuck/"
                         "damn). Pass 0 to disable, or a value to override all."))
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N clips (for testing).")
    args = p.parse_args(argv)

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        log("[FAILED] ffmpeg/ffprobe not on PATH.")
        return 1

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        log(f"[FAILED] Input not found: {input_path}")
        return 1

    clips = discover_clips(input_path)
    if not clips:
        log(f"[FAILED] No audio files found at/under: {input_path}")
        return 1
    if args.limit:
        clips = clips[: args.limit]

    log("=" * 78)
    log("  BATCH CENSOR")
    log("=" * 78)
    log(f"  Input:          {input_path}")
    log(f"  Output:         {output_dir}")
    log(f"  Clips:          {len(clips)}")
    log(f"  Voice:          {args.voice}")
    log(f"  Whisper model:  {args.model}")
    log(f"  Skip existing:  {args.skip_existing}")
    log(f"  Continue err:   {args.continue_on_error}")
    log(f"  Copy clean:     {args.copy_clean}")
    lead_desc = (f"{args.lead_in_ms}ms (all words)" if args.lead_in_ms is not None
                 else "per-family auto")
    log(f"  Lead-in pad:    {lead_desc}")
    log("  (Model is loaded ONCE for the whole batch.)")

    asr_model, align_model, align_metadata, device, lang = load_models(
        args.model, args.device, args.language,
    )

    manifest_path = output_dir / "_censor_manifest.jsonl"
    counts = {"censored": 0, "clean": 0, "skipped": 0, "error": 0}
    t_start = time.time()

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for i, clip in enumerate(clips, start=1):
            rel = relative_to(clip, input_path)
            out_clip = output_dir / rel

            if args.skip_existing and out_clip.is_file():
                counts["skipped"] += 1
                manifest.write(json.dumps({
                    "clip": str(clip), "status": "skipped",
                }, ensure_ascii=False) + "\n")
                log(f"[{i}/{len(clips)}] SKIP (exists): {rel}")
                continue

            log(f"[{i}/{len(clips)}] {rel}")
            try:
                rec = censor_clip(
                    clip_path=clip,
                    output_path=out_clip,
                    asr_model=asr_model,
                    align_model=align_model,
                    align_metadata=align_metadata,
                    device=device,
                    language=lang,
                    voice=args.voice,
                    lead_in_ms=args.lead_in_ms,
                )
            except Exception as exc:
                rec = {"clip": str(clip), "status": "error",
                       "error": f"{type(exc).__name__}: {exc}"}

            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            rec["relative"] = str(rel)
            manifest.write(json.dumps(rec, ensure_ascii=False) + "\n")
            manifest.flush()

            if rec["status"] == "clean":
                log(f"  -> clean (no profanity)")
                if args.copy_clean:
                    out_clip.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(clip, out_clip)
            elif rec["status"] == "censored":
                words = ", ".join(
                    f"{w['word']}->{w['replacement']}"
                    for w in rec["replacements"]
                )
                log(f"  -> CENSORED ({len(rec['replacements'])}): {words}")
            else:
                err = rec.get("error", "unknown")
                log(f"  -> ERROR: {err}")
                if not args.continue_on_error:
                    log("[STOP] A clip failed and --continue-on-error not set.")
                    break

    elapsed = time.time() - t_start
    log("=" * 78)
    log("  BATCH SUMMARY")
    log("=" * 78)
    log(f"  Censored: {counts['censored']}")
    log(f"  Clean:    {counts['clean']}")
    log(f"  Skipped:  {counts['skipped']}")
    log(f"  Errors:   {counts['error']}")
    log(f"  Elapsed:  {elapsed:.1f}s")
    log(f"  Manifest: {manifest_path}")
    log(f"  Output:   {output_dir}  (censored clips only, unless --copy-clean)")
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
