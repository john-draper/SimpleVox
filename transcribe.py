#!/usr/bin/env python
"""
Transcribe a video (or audio) file with WhisperX and export word-level
timestamps to JSON.

SimpleVox does NOT run speaker diarization — every replacement word is spoken
by the same generic voice, so speaker labels are unnecessary. This makes the
transcription stage simpler, faster, and free of the HuggingFace token /
gated-repo requirements that the full revox pipeline needed.

WhisperX (and ffmpeg underneath it) reads the audio track directly out of any
common video container (.mkv, .mp4, .m4v, .mov, .avi, .webm, ...), so this stage
is identical for video as for audio — only the audio is transcribed; the video
stream is ignored here.

Output JSON is a list of objects:
    {"word": str, "start": float, "end": float}

Designed for an NVIDIA CUDA GPU. Falls back to CPU automatically if CUDA is
unavailable.

Example:
    python transcribe.py "input/movie.mkv"
    python transcribe.py input.mkv -o output/words.json --model small
"""

import argparse
import json
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def select_device() -> str:
    """Return "cuda" if available, otherwise "cpu"."""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            log(f"[device] Using CUDA GPU: {gpu_name}")
            return "cuda"
    except Exception as exc:  # pragma: no cover - torch may not be importable yet
        log(f"[device] Could not query CUDA via torch ({exc}); defaulting to device arg.")
    return "cpu"


def transcribe_audio(
    audio_path: str,
    language: str | None,
    model_name: str,
    batch_size: int,
    compute_type: str,
    device: str | None,
    output_path: str | None,
) -> list[dict]:
    """Transcribe audio and return a list of word dicts with timestamps."""
    import whisperx

    if device is None:
        device = select_device()

    # On CUDA, float16 is the recommended/safe default; let users override.
    if device == "cuda" and compute_type == "auto":
        compute_type = "float16"

    audio_file = Path(audio_path)
    if not audio_file.is_file():
        raise FileNotFoundError(f"Input file not found: {audio_file}")

    log(f"[input] {audio_file}")

    # 1) Load model
    t0 = time.time()
    log(f"[model] Loading WhisperX model '{model_name}' on {device} "
        f"(compute_type={compute_type})...")
    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language,
    )
    log(f"[model] Loaded in {time.time() - t0:.1f}s")

    # 2) Load audio (decoded via ffmpeg -> mono 16kHz float32 numpy array).
    #    ffmpeg auto-selects the audio track from a video container.
    t0 = time.time()
    log("[audio] Decoding audio track with ffmpeg "
        "(this may take a moment for long files)...")
    audio = whisperx.load_audio(str(audio_file))
    log(f"[audio] Decoded in {time.time() - t0:.1f}s "
        f"({audio.shape[0] / 16000 / 60:.1f} min of audio)")

    # 3) Transcribe
    t0 = time.time()
    log("[transcribe] Running transcription (VAD-batched)...")
    result = model.transcribe(
        audio,
        batch_size=batch_size,
        language=language,
    )
    detected_language = result.get("language", language or "en")
    log(f"[transcribe] Done in {time.time() - t0:.1f}s "
        f"(detected/forced language: {detected_language})")

    # 4) Align for accurate word-level timestamps
    t0 = time.time()
    log(f"[align] Loading alignment model for language '{detected_language}'...")
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=device,
        )
    except ValueError as exc:
        # Language not supported by the default alignment model.
        raise SystemExit(
            f"[align] No alignment model for language '{detected_language}' ({exc}). "
            "Try --language en or a different alignment model."
        ) from exc

    log("[align] Aligning word timestamps...")
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    log(f"[align] Done in {time.time() - t0:.1f}s")

    # 5) Flatten segment word lists into [{word, start, end}, ...]
    #    (No diarization — SimpleVox uses one voice for all words.)
    words: list[dict] = []
    skipped = 0
    for segment in result.get("segments", []):
        for w in segment.get("words", []) or []:
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
        log(f"[words] Skipped {skipped} token(s) with missing timestamps.")

    # 6) Write JSON
    if output_path is None:
        output_path = str(audio_file.with_suffix(".json"))
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)

    duration_min = (words[-1]["end"] - words[0]["start"]) / 60.0 if words else 0.0
    log(f"[done] Wrote {len(words)} words to {out_path}")
    if words:
        log(f"[done] Time span: {words[0]['start']:.2f}s -> {words[-1]['end']:.2f}s "
            f"({duration_min:.1f} min)")
    return words


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transcribe a video/audio file with WhisperX and export "
                    "word-level JSON timestamps.",
    )
    p.add_argument(
        "audio",
        help="Path to the video (or audio) file (e.g. input/movie.mkv)",
    )
    p.add_argument(
        "-o", "--output",
        default=None,
        help="Output JSON path (default: <input>.json next to the input).",
    )
    p.add_argument(
        "--language",
        default="en",
        help="Language code, e.g. 'en' (default: en). Use 'auto' to auto-detect.",
    )
    p.add_argument(
        "--model",
        default="small",
        help="Whisper model size/name (default: small). e.g. medium, large-v3.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="VAD batch size for transcription (default: 8). Lower if you hit CUDA OOM.",
    )
    p.add_argument(
        "--compute-type",
        default="auto",
        help="ctranslate2 compute type (default: auto -> float16 on CUDA). "
             "Try 'int8' to cut VRAM usage if you run out of memory.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device: 'cuda' or 'cpu' (default: cuda if available).",
    )
    args = p.parse_args(argv)
    if args.language and args.language.lower() == "auto":
        args.language = None  # let Whisper auto-detect
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        transcribe_audio(
            audio_path=args.audio,
            language=args.language,
            model_name=args.model,
            batch_size=args.batch_size,
            compute_type=args.compute_type,
            device=args.device,
            output_path=args.output,
        )
    except FileNotFoundError as exc:
        log(f"[error] {exc}")
        return 2
    except SystemExit as exc:
        log(str(exc))
        return 1
    except KeyboardInterrupt:
        log("[abort] Interrupted by user.")
        return 130
    except Exception as exc:
        log(f"[error] {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())