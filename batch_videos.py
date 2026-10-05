#!/usr/bin/env python
"""
Batch profanity censor for a folder of VIDEOS with a persistent WhisperX model.

Identical 4-stage pipeline to run.py, but the WhisperX ASR model and the
alignment model are loaded ONCE for the whole batch instead of once per file
(run.py reloads both on every episode, ~15-30s of pure overhead each time —
prohibitive for a 300+ episode TV library).

Reuses the proven stage modules directly (no re-implementation):
    1. Transcribe  (whisperx, models cached across files)
    2. Filter      (replacements.find_matches)
    3. Generate    (generate_audio.process_replacement, edge-tts)
    4. Splice+Mux  (splice_audio.splice_audio_ffmpeg, per-family lead-in auto)

Resume safety:
    --skip-existing     skip episodes whose final output video already exists
    valid words.json    an existing transcription is reused (not re-transcribed)
    _manifest.jsonl     one record per episode (ok / clean / error) for audit

Usage:
    python batch_videos.py "E:\\Videos\\Unedited\\Extracted" "E:\\Videos\\Unedited\\Censored"
    python batch_videos.py IN OUT --limit 1          # test one episode
    python batch_videos.py IN OUT --skip-existing --continue-on-error
"""

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

from replacements import find_matches
from generate_audio import process_replacement, format_filename
from splice_audio import splice_audio_ffmpeg
from transcribe import select_device

VIDEO_EXTS = {".mkv", ".mp4", ".m4v", ".mov", ".avi", ".webm", ".wmv", ".flv"}

# Passed as faster-whisper's initial_prompt. Whisper has a politeness bias:
# it sometimes DELETES short profane utterances ("What the fuck?" at S02E07
# 09:35 never made it into the pass-2 transcript) or swaps in a euphemism
# ("heck", "f***ing"). The prompt tells it to transcribe verbatim instead.
# whisperx's FasterWhisperPipeline.transcribe() does NOT forward initial_prompt
# (silently dropped from asr_options too), so Transcriber below calls the
# underlying faster-whisper model directly and uses whisperx only for word
# alignment.
VERBATIM_PROMPT = (
    "The following is a verbatim transcript of an uncensored adult animated "
    "comedy. Profanity is transcribed exactly as spoken: fuck, fucking, shit, "
    "bitch, asshole."
)


def merge_extra_words(words: list[dict], extra: list[dict]) -> list[dict]:
    """Merge confirmed extra tokens (audit extras / swap suspects / bleed-fix
    windows) into a fresh transcription, dropping extras that overlap a
    transcribed token (the fresh pass already heard that spot) or each other.
    Keeps the transcribed token on overlap - its timestamps match this audio."""
    def overlaps(a, b):
        return min(a["end"], b["end"]) - max(a["start"], b["start"]) > 0.05

    merged = list(words)
    for e in sorted(extra, key=lambda x: x["start"]):
        if any(overlaps(e, w) for w in merged):
            continue
        merged.append(e)
    return sorted(merged, key=lambda w: w["start"])


def log(msg: str = "") -> None:
    print(msg, flush=True)


def discover_videos(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


class Transcriber:
    """WhisperX ASR + alignment models loaded once, reused for every file."""

    def __init__(self, model_name: str, device: str | None, batch_size: int):
        import whisperx

        if device is None:
            device = select_device()
        compute_type = "float16" if device == "cuda" else "int8"

        t0 = time.time()
        log(f"[model] Loading WhisperX '{model_name}' on {device} "
            f"(compute_type={compute_type})...")
        self.whisperx = whisperx
        self.batch_size = batch_size
        self.device = device
        self.model = whisperx.load_model(
            model_name, device, compute_type=compute_type, language="en",
        )
        log(f"[model] ASR model loaded in {time.time() - t0:.1f}s")

        t0 = time.time()
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code="en", device=device,
        )
        log(f"[model] alignment model loaded in {time.time() - t0:.1f}s")

    def transcribe(self, media_path: Path) -> list[dict]:
        """Transcribe via faster-whisper (verbatim prompt) + whisperx alignment."""
        wx = self.whisperx
        audio = wx.load_audio(str(media_path))
        fw_segments, _info = self.model.model.transcribe(
            audio, language="en", beam_size=5, vad_filter=True,
            initial_prompt=VERBATIM_PROMPT,
        )
        segments = [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in fw_segments
        ]
        result = wx.align(
            segments, self.align_model, self.align_metadata,
            audio, self.device, return_char_alignments=False,
        )
        words: list[dict] = []
        for segment in result.get("segments", []):
            for w in segment.get("words", []) or []:
                if w.get("start") is None or w.get("end") is None:
                    continue
                words.append({
                    "word": (w.get("word") or "").strip(),
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                })
        return words


def generate_wavs(matches: list[dict], audio_dir: Path, voice: str) -> tuple[int, int]:
    """Stage 3 equivalent (generate_audio.main with --skip-existing semantics)."""
    ok = failed = 0
    for entry in matches:
        expected = audio_dir / format_filename(
            entry["start"], entry["end"], entry["replacement"])
        if expected.is_file() and expected.stat().st_size > 500:
            ok += 1
            continue
        result = process_replacement(entry=entry, output_dir=audio_dir, voice=voice)
        if result is not None:
            ok += 1
        else:
            failed += 1
    return ok, failed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("input_dir")
    p.add_argument("output_dir")
    p.add_argument("--voice", default="en-US-GuyNeural")
    p.add_argument("--model", default="small")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N videos (testing).")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args(argv)

    input_root = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_root.is_dir():
        log(f"[error] Input dir not found: {input_root}")
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(input_root)
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        log(f"[error] No video files under {input_root}")
        return 2

    manifest_path = output_dir / "_manifest.jsonl"
    log(f"[batch] {len(videos)} video(s) under {input_root}")
    log(f"[batch] output -> {output_dir}")
    log(f"[batch] voice={args.voice} model={args.model} "
        f"skip_existing={args.skip_existing} continue_on_error={args.continue_on_error}")

    transcriber = Transcriber(args.model, args.device, args.batch_size)

    counts = {"ok": 0, "clean": 0, "error": 0, "skipped": 0}
    t_batch = time.time()

    for idx, video_path in enumerate(videos, start=1):
        rel = video_path.relative_to(input_root)
        final_output = output_dir / rel
        intermediate_dir = output_dir / "_intermediate" / rel.parent
        words_json = intermediate_dir / f"{video_path.stem}.json"
        replacements_json = intermediate_dir / f"{video_path.stem}_replacements.json"
        audio_dir = intermediate_dir / "generated_audio"

        final_output.parent.mkdir(parents=True, exist_ok=True)
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and final_output.is_file():
            log(f"\n[{idx}/{len(videos)}] {rel} — SKIP (output exists)")
            counts["skipped"] += 1
            continue

        t0 = time.time()
        record = {"file": str(rel), "status": None}
        log(f"\n{'=' * 70}\n[{idx}/{len(videos)}] {rel}")

        try:
            # --- Stage 1: transcribe (reuse valid existing JSON if present) ---
            words = None
            if words_json.is_file():
                try:
                    words = json.loads(words_json.read_text(encoding="utf-8"))
                    log(f"  [1/4] transcribe: reusing {len(words)} words "
                        f"from existing JSON")
                except (json.JSONDecodeError, OSError):
                    words = None
            if words is None:
                words = transcriber.transcribe(video_path)
                words_json.parent.mkdir(parents=True, exist_ok=True)
                words_json.write_text(
                    json.dumps(words, ensure_ascii=False, indent=2),
                    encoding="utf-8")
                log(f"  [1/4] transcribed {len(words)} words")

            # Confirmed extra tokens from earlier audit passes (gap extras,
            # euphemism swaps, f-word bleed-fix windows) live beside the
            # transcription as <stem>_extra_words.json; merge non-overlapping.
            extra_json = intermediate_dir / f"{video_path.stem}_extra_words.json"
            if extra_json.is_file():
                try:
                    extra = json.loads(extra_json.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    extra = []
                if extra:
                    before = len(words)
                    words = merge_extra_words(words, extra)
                    log(f"  [1/4] merged {len(words) - before} extra word(s) "
                        f"from {extra_json.name}")

            # --- Stage 2: profanity filter ---
            matches = find_matches(words)
            replacements_json.parent.mkdir(parents=True, exist_ok=True)
            replacements_json.write_text(
                json.dumps(matches, ensure_ascii=False, indent=2),
                encoding="utf-8")
            log(f"  [2/4] {len(matches)} profane word(s) found")

            if not matches:
                shutil.copy2(video_path, final_output)
                log(f"  [2/4] no profanity — copied original to output")
                record.update(status="clean", replacements=0)
                counts["clean"] += 1
            else:
                # --- Stage 3: TTS wavs ---
                ok, failed = generate_wavs(matches, audio_dir, args.voice)
                log(f"  [3/4] generated {ok}/{len(matches)} wav(s) "
                    f"({failed} failed)")
                if failed:
                    raise RuntimeError(f"TTS failed for {failed} word(s)")

                # --- Stage 4: splice + mux ---
                success = splice_audio_ffmpeg(
                    audio_path=str(video_path),
                    replacements=matches,
                    audio_dir=audio_dir,
                    output_path=str(final_output),
                    lead_in_ms=None,  # per-family auto (50ms ass/shit)
                )
                if not success or not final_output.is_file():
                    raise RuntimeError("splice/mux failed")
                record.update(status="ok", replacements=len(matches))
                counts["ok"] += 1

        except Exception as exc:
            log(f"  [ERROR] {type(exc).__name__}: {exc}")
            traceback.print_exc()
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            counts["error"] += 1
            if not args.continue_on_error:
                log("[STOP] --continue-on-error not set; aborting batch.")
                break

        record["seconds"] = round(time.time() - t0, 1)
        with manifest_path.open("a", encoding="utf-8") as mf:
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")
        log(f"  [{record['status']}] in {record['seconds']}s")

    elapsed = time.time() - t_batch
    log(f"\n{'=' * 70}\n[batch done] {counts} in {elapsed / 3600:.2f}h "
        f"({elapsed / max(1, sum(counts.values())):.1f}s avg/episode)")
    log(f"[manifest] {manifest_path}")
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
