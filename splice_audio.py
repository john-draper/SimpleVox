#!/usr/bin/env python
"""
Splice generated replacement audio back into the original VIDEO file using
ffmpeg directly (via subprocess) to avoid loading huge files into memory.

The original VIDEO stream is copied losslessly (-c:v copy); only the audio
track is edited and re-written. The result is a new video file with the SAME
video quality and a CENSORED audio track.

Strategy:
  1. Sort replacements by start time.
  2. Use ffmpeg to extract each "gap" segment from the original
     (the audio between consecutive replacement words).
  3. Build a concat list: [gap_0, replacement_0.wav, gap_1, replacement_1.wav, ...]
  4. Use ffmpeg's concat demuxer to join all segments into an intermediate
     censored-audio WAV.
  5. Each replacement is time-stretched/trimmed to EXACTLY match the original
     word's duration so the audio stays in sync with the video.
  6. Mux the censored audio with the original (losslessly-copied) video stream.

Requirements:
    - ffmpeg installed system-wide
    - ffprobe installed system-wide (usually comes with ffmpeg)

Example:
    python splice_audio.py "input/movie.mkv" \\
        --replacements-json "output/movie_replacements.json" \\
        --audio-dir "output/generated_audio" \\
        --output "output/movie_censored.mkv"
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CROSSFADE_MS = 20  # 20ms fade applied at segment boundaries
DEFAULT_GAIN_DB = 6.0  # fallback boost when measurement fails (TTS is typically quieter)
MAX_GAIN_DB = 24.0  # safety clamp to prevent excessive amplification
MIN_GAIN_DB = -12.0  # safety clamp to prevent excessive attenuation


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# ffprobe helpers
# --------------------------------------------------------------------------- #

def get_audio_duration_seconds(audio_path: str) -> float:
    """Get the duration of an audio/video file in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to get duration of '{audio_path}' with ffprobe: {exc}"
        ) from exc


def get_audio_properties(audio_path: str) -> dict:
    """Get sample rate and channel count of an audio file using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,channels",
                "-of", "default=noprint_wrappers=1",
                audio_path,
            ],
            capture_output=True, text=True, check=True,
        )
        props = {}
        for line in result.stdout.strip().split("\n"):
            if "=" in line:
                key, val = line.split("=", 1)
                props[key.strip()] = val.strip()
        return {
            "sample_rate": int(props.get("sample_rate", 44100)),
            "channels": int(props.get("channels", 2)),
        }
    except (subprocess.CalledProcessError, ValueError, KeyError) as exc:
        log(f"[warning] Could not determine audio properties of '{audio_path}': {exc}")
        return {"sample_rate": 44100, "channels": 2}


# --------------------------------------------------------------------------- #
# Loudness matching
# --------------------------------------------------------------------------- #

def measure_file_mean_volume_db(file_path: str) -> float | None:
    """Measure mean volume (dBFS) of an audio file using ffmpeg volumedetect."""
    cmd = [
        "ffmpeg", "-hide_banner", "-i", file_path,
        "-af", "volumedetect", "-vn", "-f", "null",
        "NUL" if os.name == "nt" else "/dev/null",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"mean_volume:\s*(-?\d+\.?\d*)\s*dB", result.stderr)
    if m:
        return float(m.group(1))
    return None


def measure_segment_mean_volume_db(
    audio_path: str, start_s: float, duration_s: float
) -> float | None:
    """Measure mean volume (dBFS) of a time range in the original audio."""
    if duration_s <= 0.02:
        return None
    cmd = [
        "ffmpeg", "-hide_banner",
        "-ss", f"{start_s:.6f}",
        "-t", f"{duration_s:.6f}",
        "-i", audio_path,
        "-af", "volumedetect", "-vn", "-f", "null",
        "NUL" if os.name == "nt" else "/dev/null",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"mean_volume:\s*(-?\d+\.?\d*)\s*dB", result.stderr)
    if m:
        return float(m.group(1))
    return None


def compute_loudness_gain_db(
    original_audio: str,
    replacement_wav: str,
    start_s: float,
    end_s: float,
) -> float:
    """Compute the gain (dB) so the replacement matches the original loudness."""
    orig_dur = end_s - start_s
    orig_mean = measure_segment_mean_volume_db(original_audio, start_s, orig_dur)
    rep_mean = measure_file_mean_volume_db(replacement_wav)

    if orig_mean is not None and rep_mean is not None:
        gain = orig_mean - rep_mean
        log(f"    [loudness] original={orig_mean:+.1f}dB replacement={rep_mean:+.1f}dB "
            f"-> gain={gain:+.1f}dB")
    else:
        gain = DEFAULT_GAIN_DB
        log(f"    [loudness] measurement failed; using default gain={gain:+.1f}dB")

    return max(MIN_GAIN_DB, min(MAX_GAIN_DB, gain))


# --------------------------------------------------------------------------- #
# Filename convention (must match generate_audio.py)
# --------------------------------------------------------------------------- #

def format_filename(start_ms: int, end_ms: int, replacement: str) -> str:
    safe_replacement = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in replacement
    ).strip("_")
    return f"{start_ms:08d}_{end_ms:08d}_{safe_replacement}.wav"


# --------------------------------------------------------------------------- #
# ffmpeg segment operations
# --------------------------------------------------------------------------- #

def extract_segment(
    audio_path: str,
    start_ms: int,
    end_ms: int,
    output_path: str,
    sample_rate: int,
    channels: int,
    fade_ms: int = CROSSFADE_MS,
) -> None:
    """Extract a segment [start_ms, end_ms] from the source file as PCM WAV."""
    duration_ms = end_ms - start_ms
    if duration_ms <= 0:
        raise ValueError(f"Invalid segment duration: {duration_ms}ms")

    fade_out_start = max(0, (duration_ms - fade_ms) / 1000)

    afilter = (
        f"aformat=sample_rates={sample_rate}:"
        f"channel_layouts={'stereo' if channels == 2 else 'mono'}"
    )
    if fade_ms > 0 and duration_ms > fade_ms * 2:
        fade_dur = fade_ms / 1000
        afilter += f",afade=t=out:st={fade_out_start:.6f}:d={fade_dur:.6f}"

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_ms / 1000:.6f}",
        "-t", f"{duration_ms / 1000:.6f}",
        "-i", audio_path,
        "-vn",  # never copy video into an audio segment
        "-af", afilter,
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )


def convert_replacement_wav(
    input_wav: str,
    output_wav: str,
    sample_rate: int,
    channels: int,
    fade_ms: int = CROSSFADE_MS,
    target_duration_s: float | None = None,
    gain_db: float = 0.0,
) -> None:
    """Convert a replacement WAV to match the original audio's format and duration.

    Trims leading/trailing silence, applies loudness gain, time-stretches to the
    target duration (keeping A/V sync), applies fades, and pads/trims to exact
    length.
    """
    try:
        duration = get_audio_duration_seconds(input_wav)
    except RuntimeError:
        duration = 1.0  # fallback

    parts = [
        f"aformat=sample_rates={sample_rate}:"
        f"channel_layouts={'stereo' if channels == 2 else 'mono'}"
    ]

    # Trim leading AND trailing silence so only the spoken word remains.
    parts.append(
        "silenceremove=start_periods=1:start_silence=0.01:"
        "start_threshold=-45dB:detection=peak"
    )
    parts.append("areverse")
    parts.append(
        "silenceremove=start_periods=1:start_silence=0.01:"
        "start_threshold=-45dB:detection=peak"
    )
    parts.append("areverse")

    # Loudness gain (applied to the raw TTS signal).
    if abs(gain_db) > 0.01:
        parts.append(f"volume={gain_db:+.2f}dB")

    # Time-stretch the SPEECH to fill the target duration (keeps A/V sync).
    if target_duration_s is not None and target_duration_s > 0.02 and duration > 0.02:
        estimated_speech = max(0.1, duration - 0.9)
        speed_factor = estimated_speech / target_duration_s
        if speed_factor > 1.05:
            remaining = speed_factor
            while remaining > 2.0:
                parts.append("atempo=2.0")
                remaining /= 2.0
            parts.append(f"atempo={remaining:.4f}")
        elif speed_factor < 0.95:
            parts.append(f"atempo={max(0.5, speed_factor):.4f}")

    effective_dur = target_duration_s if target_duration_s else duration
    fade_dur = min(fade_ms / 1000, effective_dur / 4) if effective_dur > 0.02 else 0
    fade_out_start = max(0, effective_dur - fade_dur)
    if fade_dur > 0:
        parts.append(f"afade=t=in:st=0:d={fade_dur:.6f}")
        parts.append(f"afade=t=out:st={fade_out_start:.6f}:d={fade_dur:.6f}")

    # Pad to exact duration.
    if target_duration_s is not None and target_duration_s > 0:
        parts.append(f"apad=whole_dur={target_duration_s:.6f}")

    afilter = ",".join(parts)

    cmd = ["ffmpeg", "-y", "-i", input_wav, "-af", afilter]
    if target_duration_s is not None and target_duration_s > 0:
        cmd += ["-t", f"{target_duration_s:.6f}"]
    cmd += ["-acodec", "pcm_s16le", "-ar", str(sample_rate),
            "-ac", str(channels), output_wav]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )


# --------------------------------------------------------------------------- #
# Core splice logic
# --------------------------------------------------------------------------- #

def load_replacements(path: str) -> list[dict]:
    in_path = Path(path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Replacements JSON not found: {in_path}")
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list of replacement objects, got {type(data).__name__}."
        )
    return data


def build_censored_audio(
    audio_path: str,
    replacements: list[dict],
    audio_dir: Path,
    output_wav: str,
    sample_rate: int,
    channels: int,
    crossfade_ms: int,
) -> tuple[bool, int, int]:
    """Concatenate gaps + replacement audio into a single censored WAV.

    Returns (success, success_count, skipped_count).
    """
    total_duration_ms = int(get_audio_duration_seconds(audio_path) * 1000)
    log(f"[splice] Total audio duration: {total_duration_ms / 1000:.1f}s")

    sorted_replacements = sorted(replacements, key=lambda r: float(r.get("start", 0)))
    segment_files: list[str] = []
    prev_end_ms = 0
    success_count = 0
    skipped_count = 0

    temp_dir = Path(output_wav).parent

    for i, entry in enumerate(sorted_replacements):
        word = entry.get("word", "")
        replacement = entry.get("replacement", "")
        start_ms = int(float(entry.get("start", 0.0)) * 1000)
        end_ms = int(float(entry.get("end", 0.0)) * 1000)

        filename = format_filename(start_ms, end_ms, replacement)
        wav_path = audio_dir / filename

        if not wav_path.is_file():
            log(f"[{i + 1}/{len(sorted_replacements)}] "
                f"[{start_ms:08d}-{end_ms:08d}] "
                f"({word:>12} -> {replacement:>12}) SKIP: .wav not found")
            skipped_count += 1
            continue

        # --- Extract the gap segment before this replacement ---
        if start_ms > prev_end_ms:
            gap_path = str(temp_dir / f"gap_{i:04d}.wav")
            log(f"[{i + 1}/{len(sorted_replacements)}] "
                f"Extracting gap [{prev_end_ms}-{start_ms}]ms "
                f"({start_ms - prev_end_ms}ms)...")
            try:
                extract_segment(
                    audio_path, prev_end_ms, start_ms, gap_path,
                    sample_rate, channels, crossfade_ms
                )
                segment_files.append(gap_path)
            except subprocess.CalledProcessError as exc:
                log(f"  ERROR extracting gap: "
                    f"{exc.stderr[:200] if exc.stderr else 'unknown'}")
                skipped_count += 1
                continue

        # --- Convert and add the replacement audio ---
        # Target duration = exact length of the original word (keeps A/V sync)
        target_dur = (end_ms - start_ms) / 1000.0
        gain_db = compute_loudness_gain_db(
            original_audio=audio_path,
            replacement_wav=str(wav_path),
            start_s=start_ms / 1000.0,
            end_s=end_ms / 1000.0,
        )
        converted_rep_path = str(temp_dir / f"rep_{i:04d}.wav")
        try:
            convert_replacement_wav(
                str(wav_path), converted_rep_path,
                sample_rate, channels, crossfade_ms,
                target_duration_s=target_dur,
                gain_db=gain_db,
            )
            segment_files.append(converted_rep_path)
            success_count += 1
            log(f"[{i + 1}/{len(sorted_replacements)}] "
                f"[{start_ms:08d}-{end_ms:08d}] "
                f"({word:>12} -> {replacement:>12}) OK")
        except subprocess.CalledProcessError as exc:
            log(f"  ERROR converting replacement: "
                f"{exc.stderr[:200] if exc.stderr else 'unknown'}")
            skipped_count += 1

        prev_end_ms = end_ms

    # --- Extract the final gap (after last replacement to end) ---
    if prev_end_ms < total_duration_ms:
        final_gap_path = str(temp_dir / "gap_final.wav")
        log(f"[final] Extracting final gap [{prev_end_ms}-{total_duration_ms}]ms "
            f"({total_duration_ms - prev_end_ms}ms)...")
        try:
            extract_segment(
                audio_path, prev_end_ms, total_duration_ms, final_gap_path,
                sample_rate, channels, fade_ms=0
            )
            segment_files.append(final_gap_path)
        except subprocess.CalledProcessError as exc:
            log(f"  ERROR extracting final gap: "
                f"{exc.stderr[:200] if exc.stderr else 'unknown'}")

    log(f"[splice] Segments prepared: {len(segment_files)} files "
        f"({success_count} replacements, {skipped_count} skipped)")

    # --- Build ffmpeg concat list ---
    concat_list_path = temp_dir / "concat_list.txt"
    with concat_list_path.open("w", encoding="utf-8") as f:
        for seg in segment_files:
            seg_path = Path(seg)
            rel_path = (
                seg_path.relative_to(temp_dir)
                if seg_path.is_relative_to(temp_dir) else seg_path
            )
            rel_str = str(rel_path).replace("\\", "/").replace("'", r"\'")
            f.write(f"file '{rel_str}'\n")

    log(f"[concat] Concatenating {len(segment_files)} segments -> {output_wav}")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "concat_list.txt",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        str(Path(output_wav).resolve()),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(temp_dir))
    if result.returncode != 0:
        log("[error] ffmpeg concat demuxer failed!")
        log(f"  stderr (last 1000 chars): {result.stderr[-1000:]}")
        return False, success_count, skipped_count

    if not Path(output_wav).is_file():
        log(f"[error] Censored audio not created: {output_wav}")
        return False, success_count, skipped_count

    audio_dur = get_audio_duration_seconds(output_wav)
    log(f"[concat] Censored audio created: {output_wav} ({audio_dur / 3600:.2f}h)")
    return True, success_count, skipped_count


def mux_video_audio(
    original_video: str,
    censored_audio: str,
    output_path: str,
    reencode_video: bool = False,
) -> bool:
    """Mux the original video stream (copied losslessly) with the censored audio."""
    log(f"[mux] Muxing original video + censored audio -> {output_path}")

    cmd = [
        "ffmpeg", "-y",
        "-i", original_video,
        "-i", censored_audio,
        "-map", "0:v:0",     # video from original
        "-map", "1:a:0",     # audio from censored track
    ]

    if reencode_video:
        log("[mux] Video: RE-ENCODING to H.264 (slower, for compatibility)")
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    else:
        log("[mux] Video: STREAM COPY (fast, lossless)")

    # Audio: encode to AAC (widely compatible)
    cmd += [
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",  # stop at the shorter stream (audio matches video length)
    ]

    # Preserve metadata/subtitles/chapters from the original where possible
    cmd += ["-map_metadata", "0"]
    cmd.append(output_path)

    log(f"[mux] Running: {' '.join(cmd[:6])} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log("[error] ffmpeg mux failed!")
        log(f"  stderr (last 1500 chars): {result.stderr[-1500:]}")
        return False

    if not Path(output_path).is_file():
        log(f"[error] Output file not created: {output_path}")
        return False

    output_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    output_duration = get_audio_duration_seconds(output_path)
    log(f"[mux] Output created: {output_path} "
        f"({output_size_mb:.1f} MB, {output_duration / 3600:.2f}h)")
    return True


def splice_audio_ffmpeg(
    audio_path: str,
    replacements: list[dict],
    audio_dir: Path,
    output_path: str,
    crossfade_ms: int = CROSSFADE_MS,
    dry_run: bool = False,
) -> bool:
    """Main splice entry point for video files."""

    if dry_run:
        log("[dry-run] Validating inputs without processing...")
        sorted_replacements = sorted(replacements, key=lambda r: float(r.get("start", 0)))
        missing = 0
        for entry in sorted_replacements:
            replacement = entry.get("replacement", "")
            start_ms = int(float(entry.get("start", 0.0)) * 1000)
            end_ms = int(float(entry.get("end", 0.0)) * 1000)
            filename = format_filename(start_ms, end_ms, replacement)
            wav_path = audio_dir / filename
            status = "OK" if wav_path.is_file() else "MISSING"
            if not wav_path.is_file():
                missing += 1
            log(f"  [{start_ms:08d}-{end_ms:08d}] "
                f"{entry.get('word', '')} -> {replacement} [{status}]")
        log(f"[dry-run] {len(sorted_replacements)} replacements, "
            f"{missing} missing .wav files")
        return missing == 0

    if not replacements:
        log("[warning] No replacements found. Output will be identical to input.")

    props = get_audio_properties(audio_path)
    sample_rate = props["sample_rate"]
    channels = props["channels"]
    log(f"[audio] Source audio: {sample_rate} Hz, {channels} channel(s)")

    temp_dir = Path(tempfile.mkdtemp(prefix="splice_"))
    try:
        censored_audio_wav = str(temp_dir / "censored_audio.wav")

        # Step 1: Build censored audio (concat gaps + replacements)
        ok, success_count, skipped_count = build_censored_audio(
            audio_path=audio_path,
            replacements=replacements,
            audio_dir=audio_dir,
            output_wav=censored_audio_wav,
            sample_rate=sample_rate,
            channels=channels,
            crossfade_ms=crossfade_ms,
        )
        if not ok:
            return False

        # Step 2: Mux original video + censored audio
        reencode = os.environ.get("VIDEO_REENCODE", "0") == "1"
        success = mux_video_audio(
            original_video=audio_path,
            censored_audio=censored_audio_wav,
            output_path=output_path,
            reencode_video=reencode,
        )

        if not success:
            return False

        log(f"[summary] {success_count} replacements spliced, "
            f"{skipped_count} skipped")
        return True

    finally:
        try:
            shutil.rmtree(temp_dir)
            log(f"[cleanup] Removed temp directory: {temp_dir}")
        except Exception as exc:
            log(f"[cleanup] Warning: Could not remove temp dir: {exc}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Splice generated replacement audio into the original video "
                    "using ffmpeg. The video stream is copied losslessly; only "
                    "the audio is censored.",
    )
    p.add_argument(
        "audio",
        help="Path to the original video file",
    )
    p.add_argument(
        "--replacements-json",
        required=True,
        help="Path to the filtered replacements JSON from find_replacements.py",
    )
    p.add_argument(
        "--audio-dir",
        required=True,
        help="Directory containing generated .wav files from generate_audio.py",
    )
    p.add_argument(
        "-o", "--output",
        required=True,
        help="Output video path (format inferred from extension).",
    )
    p.add_argument(
        "--crossfade-ms",
        type=int,
        default=CROSSFADE_MS,
        help=f"Fade duration in ms at segment boundaries (default: {CROSSFADE_MS})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without processing.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not Path(args.audio).is_file():
        log(f"[error] Original file not found: {args.audio}")
        return 2
    if not Path(args.replacements_json).is_file():
        log(f"[error] Replacements JSON not found: {args.replacements_json}")
        return 2
    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_dir():
        log(f"[warning] Audio directory not found: {audio_dir}")

    try:
        replacements = load_replacements(args.replacements_json)
    except (ValueError, json.JSONDecodeError) as exc:
        log(f"[error] {type(exc).__name__}: {exc}")
        return 1

    log(f"[input] Loaded {len(replacements)} replacement(s) from {args.replacements_json}")
    log(f"[input] Original file: {args.audio}")
    log(f"[input] Replacement .wav directory: {audio_dir}")
    log(f"[input] Crossfade: {args.crossfade_ms}ms fades at boundaries")
    log(f"[input] Output: {args.output}")

    try:
        success = splice_audio_ffmpeg(
            audio_path=args.audio,
            replacements=replacements,
            audio_dir=audio_dir,
            output_path=args.output,
            crossfade_ms=args.crossfade_ms,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        log("[abort] Interrupted by user.")
        return 130
    except Exception as exc:
        log(f"[error] {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())