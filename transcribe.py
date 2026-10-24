#!/usr/bin/env python
"""
Transcribe a video (or audio) file with faster-whisper and export word-level
timestamps to JSON.

One model, one pass: faster-whisper produces word-level timestamps itself
(DTW over the decoder's cross-attention), so there is no separate forced-
alignment stage -- no WhisperX, no wav2vec2, no per-language aligner download.
SimpleVox also runs no speaker diarization: every replacement word is spoken by
the same generic voice, so speaker labels are unnecessary.

ffmpeg reads the audio track directly out of any common video container (.mkv,
.mp4, .m4v, .mov, .avi, .webm, ...); only the audio is decoded, the video
stream is ignored here.

Output JSON is a list of objects:
    {"word": str, "start": float, "end": float}

Word spans are padded symmetrically by --word-pad-ms (default 50ms) and clamped
so they never overlap a neighbour. The pad covers the boundary imprecision of
native word timestamps (forced alignment was tighter): a little of the
neighbouring audio is inside the censored window, which is safe, because the
replacement always fills exactly the window it is given and the timeline does
not move.

Designed for an NVIDIA CUDA GPU. Falls back to CPU automatically if CUDA is
unavailable.

Example:
    python transcribe.py "input/movie.mkv"
    python transcribe.py input.mkv -o output/words.json --model large-v3
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Default model. `small` (244M) is fast but misses profanity; `large-v3`
# catches materially more and still runs comfortably on a modern GPU.
DEFAULT_MODEL = "large-v3"
DEFAULT_WORD_PAD_MS = 50

# Passed as faster-whisper's initial_prompt. Whisper has a politeness bias: it
# sometimes DELETES short profane utterances ("What the fuck?" at S02E07 09:35
# never made it into one pass's transcript) or swaps in a euphemism ("heck",
# "f***ing"). The prompt tells it to transcribe verbatim instead.
VERBATIM_PROMPT = (
    "The following is a verbatim transcript of an uncensored adult animated "
    "comedy. Profanity is transcribed exactly as spoken: fuck, fucking, shit, "
    "bitch, asshole."
)


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


def decode_audio_16k(audio_path: str | Path) -> "object":
    """Decode the source's audio track to 16kHz mono float32 (ffmpeg -> numpy).

    Replaces whisperx.load_audio: same output (mono 16k float32 in [-1, 1]),
    one ffmpeg call, no extra dependency.
    """
    import numpy as np

    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(audio_path),
        "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", "16000",
        "-f", "s16le", "-acodec", "pcm_s16le",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to decode '{audio_path}': "
            f"{result.stderr.decode('utf-8', 'replace')[:300]}"
        )
    return np.frombuffer(result.stdout, dtype=np.int16).astype("float32") / 32768.0


def load_model(model_name: str, device: str, compute_type: str = "auto"):
    """Load a faster-whisper model (the only ASR model SimpleVox needs)."""
    from faster_whisper import WhisperModel

    if device == "cuda" and compute_type == "auto":
        compute_type = "float16"
    elif device == "cpu" and compute_type == "auto":
        compute_type = "int8"

    t0 = time.time()
    log(f"[model] Loading faster-whisper '{model_name}' on {device} "
        f"(compute_type={compute_type})...")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    log(f"[model] Loaded in {time.time() - t0:.1f}s")
    return model


def rescan_gaps(words: list[dict], audio, model, sr: int = 16000,
                min_gap: float = 0.7, max_gap: float = 15.0) -> list[dict]:
    """Re-transcribe short silent gaps between words and recover dropped words.

    Whisper's VAD drops a word entirely at some chunk boundaries: the word is
    never in the transcript, so the dictionary can never censor it. This pass
    looks at every pause between consecutive words that still contains speech
    energy, transcribes just that window without VAD, and adds any profanity
    the first pass missed.

    Only dictionary-matching words are added, so music/ambience in a gap can
    never inject arbitrary text into the transcript.
    """
    import numpy as np
    from replacements import find_matches

    if not words:
        return words

    frame = 0.02
    n = int(sr * frame)
    usable = (len(audio) // n) * n
    if usable < n * 10:
        return words
    frames = audio[:usable].reshape(-1, n).astype("float32")
    rms = np.sqrt((frames ** 2).mean(axis=1))

    idx = []
    for w in words:
        a, b = int(w["start"] / frame), int(w["end"] / frame)
        idx.extend(range(max(0, a), min(len(rms), b)))
    ref = float(np.median(rms[idx])) if idx else 0.0
    thresh = max(ref * 0.35, 1e-4)

    bounds = [(0.0, words[0]["start"])]
    bounds += [(words[i]["end"], words[i + 1]["start"]) for i in range(len(words) - 1)]
    bounds += [(words[-1]["end"], len(audio) / sr)]

    # VAD can drop a long stretch, not just a single word (seen: 51 s in an
    # Outer Banks episode). Split any gap longer than max_gap into overlapping
    # windows so those stretches are re-listened to as well.
    spans: list[tuple[float, float]] = []
    for g0, g1 in bounds:
        if (g1 - g0) < min_gap:
            continue
        if (g1 - g0) <= max_gap:
            spans.append((g0, g1))
            continue
        win, step = 18.0, 16.0
        t = g0
        while t < g1 - min_gap:
            spans.append((t, min(t + win, g1)))
            t += step

    added: list[dict] = []
    checked = 0
    for g0, g1 in spans[:60]:
        seg = rms[int(g0 / frame):int(g1 / frame)]
        if seg.size == 0 or int((seg > thresh).sum()) < 6:  # <120 ms of sound
            continue
        checked += 1
        off = max(0.0, g0 - 0.20)
        end = min(len(audio) / sr, g1 + 0.20)
        clip = audio[int(off * sr):int(end * sr)]
        try:
            segs, _ = model.transcribe(
                clip, language="en", vad_filter=False, word_timestamps=True,
                condition_on_previous_text=False, beam_size=5,
            )
            found = []
            for s in segs:
                for w in (s.words or []):
                    if float(getattr(w, "probability", 1.0)) < 0.45:
                        continue
                    found.append({"word": (w.word or "").strip(),
                                  "start": off + float(w.start),
                                  "end": off + float(w.end)})
        except Exception:
            continue
        known = words + added
        for m in find_matches(found):
            if any(m["start"] < k["end"] - 0.05 and m["end"] > k["start"] + 0.05
                   for k in known):
                continue
            added.append({"word": m["word"], "start": m["start"], "end": m["end"]})

    if added:
        log(f"[rescan] Recovered {len(added)} missed profane word(s) from "
            f"{checked} gap(s): "
            + ", ".join(f"{a['word']}@{a['start']:.1f}s" for a in added[:8]))
        words = sorted(words + added, key=lambda w: w["start"])
        # The same utterance can appear in both passes with slightly different
        # timings (e.g. "FUCK" 720.9-721.3 and "Fuck" 721.34-721.58). Collapse
        # adjacent words with the same stem so one word is never censored twice.
        import string

        def stem(w):
            return w.strip().strip(string.punctuation).lower()

        merged = []
        for w in words:
            if (merged and stem(merged[-1]["word"]) == stem(w["word"])
                    and (w["start"] - merged[-1]["end"]) < 0.15):
                merged[-1]["end"] = max(merged[-1]["end"], w["end"])
                continue
            merged.append(dict(w))
        words = merged
    else:
        log(f"[rescan] Checked {checked} gap(s); nothing missed.")
    return words


def pad_words(words: list[dict], pad_ms: int) -> list[dict]:
    """Widen each word span by pad_ms/2 on both sides, never overlapping.

    Native word timestamps are looser than forced alignment, and this branch
    splices exact word spans (no per-family lead-in). The pad buys back the
    onset/offset precision that alignment used to provide. Widening a window
    does not move the timeline: the replacement is still stretched to fill
    exactly the window it is given.
    """
    if pad_ms <= 0 or not words:
        return words
    half = pad_ms / 2000.0
    out: list[dict] = []
    for i, w in enumerate(words):
        start = max(0.0, float(w["start"]) - half)
        end = float(w["end"]) + half
        if out:
            start = max(start, out[-1]["end"])
        if i + 1 < len(words):
            end = min(end, float(words[i + 1]["start"]))
        out.append({"word": w["word"], "start": start,
                    "end": max(start + 0.01, end)})
    return out


def transcribe_words(model, audio, language: str | None = None,
                     vad: bool = True,
                     initial_prompt: str | None = None) -> tuple[list[dict], str]:
    """Run one faster-whisper pass and return (words, detected_language)."""
    t0 = time.time()
    log(f"[transcribe] Running faster-whisper (VAD {'on' if vad else 'off'})...")
    segments, info = model.transcribe(
        audio,
        language=language,
        vad_filter=vad,
        condition_on_previous_text=False,
        beam_size=5,
        word_timestamps=True,
        initial_prompt=initial_prompt,
    )

    words: list[dict] = []
    skipped = 0
    for s in segments:
        for w in (s.words or []):
            start, end = w.start, w.end
            if start is None or end is None:
                skipped += 1
                continue
            token = (w.word or "").strip()
            if not token:
                continue
            words.append({"word": token, "start": float(start),
                          "end": float(end)})

    detected = info.language or language or "en"
    log(f"[transcribe] Done in {time.time() - t0:.1f}s "
        f"(language: {detected}, {len(words)} words)")
    if skipped:
        log(f"[words] Skipped {skipped} token(s) with missing timestamps.")
    return words, detected


def transcribe_audio(
    audio_path: str,
    language: str | None,
    model_name: str,
    compute_type: str,
    device: str | None,
    output_path: str | None,
    word_pad_ms: int = DEFAULT_WORD_PAD_MS,
    model=None,
) -> list[dict]:
    """Transcribe audio and return a list of word dicts with timestamps.

    Pass a preloaded `model` to reuse it across files (batch mode); otherwise
    one is loaded from `model_name`.
    """
    if device is None:
        device = select_device()

    audio_file = Path(audio_path)
    if not audio_file.is_file():
        raise FileNotFoundError(f"Input file not found: {audio_file}")

    log(f"[input] {audio_file}")

    if model is None:
        model = load_model(model_name, device, compute_type)

    t0 = time.time()
    log("[audio] Decoding audio track with ffmpeg "
        "(this may take a moment for long files)...")
    audio = decode_audio_16k(audio_file)
    log(f"[audio] Decoded in {time.time() - t0:.1f}s "
        f"({audio.shape[0] / 16000 / 60:.1f} min of audio)")

    words, _lang = transcribe_words(model, audio, language=language)

    # Recover words dropped at VAD boundaries (whole-word misses).
    words = rescan_gaps(words, audio, model)

    # Widen spans so native timestamps don't leak word onsets/offsets.
    words = pad_words(words, word_pad_ms)

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
        description="Transcribe a video/audio file with faster-whisper and "
                    "export word-level JSON timestamps (no alignment stage).",
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
        default=DEFAULT_MODEL,
        help=f"faster-whisper model name (default: {DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--word-pad-ms",
        type=int,
        default=DEFAULT_WORD_PAD_MS,
        help=f"Widen each word span by this much (default: {DEFAULT_WORD_PAD_MS}ms) "
             "to cover native-timestamp imprecision. 0 disables.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Deprecated / ignored (kept so old command lines still parse).",
    )
    p.add_argument(
        "--compute-type",
        default="auto",
        help="ctranslate2 compute type (default: auto -> float16 on CUDA, "
             "int8 on CPU). Try 'int8' to cut VRAM usage.",
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
            compute_type=args.compute_type,
            device=args.device,
            output_path=args.output,
            word_pad_ms=args.word_pad_ms,
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
