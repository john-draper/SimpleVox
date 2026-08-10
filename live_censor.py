#!/usr/bin/env python
"""
LiveVox — real-time profanity mute for live audio (e.g. a game's audio
routed through a virtual cable, or any system audio via WASAPI loopback).

Unlike SimpleVox (which edits audio FILES offline), LiveVox runs as a live
audio processor: it captures audio in real time, buffers it (broadcast delay),
runs streaming faster-whisper transcription, detects profane words, and MUTES
the delayed output during each profane word (a "broadcast bleep"). Optionally
overlays a beep or a TTS euphemism in the muted window.

Why a delay is unavoidable: to mute a word BEFORE it reaches your ears, the
audio must be held back long enough for ASR to process it and decide to mute.
This is the broadcast-delay principle; there is no way around it for a system
that cannot see the future.

Safety: this is fully EXTERNAL. It does not inject into any process, access
memory, or modify game files. It only reads from an audio input device and
writes to an audio output device. For a game, you route the game's audio to a
virtual cable (e.g. VB-Audio Virtual Cable), and LiveVox reads from that cable
and writes to your real speakers. The game is unaware.

Two modes:
  --simulate FILE   Run the ASR+mute core against an audio FILE (deterministic,
                    no real-time I/O). Used for testing. Prints detected
                    profanity and the mute windows that would be applied.
  (default)         Real-time mode: capture from --input-device, buffer by
                    --delay, play to --output-device, muting profane windows.

Usage:
  # Test the core logic on a file (no audio I/O):
  python live_censor.py --simulate profane.wav

  # Real-time: list devices, then run
  python live_censor.py --list-devices
  python live_censor.py --input-device 5 --output-device 7 --delay 2.0
  python live_censor.py --input-device "CABLE Output" --beep
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Reuse SimpleVox's profanity dictionary + matcher.
from replacements import find_matches, clean_word, REPLACEMENTS

PROFANE_KEYS = set(REPLACEMENTS.keys())

SAMPLE_RATE = 16000  # Whisper wants 16kHz mono


# --------------------------------------------------------------------------- #
# Profanity detection (word-level, reuses replacements.py)
# --------------------------------------------------------------------------- #

def detected_profane_words(segments) -> list[dict]:
    """Given faster-whisper segments (with word timestamps), return profane hits.

    Each hit: {"word","start","end","replacement"}.
    """
    hits = []
    for seg in segments:
        for w in (seg.words or []):
            key = clean_word(w.word)
            if key in PROFANE_KEYS:
                hits.append({
                    "word": w.word,
                    "start": float(w.start),
                    "end": float(w.end),
                    "replacement": REPLACEMENTS[key],
                })
    return hits


# --------------------------------------------------------------------------- #
# Mode 1: simulate (file-based core test, deterministic, no audio I/O)
# --------------------------------------------------------------------------- #

def run_simulate(audio_path: str, model_name: str, device: str,
                 compute_type: str, language: str, beep: bool) -> int:
    """Transcribe a file with faster-whisper and report profanity + mute windows."""
    from faster_whisper import WhisperModel

    print(f"[simulate] Loading faster-whisper '{model_name}' on {device} "
          f"({compute_type})...")
    t0 = time.time()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    print(f"[simulate] Model loaded in {time.time() - t0:.1f}s")

    print(f"[simulate] Transcribing: {audio_path}")
    t0 = time.time()
    segments_gen, info = model.transcribe(
        audio_path, language=language, word_timestamps=True, vad_filter=True,
    )
    segments = list(segments_gen)
    print(f"[simulate] Transcribed in {time.time() - t0:.1f}s "
          f"({info.duration:.1f}s audio, language={info.language})")

    hits = detected_profane_words(segments)
    print()
    print("=" * 70)
    if not hits:
        print("  NO PROFANITY DETECTED")
        print("=" * 70)
        return 0

    print(f"  {len(hits)} PROFANE WORD(S) DETECTED — would mute these windows:")
    print("=" * 70)
    total_mute = 0.0
    for h in hits:
        dur = h["end"] - h["start"]
        total_mute += dur
        action = "BEEP" if beep else "MUTE"
        print(f"  {h['start']:7.2f}s - {h['end']:7.2f}s  ({dur:.2f}s)  "
              f"{h['word']!r:>10} -> {h['replacement']!r:<10}  [{action}]")
    print("-" * 70)
    print(f"  Total audio muted: {total_mute:.2f}s of {info.duration:.1f}s "
          f"({100 * total_mute / info.duration:.1f}%)")
    print()
    print("  (In real-time mode these windows would be silenced in the "
          "delayed output.)")
    return 0


# --------------------------------------------------------------------------- #
# Mode 2: device discovery + real-time capture/delay/mute/playback
# --------------------------------------------------------------------------- #

def list_devices() -> int:
    import sounddevice as sd
    print("Audio devices (use the index or name substring with --input/--output):")
    print()
    for i, d in enumerate(sd.query_devices()):
        api = ""
        try:
            api = sd.query_hostapis(d["hostapi"])["name"]
        except Exception:
            pass
        inw = "IN " if d["max_input_channels"] > 0 else "   "
        outw = "OUT" if d["max_output_channels"] > 0 else "   "
        print(f"  [{i:2d}] {inw} {outw}  {d['name'][:50]:<50} ({api})")
    print()
    print("WASAPI devices support loopback. To capture a virtual cable, pick its")
    print("Output device as the LiveVox INPUT.")
    return 0


def find_device(name_or_index: str, want_input: bool) -> int:
    import sounddevice as sd
    devs = sd.query_devices()
    try:
        return int(name_or_index)
    except ValueError:
        pass
    cap_key = "max_input_channels" if want_input else "max_output_channels"
    matches = [i for i, d in enumerate(devs)
               if name_or_index.lower() in d["name"].lower() and d[cap_key] > 0]
    if not matches:
        matches = [i for i, d in enumerate(devs)
                   if name_or_index.lower() in d["name"].lower()]
    if not matches:
        raise SystemExit(f"No device matching {name_or_index!r}")
    return matches[0]


def run_realtime(input_dev, output_dev, delay: float, model_name: str,
                 device: str, compute_type: str, language: str,
                 beep: bool, chunk_s: float) -> int:
    """Real-time capture -> delay -> faster-whisper -> mute -> playback."""
    import sounddevice as sd
    from faster_whisper import WhisperModel
    import threading
    import collections

    print(f"[realtime] Loading faster-whisper '{model_name}' on {device}...")
    t0 = time.time()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    print(f"[realtime] Model loaded in {time.time() - t0:.1f}s")

    in_idx = find_device(input_dev, want_input=True)
    out_idx = find_device(output_dev, want_input=False)
    print(f"[realtime] Input device:  [{in_idx}] {sd.query_devices(in_idx)['name']}")
    print(f"[realtime] Output device: [{out_idx}] {sd.query_devices(out_idx)['name']}")
    print(f"[realtime] Delay buffer:  {delay:.2f}s")
    print(f"[realtime] Mute mode:     {'BEEP' if beep else 'MUTE'}")
    print()

    # --- Shared state ---
    delay_samples = max(1, int(delay * SAMPLE_RATE))
    state = {
        "buffer": np.zeros(delay_samples, dtype=np.float32),  # delay line
        "write_pos": 0,
        "asr_chunk": collections.deque(),  # samples queued for ASR
        "mute_until": 0.0,  # mute output while time.monotonic() < this
        "lock": threading.Lock(),
        "stats": {"muted_words": 0, "chunks": 0},
    }
    chunk_samples = int(chunk_s * SAMPLE_RATE)

    def asr_worker():
        """Background thread: pulls chunks, transcribes, sets mute windows."""
        accum = []
        while True:
            try:
                blk = state["asr_chunk"].popleft()
            except IndexError:
                time.sleep(0.01)
                continue
            accum.append(blk)
            if sum(len(a) for a in accum) >= chunk_samples:
                audio = np.concatenate(accum)
                accum = []
                state["stats"]["chunks"] += 1
                try:
                    segs, _info = model.transcribe(
                        audio, language=language, word_timestamps=True,
                        vad_filter=False,
                    )
                    segs = list(segs)
                except Exception as exc:
                    print(f"[asr] transcribe error: {exc}", flush=True)
                    continue
                hits = detected_profane_words(segs)
                for h in hits:
                    # The word was spoken ~now in the live stream. The delay
                    # line holds it back by `delay` seconds, so muting now
                    # silences the (slightly future) playback of that word.
                    word_dur = max(0.15, h["end"] - h["start"])
                    with state["lock"]:
                        state["mute_until"] = max(
                            state["mute_until"], time.monotonic() + word_dur
                        )
                    state["stats"]["muted_words"] += 1
                    print(f"[mute] {h['word']!r} -> {h['replacement']!r}  "
                          f"muting {word_dur:.2f}s", flush=True)

    threading.Thread(target=asr_worker, daemon=True).start()

    blocksize = 1024

    def callback(indata, outdata, frames, time_info, status):
        in_mono = indata[:, 0] if indata.ndim > 1 else indata
        state["asr_chunk"].append(in_mono.copy())
        n = len(in_mono)
        out = np.empty(n, dtype=np.float32)
        wp = state["write_pos"]
        for i in range(n):
            out[i] = state["buffer"][wp]
            state["buffer"][wp] = in_mono[i]
            wp = (wp + 1) % delay_samples
        state["write_pos"] = wp
        with state["lock"]:
            muted = time.monotonic() < state["mute_until"]
        if muted:
            if beep:
                t = np.arange(n) / SAMPLE_RATE
                out = 0.3 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
            else:
                out[:] = 0.0
        outdata[:, 0] = out

    print("[realtime] Streaming. Press Ctrl+C to stop.")
    print("[realtime] (Make sure the source/game outputs to the input device.)")
    try:
        with sd.Stream(
            samplerate=SAMPLE_RATE, blocksize=blocksize,
            dtype="float32", channels=1,
            device=(in_idx, out_idx), callback=callback,
        ):
            while True:
                time.sleep(1.0)
                print(f"[stats] chunks={state['stats']['chunks']} "
                      f"muted_words={state['stats']['muted_words']}", flush=True)
    except KeyboardInterrupt:
        print("\n[realtime] Stopped.")
        print(f"[stats] chunks={state['stats']['chunks']} "
              f"muted_words={state['stats']['muted_words']}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="LiveVox: real-time profanity mute for live audio. "
                    "Captures audio, buffers it (broadcast delay), runs "
                    "faster-whisper, and mutes profane words in the delayed "
                    "output. Fully external (no process injection).")
    p.add_argument("--simulate", metavar="FILE",
                   help="Test mode: run ASR+mute core on an audio FILE (no I/O).")
    p.add_argument("--list-devices", action="store_true",
                   help="List audio devices and exit.")
    p.add_argument("--input-device", default=None,
                   help="Input device index or name (real-time mode).")
    p.add_argument("--output-device", default=None,
                   help="Output device index or name (real-time mode).")
    p.add_argument("--delay", type=float, default=2.0,
                   help="Delay/buffer seconds (default 2.0). Higher = safer "
                        "mute but more audio lag.")
    p.add_argument("--model", default="tiny",
                   help="faster-whisper model (default tiny for low latency).")
    p.add_argument("--device", default="cuda",
                   help="cuda or cpu (default cuda).")
    p.add_argument("--compute-type", default="float16",
                   help="float16/int8 (default float16 on cuda).")
    p.add_argument("--language", default="en", help="Language code (default en).")
    p.add_argument("--beep", action="store_true",
                   help="Overlay a 1kHz beep during mute (default: silence).")
    p.add_argument("--chunk", type=float, default=1.0,
                   help="ASR chunk size in seconds (default 1.0).")
    args = p.parse_args(argv)

    if args.list_devices:
        return list_devices()
    if args.simulate:
        return run_simulate(args.simulate, args.model, args.device,
                            args.compute_type, args.language, args.beep)
    if not args.input_device or not args.output_device:
        p.error("real-time mode requires --input-device and --output-device "
                "(use --list-devices to see them), or use --simulate FILE.")
    return run_realtime(args.input_device, args.output_device, args.delay,
                        args.model, args.device, args.compute_type,
                        args.language, args.beep, args.chunk)


if __name__ == "__main__":
    sys.exit(main())
