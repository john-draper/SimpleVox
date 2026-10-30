#!/usr/bin/env python
"""Pitch-preservation regression test for the euphemism fit.

A replacement has to fill the original word's spoken span, so it is
time-stretched to an exact target duration. That stretch must not change the
voice's pitch: resampling a 150 Hz tone to 60% of its length must still be
150 Hz, not 250 Hz.

Run after any change to convert_replacement_wav()/_pitch_preserving_fit():

    python tests/test_pitch.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import splice_audio  # noqa: E402

SR = 48000
HZ = 150.0


def write_tone(path: Path, secs: float = 0.6, hz: float = HZ) -> None:
    t = np.arange(int(secs * SR)) / SR
    x = (0.4 * np.sin(2 * np.pi * hz * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(x.tobytes())


def read_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    x = x.reshape(-1, ch).astype(np.float32).mean(axis=1) / 32768.0
    assert sr == SR, f"unexpected sample rate {sr}"
    return x


def est_f0(x: np.ndarray) -> float:
    a, b = int(0.10 * SR), int(0.25 * SR)
    seg = x[a:b] - x[a:b].mean()
    c = np.correlate(seg, seg, "full")[len(seg) - 1:]
    lo, hi = int(SR / 400), int(SR / 70)
    return SR / (lo + int(np.argmax(c[lo:hi])))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="simplevox_pitch_"))
    try:
        tone = tmp / "tone.wav"
        write_tone(tone)
        base = read_mono(tone)
        f0_in = est_f0(base)
        print(f"input: {len(base) / SR:.3f}s, {f0_in:.1f} Hz")

        if os.name == "nt":
            assert getattr(subprocess.Popen, "_simplevox_hidden", False), \
                "subprocess.Popen was not patched by _nowindow"

        worst = 0.0
        for target in (0.30, 0.45, 0.90):
            out = tmp / f"out_{int(target * 1000)}.wav"
            splice_audio.convert_replacement_wav(
                str(tone), str(out), SR, 1, fade_ms=0,
                target_duration_s=target, gain_db=0.0)
            x = read_mono(out)
            dur = len(x) / SR
            f0 = est_f0(x)
            df = abs(f0 - f0_in)
            worst = max(worst, df)
            print(f"target {target:.2f}s: dur={dur:.3f}s f0={f0:.1f}Hz "
                  f"(delta {f0 - f0_in:+.1f}Hz)")
            if abs(dur - target) > 0.005:
                print(f"FAIL: duration {dur:.3f}s != {target:.3f}s")
                return 1
            if df > 4.0:
                print(f"FAIL: pitch moved {df:.1f}Hz (resampled instead of stretched)")
                return 1

        print(f"PASS: duration exact, worst pitch drift {worst:.1f}Hz")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
