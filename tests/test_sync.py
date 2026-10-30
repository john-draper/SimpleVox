#!/usr/bin/env python
"""A/V sync regression test for splice_audio.py.

Builds a small MKV with a sharp 2 kHz click every 4 s, censors 8 windows that
avoid the clicks, then re-splices the OUTPUT several times (what SimpleVox does
across passes). Each generation must keep every click at its original time.

Why this exists: cutting segments out of a compressed source (AAC) loses one
codec frame (~21 ms) at the first seek. That is invisible in one pass but
compounds on every re-splice (-130 ms after six), which is how the Rick and
Morty library drifted. splice_audio.py now cuts from a PCM copy; this test is
the guard rail. Run it after ANY change to splice_audio.py.

Usage:
    python tests/test_sync.py            # 3 generations, exit 0 = pass
    python tests/test_sync.py --gens 6   # longer soak
"""
import argparse
import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import splice_audio  # noqa: E402

SR = 48000
DUR = 40.0
CLICKS = [4.0 * i for i in range(1, 10)]  # 4..36 s
TOLERANCE_MS = 5.0


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def build_source(tmp: Path) -> Path:
    n = int(DUR * SR)
    x = np.zeros(n, dtype=np.float32)
    for t in CLICKS:
        i0 = int(t * SR)
        i1 = i0 + int(0.03 * SR)
        tt = np.arange(i1 - i0) / SR
        x[i0:i1] += (0.7 * np.sin(2 * np.pi * 2000 * tt) *
                     np.hanning(i1 - i0)).astype(np.float32)
    wav = tmp / "click.wav"
    sf.write(wav, np.stack([x, x], axis=1), SR, subtype="PCM_16")
    out = tmp / "source.mkv"
    run(["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc2=size=320x180:rate=24:duration={DUR}",
         "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "ultrafast", "-c:a", "aac", "-b:a", "128k", "-shortest",
         str(out)])
    return out


def build_replacements(tmp: Path):
    reps, audio_dir = [], tmp / "reps"
    audio_dir.mkdir()
    for k in range(8):
        start = 1.5 + 4.3 * k          # never lands on a click (clicks are every 4.0s)
        end = start + 0.30             # keep the tail-out clear of the 32.0s click
        s_ms, e_ms = int(start * 1000), int(end * 1000)
        name = splice_audio.format_filename(s_ms, e_ms, "freak")
        ns = int(0.45 * 24000)
        tt = np.arange(ns) / 24000
        y = (0.5 * np.sin(2 * np.pi * 500 * tt) *
             np.minimum(1, np.minimum(tt / 0.01, (0.45 - tt) / 0.01))).astype(np.float32)
        sf.write(audio_dir / name, y, 24000, subtype="PCM_16")
        reps.append({"word": "fuck", "start": start, "end": end,
                     "replacement": "freak"})
    return reps, audio_dir


def decode_mono(path: Path, out: Path):
    run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vn",
         "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(out)])


def click_onsets(wav: Path, thresh: float = 0.05):
    x, sr = sf.read(str(wav), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    win = int(0.01 * sr)
    frames = x[: (len(x) // win) * win].reshape(-1, win)
    spec = np.abs(np.fft.rfft(frames * np.hanning(win), axis=1))
    freqs = np.fft.rfftfreq(win, 1 / sr)
    energy = spec[:, (freqs > 1800) & (freqs < 2200)].sum(axis=1)
    energy /= energy.max() + 1e-9
    on = energy > thresh
    out, i = [], 0
    while i < len(on):
        if on[i]:
            out.append(i * win / sr)
            i += 20
        else:
            i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=3)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="simplevox_sync_"))
    try:
        source = build_source(tmp)
        reps, audio_dir = build_replacements(tmp)

        decode_mono(source, tmp / "ref.wav")
        base = click_onsets(tmp / "ref.wav")
        if len(base) != len(CLICKS):
            print(f"FAIL: expected {len(CLICKS)} clicks, detected {len(base)}")
            return 1

        current = source
        worst = 0.0
        for gen in range(1, args.gens + 1):
            nxt = tmp / f"gen{gen}.mkv"
            with contextlib.redirect_stdout(io.StringIO()):
                ok = splice_audio.splice_audio_ffmpeg(
                    audio_path=str(current), replacements=reps, audio_dir=audio_dir,
                    output_path=str(nxt), lead_in_ms=0)
            if not ok:
                print(f"FAIL: splice failed at generation {gen}")
                return 1
            decode_mono(nxt, tmp / "cur.wav")
            got = click_onsets(tmp / "cur.wav")
            if len(got) != len(base):
                print(f"FAIL: gen{gen} detected {len(got)} clicks, expected {len(base)}")
                return 1
            drift = [(g - b) * 1000 for b, g in zip(base, got)]
            worst_gen = max(abs(d) for d in drift)
            worst = max(worst, worst_gen)
            print(f"  gen{gen}: first={drift[0]:+.1f}ms last={drift[-1]:+.1f}ms "
                  f"max|drift|={worst_gen:.1f}ms")
            if worst_gen > TOLERANCE_MS:
                print(f"FAIL: gen{gen} exceeds {TOLERANCE_MS}ms tolerance")
                return 1
            current = nxt

        print(f"PASS: {args.gens} generation(s), worst drift {worst:.1f}ms "
              f"(tolerance {TOLERANCE_MS}ms)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
