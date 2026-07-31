# SimpleVox

**Simple Vox**ice censor — automatically detect profanity in a video file and replace each profane word with a clean euphemism spoken by a single, generic male voice. The video stream is preserved losslessly; only the audio track is edited and re-written.

The final output is a video file with identical picture quality and a censored soundtrack.

SimpleVox is a deliberately **simplified** version of [Revox](#relationship-to-revox). It drops voice cloning, speaker diarization, and vocal-stem separation. Every replacement word is spoken by the **same** generic, high-quality male neural voice (Microsoft Edge TTS). This makes it:

- **Simple** — 4 stages instead of 6, one TTS voice, no model juggling.
- **Free** — no API keys, no HuggingFace tokens, no self-hosted servers.
- **Fast** — skips the expensive Demucs and pyannote stages entirely.

```
video.mkv ──▶ [1. Transcribe] ──▶ words.json      (WhisperX word timestamps)
                    │
                    ▼
            [2. Find Replacements] ──▶ replacements.json   (profane -> clean)
                    │
                    ▼
            [3. Generate Audio] ──▶ *.wav   (one generic male voice, all words)
                    │
                    ▼
            [4. Splice + Mux] ──▶ video_censored.mkv
               (censored audio + ORIGINAL video stream, copied losslessly)
```

## Relationship to Revox

SimpleVox is a trimmed-down fork of Revox. The following Revox features are
**intentionally removed**:

| Revox feature | SimpleVox |
|---|---|
| Demucs vocal-stem separation | ❌ Removed — transcribe the original audio directly |
| pyannote speaker diarization | ❌ Removed — no per-character speaker labels |
| Per-speaker voice cloning (Fish Speech / ElevenLabs / Pocket-TTS) | ❌ Removed — one generic voice for all words |
| pyttsx3 (Windows SAPI5) fallback | ❌ Removed — edge-tts is the sole provider |
| GUI | ❌ Removed — CLI only |

What remains is the proven core: WhisperX transcription, the profanity
dictionary, and the ffmpeg-based audio splice + lossless video mux.

## Quick Start

### Prerequisites

1. **Python 3.10+** on PATH
2. **ffmpeg** on PATH — `winget install Gyan.FFmpeg` (Windows) or `brew install ffmpeg` (macOS)
3. **NVIDIA GPU** (recommended) — the pipeline falls back to CPU but it's much slower

> **No HuggingFace token required.** SimpleVox doesn't use diarization.

### Installation

```bash
# 1. Install PyTorch for your CUDA version (GPU only)
#    For CUDA 11.8:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
#    For other versions: https://pytorch.org/get-started/locally/

# 2. Install the remaining dependencies
pip install -r requirements.txt
```

### Configure (optional)

SimpleVox works with no configuration. To tune transcription/muxing, copy
`.env.example` to `.env` and edit it:

```bash
copy .env.example .env        (Windows)
cp .env.example .env          (macOS/Linux)
```

### Place Your Videos

Drop video files into the `input/` folder:

```
input/movie.mkv
```

### Run

```bash
# Single file:
python run.py "input/movie.mkv"

# Choose a different generic voice:
python run.py "input/movie.mkv" --voice en-US-DavisNeural

# Use a more accurate (slower) Whisper model:
python run.py "input/movie.mkv" --model large-v3

# Windows launcher (processes input/ if no argument given):
run.bat "input\movie.mkv"
```

The censored video is saved to `output/` — same video quality, censored audio.

---

## Pipeline Stages

### Stage 1: Transcription (`transcribe.py`)

Transcribes the audio track of the video with WhisperX for word-level timestamps. **No speaker diarization** is run (SimpleVox uses one voice for everything, so speaker labels aren't needed).

```bash
python transcribe.py "input/video.mkv" -o output/words.json
```

| Option | Default | Notes |
|--------|---------|-------|
| `-o PATH` | `<input>.json` | Output JSON path |
| `--model NAME` | `small` | Whisper model |
| `--device D` | auto | `cuda` or `cpu` |
| `--compute-type T` | auto | `float16`/`int8` |
| `--language CODE` | `en` | language code, or `auto` |

### Stage 2: Profanity Filtering (`replacements.py`)

Matches transcribed words against the replacement dictionary in `replacements.py`.

```bash
# As a module (run.py calls this directly). The dictionary is editable:
#   REPLACEMENTS = { "bitch": "brat", "fuck": "freak", ... }
```

Edit the `REPLACEMENTS` dictionary in `replacements.py` to customize.

### Stage 3: Audio Generation (`generate_audio.py`)

Generates replacement `.wav` files with the **single generic male voice** (`en-US-GuyNeural` by default). All words use the same voice.

```bash
python generate_audio.py output/movie_replacements.json --output-dir output/generated_audio
```

| Option | Default | Notes |
|--------|---------|-------|
| `--output-dir PATH` | `output/generated_audio` | Output directory |
| `--voice NAME` | `en-US-GuyNeural` | edge-tts voice |
| `--skip-existing` | off | Don't regenerate existing files |

### Stage 4: Audio Splice + Video Mux (`splice_audio.py`)

Builds the censored audio track (each replacement trimmed/stretched/loudness-matched to exactly fill the original word's slot), then muxes it with the **original video stream** (copied losslessly).

```bash
python splice_audio.py "input/video.mkv" \
    --replacements-json output/movie_replacements.json \
    --audio-dir output/generated_audio \
    --output output/video_censored.mkv
```

If ffmpeg can't stream-copy the video, set `VIDEO_REENCODE=1` to re-encode.

---

## Choosing a Voice

SimpleVox uses [edge-tts](https://github.com/rany2/edge-tts) (Microsoft's free
neural TTS). The default is a neutral US male voice. Some alternatives:

| Voice | Style |
|-------|-------|
| `en-US-GuyNeural` *(default)* | Conversational US male |
| `en-US-DavisNeural` | Warm US male |
| `en-GB-RyanNeural` | British male |
| `en-AU-WilliamNeural` | Australian male |

Pass any of these with `--voice`.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu` |
| `WHISPER_MODEL` | `small` | Whisper model size |
| `VIDEO_REENCODE` | `0` | Set to `1` to re-encode video |

---

## Troubleshooting

- **Audio out of sync** — SimpleVox time-stretches each replacement to exactly match the original word's duration. If you still see drift, ensure you're using the latest `splice_audio.py`.
- **CUDA crashes / cuDNN errors** — Install cuDNN 8 DLLs (`pip install nvidia-cudnn-cu11==8.9.4.25`) and copy them to your `torch/lib/` folder. Or fall back to CPU with `--device cpu`.
- **FFmpeg not found** — `winget install Gyan.FFmpeg`, then reopen your terminal.
- **edge-tts returns tiny/empty audio** — This is usually a transient network issue; rerun. The stage skips already-generated files with `--skip-existing`.

## License

MIT.