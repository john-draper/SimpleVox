# SimpleVox

🌐 **[View the website →](https://john-draper.github.io/SimpleVox/)**

**Simple Vox**ice censor — automatically detect profanity in a video file and replace each profane word with a clean euphemism spoken by a single, generic male voice. The video stream is preserved losslessly; only the audio track is edited and re-written.

The final output is a video file with identical picture quality and a censored soundtrack.

SimpleVox uses a **single generic male voice** for every replacement word, powered by Microsoft Edge TTS (free neural TTS). This focused design keeps the tool:

- **Simple** — a streamlined 4-stage pipeline with one TTS voice.
- **Free** — no API keys, no HuggingFace tokens, no self-hosted servers.
- **Fast** — transcription runs directly on the source audio with no extra preprocessing.

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

Drop video files into the `input/` folder. Subfolders are supported and their
structure is **mirrored** into the output directory:

```
input/
├── movie.mkv
└── Season 1/
    └── episode 01.mkv
```

### Run

```bash
# Process the whole input/ folder (recursive):
python run.py

# Single file:
python run.py "input/movie.mkv"

# A whole folder (recursive):
python run.py "input/Season 1"

# Choose a different generic voice:
python run.py "input/movie.mkv" --voice en-US-DavisNeural

# Use a more accurate (slower) Whisper model:
python run.py "input/movie.mkv" --model large-v3

# Skip videos that already have an output file:
python run.py --skip-existing

# Keep going even if one video fails (batch mode):
python run.py --continue-on-error

# Windows launcher (any of the above work; pass-through):
run.bat
run.bat "input\Season 1"
run.bat "input\American Dad! S17E11.mkv" --voice en-US-DavisNeural
```

The censored video is saved to `output/` with the **same filename** as the
input and the **same relative folder structure**. Intermediate files
(transcription JSON, replacements JSON, generated WAVs) are stored under
`output/_intermediate/<same subpath>/`.

```
output/
├── movie.mkv                                  ← censored output
├── Season 1/
│   └── episode 01.mkv                          ← censored output
└── _intermediate/
    ├── movie.json
    ├── movie_replacements.json
    └── Season 1/
        └── episode 01.json
```

> **Note on special characters:** Filenames containing spaces, `!`, `&`, and
> other special characters (e.g. `American Dad! S17E11.mkv`) are handled
> correctly. All discovery happens in Python to avoid shell-quoting pitfalls.

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