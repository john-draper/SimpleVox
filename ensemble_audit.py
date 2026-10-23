#!/usr/bin/env python
r"""
Ensemble auditor: full-coverage multi-detector profanity sweep.

Policy (owner, 2026-10-08): the pipeline must catch everything it can by
itself - no region is exempt for human review, and no single model is
trusted (each sanitizes differently: Whisper deletes/euphemizes, Parakeet
mishears fuck->heck/freak and drops music-occluded speech).

Every episode is chunked (CHUNK_S + OVERLAP_S) and each chunk is listened to
by ALL detectors:

  * whisper-medium + VERBATIM_PROMPT (best single catcher; the prompt tips
    it toward verbatim) - word timestamps from faster-whisper
  * NVIDIA Parakeet, a different architecture with an independent bias:
      - parakeet-tdt-0.6b-v3 (preferred; native word timestamps, punctuation,
        needs transformers >= 5.9)
      - parakeet-ctc-0.6b (fallback; CTC frame alignment, no punctuation)

Union of profane tokens is written per episode as Season N/<stem>_ens.json:
[{"word","start","end","detector"}]. Downstream post-processing (cluster
merge, applied-window filter, dictionary replacement) is a separate step.

`detect_media()` is also used by run.py's optional --second-ear stage, which
appends the second ear's finds to the primary transcript.

Usage:
  python ensemble_audit.py "E:\Videos\TVShows\Rick and Morty (2013)" "E:\...\RnM_ens\_intermediate" [--limit N] [--detectors whisper,parakeet]
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from transcribe import VERBATIM_PROMPT
from replacements import find_matches

# Multi-scale validated 2026-10-08 on known-raw spots: no single window size
# catches everything (20s caught the credits chant; 12s caught clear-dialogue
# fucks; 8s caught neither; tight ~4s gap-clips catch what all chunk sizes
# miss - the gap audit covers those). Sweep both scales and union.
CHUNK_SIZES = [12.0, 20.0]
OVERLAP_S = 3.0  # 3s: unlucky chunk boundaries hid known catches at 1.5s
MIN_PROB = 0.4


def ffprobe_duration(path: Path) -> float:
    import subprocess

    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, timeout=60)
    return float(r.stdout.strip())


def extract_clip(media: Path, t0: float, t1: float) -> Path:
    import subprocess

    tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tf.close()
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}",
         "-i", str(media), "-vn", "-ac", "1", "-ar", "16000", tf.name],
        check=True)
    return Path(tf.name)


class WhisperEar:
    """faster-whisper, prompted. Gives word timestamps."""

    name = "whisper"

    def __init__(self, model: str = "medium"):
        from faster_whisper import WhisperModel
        self.m = WhisperModel(model, device="cuda", compute_type="float16")

    def hear(self, wav: Path, offset: float) -> list[dict]:
        segments, _ = self.m.transcribe(str(wav), language="en", beam_size=5,
                                        initial_prompt=VERBATIM_PROMPT,
                                        word_timestamps=True, vad_filter=False)
        out = []
        for seg in segments:
            for w in seg.words or []:
                if w.probability is not None and w.probability < MIN_PROB:
                    continue
                tok = w.word.strip()
                if tok:
                    out.append({"word": tok, "start": offset + w.start,
                                "end": offset + w.end})
        return out


class ParakeetCTCEar:
    """Parakeet-CTC, unprompted (its value IS its different bias).
    Token timestamps from CTC frame alignment (stride = chunk_len / frames)."""

    name = "parakeet"

    def __init__(self, model_id: str = "nvidia/parakeet-ctc-0.6b"):
        import contextlib
        import io

        import torch
        # transformers 4.57 prints an emoji "Config not found" banner while
        # building this class's docstring; swallow it, it is noise.
        with contextlib.redirect_stdout(io.StringIO()):
            from transformers import AutoProcessor, ParakeetForCTC
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = ParakeetForCTC.from_pretrained(
            model_id, dtype=torch.bfloat16).cuda().eval()
        self.blank = self.model.config.pad_token_id

    def hear(self, wav: Path, offset: float) -> list[dict]:
        import soundfile as sf
        data, sr = sf.read(str(wav))
        inputs = self.proc(data, sampling_rate=16000,
                           return_tensors="pt").to("cuda")
        with self.torch.no_grad():
            logits = self.model(inputs.input_features.to(self.torch.bfloat16)).logits[0]
        ids = logits.argmax(dim=-1).tolist()
        if not ids:
            return []
        stride = (len(data) / 16000.0) / len(ids)
        # CTC collapse over frames, keeping frame spans for timing
        tokens = []  # (id, f_start, f_end)
        prev, f0 = ids[0], 0
        for f in range(1, len(ids) + 1):
            if f == len(ids) or ids[f] != prev:
                if prev != self.blank:
                    tokens.append((prev, f0, f - 1))
                f0 = f
                if f < len(ids):
                    prev = ids[f]
        # sentencepiece pieces -> words: pieces starting with '▁' begin a word.
        # convert_ids_to_tokens keeps the '▁' marker; tokenizer.decode() strips
        # it, which would glue a whole chunk into one giant "word".
        words = []
        cur, fs, fe = [], None, None
        for tid, a, b in tokens:
            piece = self.proc.tokenizer.convert_ids_to_tokens(tid)
            if isinstance(piece, bytes):
                piece = piece.decode("utf-8", "replace")
            if piece is None:
                continue
            if piece.startswith("<") and piece.endswith(">"):
                continue  # <unk>/<blank>/<pad> -- not speech
            starts_word = piece.startswith("\u2581") or piece.startswith(" ")
            text = piece.replace("\u2581", "").strip()
            if not text:
                continue
            if starts_word and cur:
                words.append(("".join(cur), fs, fe))
                cur, fs = [], None
            if fs is None:
                fs = a
            cur.append(text)
            fe = b
        if cur:
            words.append(("".join(cur), fs, fe))
        return [{"word": w, "start": offset + a * stride,
                 "end": offset + (b + 1) * stride} for w, a, b in words]


class ParakeetTDTEar:
    """Parakeet-TDT: same family, newer decoder, native word timestamps.

    Preferred over CTC when the installed transformers has AutoModelForTDT
    (parakeet TDT support landed in transformers 5.9). Timestamps come from
    the TDT decoder's durations, so no frame-stride arithmetic is needed.
    """

    name = "parakeet"

    def __init__(self, model_id: str = "nvidia/parakeet-tdt-0.6b-v3"):
        import contextlib
        import io

        import torch
        with contextlib.redirect_stdout(io.StringIO()):
            from transformers import AutoModelForTDT, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForTDT.from_pretrained(
            model_id, dtype=torch.bfloat16).cuda().eval()

    def _decode(self, out):
        return self.proc.decode(out.sequences,
                                durations=getattr(out, "durations", None),
                                skip_special_tokens=True)

    @staticmethod
    def _merge(tokens):
        """Merge token-level (text, start, end) into word-level spans."""
        words = []
        cur, ws, we = [], None, None
        for text, start, end in tokens:
            piece = (text or "").replace("\u2581", " ").strip()
            if not piece:
                continue
            starts_word = (text or "").startswith(("\u2581", " "))
            if starts_word and cur:
                words.append(("".join(cur), ws, we))
                cur, ws = [], None
            if ws is None:
                ws = start
            cur.append(piece)
            we = end
        if cur:
            words.append(("".join(cur), ws, we))
        return words

    def hear(self, wav: Path, offset: float) -> list[dict]:
        import soundfile as sf
        data, sr = sf.read(str(wav))
        inputs = self.proc(data, sampling_rate=16000, return_tensors="pt").to("cuda")
        inputs = {k: (v.to(self.torch.bfloat16) if v.is_floating_point() else v)
                  for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model.generate(**inputs, return_dict_in_generate=True)

        decoded = self._decode(out)
        if isinstance(decoded, tuple):
            _text, stamps = decoded
        else:
            stamps = None
        if not stamps:
            return []

        tokens = []
        for item in stamps:
            if isinstance(item, dict):
                text = item.get("text") or item.get("token") or ""
                start, end = item.get("start"), item.get("end")
            else:  # object with attributes
                text = getattr(item, "text", "") or getattr(item, "token", "")
                start, end = getattr(item, "start", None), getattr(item, "end", None)
            if start is None or end is None:
                continue
            tokens.append((text, float(start), float(end)))

        return [{"word": w, "start": offset + s, "end": offset + e}
                for w, s, e in self._merge(tokens)]


def load_second_ear(prefer: str = "auto"):
    """Load the second ear, preferring TDT and falling back to CTC.

    Returns a cached instance so batch runs load the model once.
    """
    global _EAR
    if _EAR is not None:
        return _EAR

    if prefer in ("auto", "tdt"):
        try:
            _EAR = ParakeetTDTEar()
            print(f"[ear] {_EAR.name} (TDT) loaded", flush=True)
            return _EAR
        except Exception as exc:
            if prefer == "tdt":
                raise
            print(f"[ear] TDT unavailable ({type(exc).__name__}: {exc}); "
                  f"falling back to Parakeet-CTC", flush=True)

    _EAR = ParakeetCTCEar()
    print(f"[ear] {_EAR.name} (CTC) loaded", flush=True)
    return _EAR


_EAR = None


def profane(entry: dict) -> bool:
    return bool(find_matches([entry]))


def dedupe(hits: list[dict]) -> list[dict]:
    """Same word heard twice (chunk overlap / two detectors) within 1s -> one."""
    hits.sort(key=lambda h: (h["start"], 0 if h["detector"] == "whisper" else 1))
    out = []
    for h in hits:
        if any(h["word"].lower().strip(".,!?") == k["word"].lower().strip(".,!?")
               and abs(h["start"] - k["start"]) < 1.0 for k in out):
            continue
        out.append(h)
    return sorted(out, key=lambda h: h["start"])


def detect_media(media: Path, ears, chunk_sizes=None, progress: bool = False) -> list[dict]:
    """Run every ear over every chunk of `media`; return deduped profane hits."""
    chunk_sizes = chunk_sizes or CHUNK_SIZES
    dur = ffprobe_duration(media)
    hits: list[dict] = []
    for chunk_s in chunk_sizes:
        t0 = 0.0
        while t0 < dur - 0.5:
            t1 = min(t0 + chunk_s, dur)
            wav = extract_clip(media, t0, t1)
            try:
                for ear in ears:
                    for w in ear.hear(wav, offset=t0):
                        if profane(w):
                            hits.append({**w, "detector": ear.name})
            finally:
                wav.unlink(missing_ok=True)
            t0 += chunk_s - OVERLAP_S
    return dedupe(hits)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("media_root")
    ap.add_argument("out_root")
    ap.add_argument("--detectors", default="whisper,parakeet")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--grid-offset", choices=["zero", "half"], default="zero",
                    help="half: start each chunk grid at half a chunk - catches "
                         "words the zero grid suppresses at unlucky boundaries "
                         "(validated: the 1.5s-overlap grid caught E10 10:00-10:04 "
                         "that the 3s grid missed on identical audio)")
    ap.add_argument("--merge", action="store_true",
                    help="merge into existing <stem>_ens.json instead of skipping")
    ap.add_argument("--seasons", default="all")
    args = ap.parse_args(argv)

    media_root = Path(args.media_root)
    out_root = Path(args.out_root)
    ears = []
    for name in args.detectors.split(","):
        name = name.strip()
        if name == "whisper":
            ears.append(WhisperEar())
        else:
            ears.append(load_second_ear())
        print(f"[ear] {ears[-1].name} loaded", flush=True)

    seasons = None if args.seasons == "all" else {f"Season {s}" for s in args.seasons.split(",")}
    videos = sorted(p for p in media_root.rglob("*")
                    if p.suffix.lower() in (".mp4", ".mkv"))
    if seasons:
        videos = [v for v in videos if v.parent.name in seasons]
    if args.limit:
        videos = videos[:args.limit]

    t_sweep = time.time()
    for i, media in enumerate(videos, 1):
        rel_season = media.parent.name
        dst = out_root / rel_season / f"{media.stem}_ens.json"
        if dst.is_file() and not args.merge:  # resume support
            continue
        prior = []
        if dst.is_file() and args.merge:
            prior = json.loads(dst.read_text(encoding="utf-8"))
        hits = list(prior) + detect_media(media, ears)
        hits = dedupe(hits)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(hits, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"[{i}/{len(videos)}] {rel_season}/{media.stem}: "
              f"{len(hits)} profane token(s) "
              f"({time.time()-t_sweep:.0f}s elapsed)", flush=True)
    print(f"[ensemble done] {len(videos)} episode(s) in {(time.time()-t_sweep)/60:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
