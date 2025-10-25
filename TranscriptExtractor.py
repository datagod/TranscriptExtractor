"""
YouTube Audio Transcription with Speaker Diarization
==================================================

Purpose:
--------
This script downloads audio from a YouTube video, converts it to WAV format, performs speaker diarization to identify speakers, transcribes the audio using the Whisper model, aligns the transcript with speaker labels, and saves the result to a text file. The script tracks total execution time, supports GPU acceleration for transcription and diarization, provides verbose output for debugging, and uses unique filenames based on the video title to avoid overwriting files. It also checks for existing video, MP3, and WAV files to avoid redundant work.

Features:
---------
- Downloads audio from a YouTube URL using yt-dlp, retaining the original video file.
- Checks for existing video and MP3 files to avoid redundant downloads and extractions when --skip-download is enabled.
- Converts MP3 audio to WAV format using pydub, skipping if a valid WAV exists.
- Performs speaker diarization using pyannote.audio to identify distinct speakers, with debug output for segment lengths and GPU support.
- Transcribes audio using OpenAI's Whisper model with timestamped segments, using GPU if available.
- Aligns transcript segments with speaker labels based on temporal overlap.
- Saves the transcript to a unique text file named after the video title, with timestamps or incrementing numbers to prevent overwriting.
- Tracks and reports total script execution time in seconds.
- Uses GPU (CUDA) for both Whisper transcription and pyannote.audio diarization if available, with fallback to CPU and clear explanations if not detected.
- Provides verbose output for each step with clean prefixes for progress and [ERROR] for errors.
- Performs CUDA checks at script start for immediate GPU troubleshooting feedback.

Dependencies:
-------------
- Python 3.8+
- yt_dlp: For downloading YouTube audio and video.
- pydub: For MP3 to WAV conversion (requires ffmpeg installed).
- whisper: OpenAI's Whisper model for audio transcription.
- pyannote.audio: For speaker diarization (requires Hugging Face token).
- torch: For GPU acceleration and model support.
- argparse, os, sys, datetime, re, time: Standard Python libraries.

Installation:
-------------
1. Install dependencies:
   ```bash
   pip install yt_dlp pydub openai-whisper pyannote.audio torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   Replace `cu121` with your CUDA version (e.g., `cu118`) or omit for CPU-only.
2. Install ffmpeg for pydub:
   - On Windows: Download from ffmpeg.org and add to PATH.
   - On Linux: `sudo apt-get install ffmpeg`
   - On macOS: `brew install ffmpeg`
3. Set Hugging Face token for pyannote.audio:
   ```bash
   export HF_TOKEN=your_huggingface_token
   ```
   Obtain a token from https://huggingface.co/settings/tokens.

Usage:
------
```bash
python YoutubeTranscript.py "https://www.youtube.com/watch?v=video_id" [--skip-download]
```
- `url`: The YouTube URL to process.
- `--skip-download`: Optional flag to skip downloading if both the video and MP3 files already exist.
- Output files (video, MP3, WAV, transcript) are named using the sanitized video title and saved in the script's working directory.

Environment Requirements:
------------------------
- NVIDIA GPU with CUDA support (optional, for faster transcription and diarization).
- CUDA-enabled PyTorch installation for GPU acceleration.
- Sufficient disk space for video, MP3, WAV, and transcript files.
- Internet connection for downloading YouTube videos and Hugging Face models.

File Handling:
--------------
- **Input**: YouTube URL.
- **Output Files**:
  - Video file: `<sanitized_title>.<ext>` (e.g., `.m4a`, `.webm`), retained after download.
  - MP3 file: `<sanitized_title>.mp3`.
  - WAV file: `<sanitized_title>.wav`.
  - Transcript: `<sanitized_title>_<timestamp>.txt` or `<sanitized_title>_<timestamp>_<counter>.txt`.
- Files are saved in the script's working directory.
- Cleanup is disabled for debugging; manually delete temporary files if needed.

GPU Usage:
----------
- The script uses CUDA for Whisper transcription and pyannote.audio diarization if an NVIDIA GPU is available.
- CUDA availability is checked at script start with `torch.cuda.is_available()`.
- Detailed debugging output includes CUDA device count, names, properties, and clear reasons for CPU fallback (e.g., no GPU, drivers not installed, PyTorch not CUDA-enabled).
- Falls back to CPU if CUDA is unavailable or improperly configured.

Notes:
------
- Ensure your GPU drivers and CUDA toolkit are up-to-date for GPU support.
- Use a smaller Whisper model (e.g., `tiny` instead of `base`) for faster processing or if GPU memory is limited.
- The script skips cleanup for debugging; check video, MP3, WAV, and transcript files manually.
- For GPU issues, verify PyTorch CUDA installation with:
  ```python
  import torch
  print(torch.cuda.is_available())
  print(torch.cuda.get_device_name(0))
  ```
- If --skip-download is enabled but video or MP3 files are missing, the script will exit with an error.
- Execution time is reported at the end of the script in seconds.
- Debug output for diarization segments helps diagnose issues with short or empty segments.

Author: William McEvoy + Grok
Last Updated: October 25, 2025

TranscriptExtractor v2.3
- Local audio reuse before any network work.
- New: --audio PATH to use an existing .mp3/.wav/.m4a and skip yt-dlp entirely.
- Smarter auto-detection: if a matching audio file already exists in --outdir (or the per-video dir),
  reuse it and skip download/conversion.
- Keeps v2.2 improvements (Windows TorchCodec bypass via in-memory audio for pyannote).
"""


print("")
print("=========================================")
print("= TRANSCRIPT EXTRACTOR                  =")
print("=========================================")
print("")



import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import yt_dlp
from pydub import AudioSegment

import torch
import whisper

# Optional deps
try:
    import torchaudio
    _HAS_TA = True
except Exception:
    _HAS_TA = False

# Optional import: only if diarization is enabled
try:
    from pyannote.audio import Pipeline
    _HAS_PYANNOTE = True
except Exception:
    _HAS_PYANNOTE = False


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)

def err(msg: str) -> None:
    print(f"[{_now()}] [ERROR] {msg}", file=sys.stderr, flush=True)

def sanitize_filename(text: str, maxlen: int = 160) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:maxlen] if text else "untitled"

def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        err("Requested --device cuda but CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hhmmss(seconds: float) -> str:
    td = timedelta(seconds=int(max(0, seconds)))
    hours, remainder = divmod(int(td.total_seconds()), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:01d}:{minutes:02d}:{secs:02d}"

# ---------- Audio preload to avoid TorchCodec ----------
def load_waveform(path: Path) -> Tuple[torch.Tensor, int]:
    """
    Return (waveform[C,T], sample_rate). Prefer torchaudio; fallback to pydub.
    Output dtype float32 in [-1, 1].
    """
    if _HAS_TA:
        wav, sr = torchaudio.load(str(path))  # shape [C, T], dtype float32/float64
        if wav.dtype != torch.float32:
            wav = wav.float()
        return wav, int(sr)
    # Fallback via pydub -> numpy -> torch
    seg = AudioSegment.from_file(path)
    sr = seg.frame_rate
    samples = seg.get_array_of_samples()
    import numpy as np
    arr = np.array(samples).astype("float32")
    channels = seg.channels
    if channels > 1:
        arr = arr.reshape((-1, channels)).T  # [C, T]
    else:
        arr = arr.reshape((1, -1))
    max_val = float(1 << (8 * seg.sample_width - 1))
    arr = arr / max_val
    wav = torch.from_numpy(arr)
    return wav, int(sr)

def _load_pyannote_pipeline(model_id: str, token: Optional[str], device: torch.device):
    if token is None:
        return Pipeline.from_pretrained(model_id).to(device)
    last_err = None
    for kw in ("token", "use_auth_token"):
        try:
            log(f"Attempting pyannote Pipeline.from_pretrained(..., {kw}=<redacted>)")
            pipe = Pipeline.from_pretrained(model_id, **{kw: token})
            return pipe.to(device)
        except TypeError as e:
            last_err = e
        except Exception as e:
            last_err = e
    raise TypeError(f"pyannote Pipeline token parameter mismatch; tried token/use_auth_token. Last error: {last_err}")

# ---------- Local audio resolution ----------
AUDIO_EXTS = (".mp3", ".m4a", ".wav", ".flac", ".ogg")

def _find_existing_audio(base_out: Path, title: str) -> Optional[Path]:
    sanitized = sanitize_filename(title)
    # Look inside outputs root and per-video dir
    candidates = [
        *((base_out / f"{title}{ext}") for ext in AUDIO_EXTS),
        *((base_out / f"{sanitized}{ext}") for ext in AUDIO_EXTS),
        *((base_out / sanitized / f"{title}{ext}") for ext in AUDIO_EXTS),
        *((base_out / sanitized / f"{sanitized}{ext}") for ext in AUDIO_EXTS),
    ]
    for p in candidates:
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            return p
    return None

def _probe_title(url: str) -> str:
    """Probe title without download. Minimal yt-dlp call."""
    ydl_opts = {"quiet": True, "noprogress": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if not info:
            raise RuntimeError("yt-dlp failed to retrieve video info.")
        return info.get("title") or "audio"

def resolve_audio_source(url: str, base_out: Path, explicit_audio: Optional[str]) -> Tuple[Path, str]:
    """
    Returns (audio_path, title). If explicit_audio is provided and exists, use it.
    Else, probe title and reuse any existing audio in base_out/per-video dir.
    If nothing exists, download via yt-dlp (audio-only) into base_out root.
    """
    base_out.mkdir(parents=True, exist_ok=True)

    # 1) Explicit override
    if explicit_audio:
        audio_path = Path(explicit_audio).expanduser().resolve()
        if not audio_path.exists() or not audio_path.is_file():
            raise FileNotFoundError(f"--audio points to a non-existent file: {audio_path}")
        title = audio_path.stem
        log(f"--audio specified, using local file: {audio_path.name}")
        return audio_path, title

    # 2) Probe title (no download) and try to find local file
    log("Probing video title (no download)...")
    title = _probe_title(url)
    existing = _find_existing_audio(base_out, title)
    if existing:
        log(f"Local audio found, skipping download: {existing}")
        return existing, title

    # 3) Download audio now (only if nothing found)
    log("No local audio found. Downloading audio (yt-dlp + ffmpeg)...")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(base_out / "%(title)s.%(ext)s"),
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ],
        "retries": 3,
        "ignoreerrors": False,
        "noprogress": True,
        "quiet": True,
        "no_warnings": True,
        "keepvideo": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if not info:
            raise RuntimeError("yt-dlp failed during download.")
        title = info.get("title") or title
        # Expect MP3 in base_out root
        mp3_path = Path(ydl.prepare_filename(info)).with_suffix(".mp3")
        if not mp3_path.exists():
            # Try m4a as fallback (sometimes postprocessor not triggered)
            m4a = Path(ydl.prepare_filename(info)).with_suffix(".m4a")
            if m4a.exists():
                mp3_path = m4a
        if not mp3_path.exists():
            raise FileNotFoundError("Expected audio file not created by yt-dlp.")
        log(f"Downloaded audio: {mp3_path.name}")
        return mp3_path, title

def ensure_wav(audio_path: Path, out_dir: Path, title: str) -> Path:
    sanitized = sanitize_filename(title)
    wav_path = out_dir / f"{sanitized}.wav"
    # If the audio_path is already a WAV matching target, reuse
    if audio_path.suffix.lower() == ".wav":
        if audio_path.resolve() != wav_path.resolve():
            # copy/rename into per-video dir if needed
            try:
                if not wav_path.exists():
                    audio = AudioSegment.from_file(audio_path)
                    audio.export(wav_path, format="wav")
            except Exception as e:
                err(f"WAV relocate/export failed: {e}")
        if wav_path.exists():
            log(f"Reusing existing WAV: {wav_path.name} ({len(AudioSegment.from_file(wav_path))/1000.0:.2f}s)")
            return wav_path

    if wav_path.exists():
        try:
            dur = len(AudioSegment.from_file(wav_path)) / 1000.0
            if dur >= 10:
                log(f"Reusing existing WAV: {wav_path.name} ({dur:.2f}s)")
                return wav_path
        except Exception:
            pass

    log(f"Converting → WAV: {audio_path.name} → {wav_path.name}")
    audio = AudioSegment.from_file(audio_path)
    audio.export(wav_path, format="wav")
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        raise RuntimeError("WAV export failed or produced empty file.")
    return wav_path

def safe_move(src: Path, dst: Path):
    try:
        if src.resolve() == dst.resolve():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return  # ignore
        src.rename(dst)
    except Exception as e:
        err(f"Non-fatal move error: {e}")

def diarize(wav_path: Path, device: torch.device, hf_token: Optional[str], model_id: str, max_speakers: Optional[int]):
    if not _HAS_PYANNOTE:
        raise RuntimeError("pyannote.audio not installed; install it or use --no-diarization.")
    # Try to detect installed pyannote version (for v3 vs v4 output handling)
    try:
        import pyannote.audio as _pya
        _pya_ver = getattr(_pya, "__version__", "unknown")
    except Exception:
        _pya_ver = "unknown"

    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    log(f"Loading diarization pipeline '{model_id}' on {device} (pyannote.audio={_pya_ver}) ...")
    pipe = _load_pyannote_pipeline(model_id, token, device)

    # Preload audio to avoid TorchCodec/AudioDecoder path
    wav, sr = load_waveform(wav_path)
    wav = wav.to(device) if device.type == "cuda" else wav
    inputs = {"waveform": wav, "sample_rate": sr}

    # Run pipeline
    output = pipe(inputs) if max_speakers is None else pipe(inputs, num_speakers=max_speakers)

    # Normalize output across pyannote versions
    annotation = None
    # pyannote >=4 returns a DiarizeOutput with .speaker_diarization / .exclusive_speaker_diarization
    if hasattr(output, "speaker_diarization"):
        annotation = output.speaker_diarization
    # pyannote 3.x returns an Annotation directly
    elif hasattr(output, "itertracks"):
        annotation = output
    # sometimes dict-like
    elif isinstance(output, dict) and "speaker_diarization" in output:
        annotation = output["speaker_diarization"]
    else:
        raise TypeError(f"Unexpected pyannote output type: {type(output)}; upgrade script or pin pyannote to ~=3.1")

    segments: List[Dict[str, Any]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        start, end = float(turn.start), float(turn.end)
        if end <= start:
            continue
        segments.append({"start": start, "end": end, "speaker": str(speaker)})
    segments.sort(key=lambda s: (s["start"], s["end"]))
    uniq = len({s["speaker"] for s in segments})
    log(f"Diarization complete: {len(segments)} segments, ~{uniq} speaker labels.")
    return segments

def transcribe(wav_path: Path, device: torch.device, model_name: str, language: Optional[str]):
    log(f"Loading Whisper model '{model_name}' on {device} ...")
    model = whisper.load_model(model_name, device=str(device))
    result = model.transcribe(str(wav_path), verbose=False, language=language)
    segs = result.get("segments") or []
    log(f"Transcription complete: {len(segs)} segments.")
    return segs

def align(transcript: List[Dict[str, Any]], diar: List[Dict[str, Any]]):
    if not diar:
        return [
            {"start": float(t["start"]), "end": float(t["end"]), "speaker": "Speaker_00", "text": t["text"].strip()}
            for t in transcript
            if t.get("text", "").strip()
        ]

    aligned: List[Dict[str, Any]] = []
    for t in transcript:
        text = (t.get("text") or "").strip()
        if not text:
            continue
        t_start, t_end = float(t["start"]), float(t["end"])
        best = "Speaker_00"
        max_overlap = 0.0
        for d in diar:
            a0, a1 = t_start, t_end
            b0, b1 = float(d["start"]), float(d["end"])
            overlap = max(0.0, min(a1, b1) - max(a0, b0))
            if overlap > max_overlap:
                max_overlap = overlap
                best = d["speaker"]
        aligned.append({"start": t_start, "end": t_end, "speaker": best, "text": text})
    return aligned

def write_txt(aligned: List[Dict[str, Any]], out_path: Path) -> Path:
    with out_path.open("w", encoding="utf-8") as f:
        for item in aligned:
            f.write(f"[{hhmmss(item['start'])}] {item['speaker']}: {item['text']}\n")
    return out_path

def write_srt(aligned: List[Dict[str, Any]], out_path: Path) -> Path:
    def fmt_srt_time(t: float) -> str:
        t = max(0.0, t)
        ms = int((t - int(t)) * 1000)
        td = timedelta(seconds=int(t))
        hours, rem = divmod(int(td.total_seconds()), 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
    with out_path.open("w", encoding="utf-8") as f:
        for i, item in enumerate(aligned, 1):
            f.write(f"{i}\n")
            f.write(f"{fmt_srt_time(item['start'])} --> {fmt_srt_time(item['end'])}\n")
            f.write(f"{item['speaker']}: {item['text']}\n\n")
    return out_path

def write_json(aligned: List[Dict[str, Any]], out_path: Path, meta: Dict[str, Any]) -> Path:
    payload = {"meta": meta, "segments": aligned}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path

def main() -> int:
    parser = argparse.ArgumentParser(description="YouTube → Diarization → Whisper transcript with speaker labels.")
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("--audio", default=None, help="Path to a local audio file (.mp3/.wav/.m4a). Skips download.")
    parser.add_argument("--skip-download", action="store_true", help="(Deprecated by --audio/local-detect) Skip download if audio exists.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device.")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size (tiny, base, small, medium, large).")
    parser.add_argument("--language", default=None, help="Language hint for Whisper (e.g., en, fr).")
    parser.add_argument("--no-diarization", action="store_true", help="Disable diarization (all text Speaker_00).")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token (or set HF_TOKEN/HUGGINGFACE_TOKEN).")
    parser.add_argument("--diarization-model", default="pyannote/speaker-diarization-3.1", help="pyannote pipeline id.")
    parser.add_argument("--max-speakers", type=int, default=None, help="Hint for number of speakers (if supported).")
    parser.add_argument("--output", nargs="+", choices=["txt", "srt", "json"], default=["txt"], help="Output formats.")
    parser.add_argument("--outdir", default="outputs", help="Base output directory.")
    args = parser.parse_args()

    start_ts = time.time()
    device = pick_device(args.device)
    log(f"Device: {device} (CUDA available: {torch.cuda.is_available()})")

    base_out = Path(args.outdir).resolve()
    # Resolve audio source first (local reuse if possible)
    audio_path, title = resolve_audio_source(args.url, base_out, args.audio)

    # Per-video directory using sanitized title
    video_dir = base_out / sanitize_filename(title)
    video_dir.mkdir(parents=True, exist_ok=True)

    # If the audio isn't already under the per-video dir, keep it where it is; we'll export WAV there.
    wav_path = ensure_wav(audio_path, video_dir, title)

    # Diarization (optional)
    diar_segments: List[Dict[str, Any]] = []
    if args.no_diarization:
        log("Diarization disabled by flag.")
    else:
        if not _HAS_PYANNOTE:
            err("pyannote.audio not available; continuing without diarization (use --no-diarization to silence this).")
        else:
            diar_segments = diarize(wav_path, device, args.hf_token, args.diarization_model, args.max_speakers)

    # Transcribe
    trans_segments = transcribe(wav_path, device, args.whisper_model, args.language)

    # Align
    aligned = align(trans_segments, diar_segments)

    # Write outputs
    meta = {
        "title": title,
        "url": args.url,
        "created": _now(),
        "device": str(device),
        "whisper_model": args.whisper_model,
        "language": args.language,
        "diarization_model": None if args.no_diarization else args.diarization_model,
        "max_speakers": args.max_speakers,
        "cuda": torch.cuda.is_available(),
    }

    written: List[Path] = []
    stem = sanitize_filename(title)
    if "txt" in args.output:
        written.append(write_txt(aligned, video_dir / f"{stem}.txt"))
    if "srt" in args.output:
        written.append(write_srt(aligned, video_dir / f"{stem}.srt"))
    if "json" in args.output:
        written.append(write_json(aligned, video_dir / f"{stem}.json", meta))

    elapsed = time.time() - start_ts
    log("Done.")
    log(f"Wrote: {', '.join(p.name for p in written)}")
    log(f"Total time: {elapsed:.2f} sec")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        err("Interrupted.")
        sys.exit(130)
    except SystemExit as se:
        raise
    except Exception as e:
        err(str(e))
        sys.exit(1)
