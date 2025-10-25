# TranscriptExtractor

YouTube audio transcription with speaker diarization and speaker-labeled output.

>   This README documents how to install all dependencies, set up GPU acceleration, and run the script end to end. The details below are derived directly from the script you provided.

## What it does

-   Downloads audio from a YouTube URL with `yt-dlp` and keeps the original video file too
-   Converts MP3 to WAV using `pydub` and `ffmpeg`
-   Performs **speaker diarization** with `pyannote.audio`
-   Transcribes with **OpenAI Whisper**
-   Aligns each transcript segment to the most likely speaker by timestamp overlap
-   Writes a timestamped, speaker-labeled transcript text file
-   Detects and uses CUDA automatically if available, falls back to CPU otherwise
-   Prints detailed progress and errors, and reports total run time

## Requirements

-   **Python:** 3.8 or newer
-   **Disk space:** enough to store the downloaded video, MP3, WAV, and transcript
-   **Internet:** required for YouTube downloads and model pulls
-   **Optional GPU:** NVIDIA GPU with CUDA drivers for acceleration

## Quick start

```bash
# 1) Create and activate a clean environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install system ffmpeg
# Windows: download ffmpeg release zip and add its /bin to PATH (see instructions below)
# Debian/Ubuntu:
sudo apt-get update && sudo apt-get install -y ffmpeg
# macOS with Homebrew:
brew install ffmpeg

# 3) Install Python packages
# CPU only
pip install --upgrade pip
pip install yt_dlp pydub openai-whisper pyannote.audio torch torchvision torchaudio

# Or if you want CUDA acceleration, install a CUDA-enabled PyTorch build.
# Example for CUDA 12.1 wheels (adjust to your CUDA version as needed):
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install yt_dlp pydub openai-whisper pyannote.audio

# 4) Authenticate pyannote on Hugging Face
# Replace the token value with your own from https://huggingface.co/settings/tokens
# Windows PowerShell:
setx HF_TOKEN "hf_xxx_your_token"
# macOS/Linux for current shell:
export HF_TOKEN="hf_xxx_your_token"

# 5) Run
python TranscriptExtractor.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Detailed dependency setup

### 1) Python environment

Use a dedicated virtual environment to avoid version conflicts.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install --upgrade pip
```

### 2) ffmpeg

`pydub` calls `ffmpeg` for audio conversion. Install it system-wide so it is on `PATH`.

-   **Windows**
    -   Download a current build from the official site or a trusted provider
    -   Unzip to a permanent folder, e.g., `C:\ffmpeg`
    -   Add `C:\ffmpeg\bin` to your **User** or **System** `PATH`
    -   Open a new terminal and verify:

```powershell
ffmpeg -version
```

-   **Debian/Ubuntu**

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
ffmpeg -version
```

-   **macOS (Homebrew)**

```bash
brew install ffmpeg
ffmpeg -version
```

### 3) Python packages

The script imports:

-   `yt_dlp` for YouTube downloads
-   `pydub` for MP3 to WAV
-   `openai-whisper` for speech to text
-   `pyannote.audio` for speaker diarization
-   `torch` for CPU or GPU acceleration
-   Standard libs: `argparse`, `os`, `sys`, `datetime`, `re`, `time`

#### CPU-only install

```bash
pip install yt_dlp pydub openai-whisper pyannote.audio torch torchvision torchaudio
```

#### GPU-accelerated install

You need:

-   A supported NVIDIA GPU
-   Up-to-date NVIDIA drivers
-   A CUDA-enabled PyTorch build

Example for CUDA 12.1 wheels:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install yt_dlp pydub openai-whisper pyannote.audio
```

>   If you use a different CUDA runtime, replace `cu121` with the correct tag, or follow the selector at https://pytorch.org to get the exact command for your OS, Python, and CUDA.

Verify PyTorch sees your GPU:

```python
python - << "PY"
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

### 4) Hugging Face access token for pyannote.audio

`pyannote.audio` diarization pipeline requires an access token.

1.  Create a token at: https://huggingface.co/settings/tokens
2.  Make it available to the script as `HF_TOKEN`.

Set it once per machine:

-   **Windows PowerShell**

```powershell
setx HF_TOKEN "hf_xxx_your_token"
```

Open a new shell so the environment picks up the change.

-   **macOS/Linux (current shell only)**

```bash
export HF_TOKEN="hf_xxx_your_token"
```

You can also prefix when running:

```bash
HF_TOKEN="hf_xxx_your_token" python TranscriptExtractor.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Usage

```bash
python TranscriptExtractor.py "https://www.youtube.com/watch?v=VIDEO_ID" [--skip-download]
```

-   `url`  
    The YouTube video URL to process.
-   `--skip-download`  
    If provided, the script will **not** attempt a download and will instead use existing files if both the original video file and the derived MP3 already exist. If either is missing, the script exits with an error.

### What gets created

All files are written in the working directory with names derived from the sanitized video title:

-   Original **video** file: `<title>.<ext>` and is kept
-   **MP3**: `<title>.mp3`
-   **WAV**: `<title>.wav`
-   **Transcript**: `<title>_YYYYMMDD_HHMMSS.txt` or a suffixed variant to avoid collisions

### Output format

Transcript lines look like:

```
[HH:MM:SS] SPEAKER_XX: text...
```

Speakers are labeled from diarization output. Each Whisper segment is aligned to the speaker segment with the largest time overlap.

## CUDA behavior

At startup the script prints CUDA diagnostics:

-   Whether CUDA is available
-   Device count and names
-   Active device properties

If CUDA is not available, the script lists common reasons and continues on CPU.

## Examples

### Standard run

```bash
python TranscriptExtractor.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### Reusing existing downloads

```bash
python TranscriptExtractor.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --skip-download
```

>   This will only proceed if both the original video file and the MP3 already exist next to the script.

## Troubleshooting

-   `ffmpeg` **not found**  
    Ensure `ffmpeg` is installed and on `PATH`. Reopen your terminal after editing `PATH`.
-   `HF_TOKEN` **not set**  
    The diarization step will fail if `HF_TOKEN` is missing. Set it as shown above.
-   **PyTorch cannot see GPU**  
    Verify GPU drivers. Install a CUDA-enabled PyTorch build that matches your runtime. Check `torch.cuda.is_available()` returns `True`. Confirm you installed from the CUDA index-url and not the CPU default.
-   **Out of GPU memory**  
    Try a smaller Whisper model such as `tiny` or `base`. The script currently loads `base`. You can edit the call to `transcribe_audio(..., model_name="base")` to use a smaller model.
-   **Very short or empty segments**  
    The diarization loop logs and continues. This can occur with silence or abrupt cuts.
-   `--skip-download` **fails**  
    Both the original video file and the MP3 must exist. If either is missing, remove the flag or restore the missing file.

## Operational notes

-   Cleanup of intermediate files is intentionally disabled for debugging
-   Transcript filenames are made unique using a timestamp to avoid accidental overwrite
-   Filenames are sanitized to remove characters not allowed by the OS and to limit length

## Security and privacy

-   YouTube content is downloaded locally
-   Diarization requires an authenticated Hugging Face pipeline
-   Be mindful of where you store your `HF_TOKEN` and any generated transcripts



-   License: MIT
-   Author: William McEvoy + Grok
