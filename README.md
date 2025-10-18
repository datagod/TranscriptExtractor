# TranscriptExtractor

![TranscriptExtractor Logo](https://via.placeholder.com/150?text=TranscriptExtractor)

**YouTube Audio Transcription with Speaker Diarization**

This open-source Python script downloads audio from a YouTube video, converts it to WAV format, performs speaker diarization to identify different speakers, transcribes the audio using OpenAI's Whisper model, aligns the transcript with speaker labels, and saves the result to a text file. It supports GPU acceleration for faster processing, skips redundant steps if files already exist, and provides detailed verbose output for debugging.

The script is designed for researchers, podcasters, journalists, or anyone needing accurate transcripts from YouTube videos with speaker identification.

**Author:** William McEvoy + Grok  
**Repository:** <https://github.com/datagod/TranscriptExtractor>  
**Last Updated:** October 16, 2025  
**License:** MIT

## Features

-   **YouTube Audio Download:** Uses `yt-dlp` to download audio (and retain the video file) in MP3 format.
-   **Skip Redundant Steps:** Checks for existing video, MP3, and WAV files to avoid unnecessary downloads or conversions (use `--skip-download` flag).
-   **Audio Conversion:** Converts MP3 to WAV using `pydub` for compatibility with transcription tools.
-   **Speaker Diarization:** Identifies distinct speakers using `pyannote.audio`, with support for GPU acceleration.
-   **Audio Transcription:** Transcribes audio with timestamps using OpenAI's Whisper model (default: "base" model; GPU-enabled if available).
-   **Alignment:** Matches transcript segments to speakers based on time overlaps.
-   **Output Management:** Saves transcripts to unique files based on the video title to prevent overwriting.
-   **Performance Tracking:** Reports total execution time and provides verbose debugging output.
-   **GPU Support:** Automatically uses CUDA for Whisper and `pyannote` if an NVIDIA GPU is detected; falls back to CPU with clear error messages.
-   **Error Handling:** Includes checks for file existence, short audio segments, and CUDA availability.

## Requirements

-   **Python Version:** 3.8 or higher
-   **Operating System:** Windows, Linux, or macOS
-   **Hardware (Optional):** NVIDIA GPU with CUDA support for faster processing
-   **Internet Connection:** Required for downloading YouTube videos and Hugging Face models
-   **Disk Space:** Sufficient space for video, MP3, WAV, and transcript files (video files can be large)
-   **Hugging Face Token:** Required for `pyannote.audio` (free; obtain from [huggingface.co](https://huggingface.co/settings/tokens))

## Installation

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/datagod/TranscriptExtractor.git
cd TranscriptExtractor
```

### 2. Install Dependencies

Install the required Python packages. Use a virtual environment for best practices (e.g., via `venv` or `conda`).

```bash
pip install yt_dlp pydub openai-whisper pyannote.audio torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

-   Replace `cu121` with your CUDA version (e.g., `cu118`) if using a GPU. Omit the `--index-url` for CPU-only installation.
-   **Note:** If you encounter issues with `torch`, verify your CUDA version and PyTorch compatibility at [pytorch.org](https://pytorch.org/get-started/locally/).

### 3. Install FFmpeg

FFmpeg is required for audio processing with `pydub`.

-   **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html), extract, and add the `bin` folder to your system PATH.
-   **Linux:** Run `sudo apt-get install ffmpeg` (Ubuntu/Debian) or equivalent for your distro.
-   **macOS:** Run `brew install ffmpeg`
