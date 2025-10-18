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
Last Updated: October 16, 2025
"""
print("")
print("=========================================")
print("= TRANSCRIPT EXTRACTOR                  =")
print("=========================================")
print("")


import yt_dlp
import os
import re
from pydub import AudioSegment
import whisper
import sys
import argparse
from pyannote.audio import Pipeline
from datetime import timedelta, datetime
import torch
import time

# Perform CUDA checks at script start for immediate troubleshooting feedback

# Track script start time
start_time = time.time()

print("Starting CUDA availability check...")
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
if cuda_available:
    print("SUCCESS: CUDA detected and will be used for transcription and diarization.")
    cuda_device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices: {cuda_device_count}")
    for i in range(cuda_device_count):
        print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device properties: {torch.cuda.get_device_properties(torch.cuda.current_device())}")
    device = torch.device("cuda")
else:
    print("WARNING: CUDA not available. Falling back to CPU.")
    print("Clear reasons for no CUDA:")
    print("- No compatible NVIDIA GPU detected in the system.")
    print("- CUDA drivers not installed, outdated, or incompatible.")
    print("- PyTorch installed without CUDA support. Reinstall with CUDA: e.g., pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print("- Environment configuration issues (e.g., PATH, LD_LIBRARY_PATH not set correctly).")
    print("- Verify with: python -c 'import torch; print(torch.cuda.is_available())'")
    device = torch.device("cpu")
print(f"Using device: {device}")

def sanitize_filename(title):
    """Sanitize YouTube video title to create a valid filename."""
    print(f"Sanitizing filename: {title}")
    sanitized = re.sub(r'[<>:"/\\|?*]', '', title)
    sanitized = re.sub(r'\s+', '_', sanitized.strip())
    sanitized = sanitized[:200]
    print(f"Sanitized filename: {sanitized}")
    return sanitized

def download_youtube_audio(url, skip_download=False):
    """Download audio from a YouTube URL using yt-dlp, or skip if files exist."""
    print(f"\nStarting audio download for URL: {url}")
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio',
        'outtmpl': '%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'retries': 3,
        'keepvideo': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Extracting video info without download...")
            info = ydl.extract_info(url, download=False)
            original_filename = ydl.prepare_filename(info)
            mp3_filename = os.path.splitext(original_filename)[0] + '.mp3'
            video_title = info.get('title', 'audio')
            print(f"Video title: {video_title}")
            print(f"Expected video file: {original_filename}")
            print(f"Expected MP3 file: {mp3_filename}")
            print(f"Expected duration: {info.get('duration', 0)} seconds")

            if skip_download:
                if os.path.exists(original_filename) and os.path.exists(mp3_filename):
                    print(f"Skipping download: both video ({original_filename}) and MP3 ({mp3_filename}) exist")
                    return mp3_filename, video_title
                else:
                    missing = []
                    if not os.path.exists(original_filename):
                        missing.append(original_filename)
                    if not os.path.exists(mp3_filename):
                        missing.append(mp3_filename)
                    print(f"[ERROR] Missing files for skip-download: {', '.join(missing)}")
                    sys.exit(1)
            else:
                if os.path.exists(original_filename) and os.path.exists(mp3_filename):
                    print(f"Files already exist: skipping download and using existing video ({original_filename}) and MP3 ({mp3_filename})")
                    return mp3_filename, video_title

                print("Downloading audio and video...")
                ydl.download([url])
                if not os.path.exists(mp3_filename):
                    raise FileNotFoundError(f"[ERROR] MP3 file not found after download: {mp3_filename}")
                if not os.path.exists(original_filename):
                    raise FileNotFoundError(f"[ERROR] Video file not found after download: {original_filename}")

        print(f"Verifying MP3 file: {mp3_filename}")
        file_size = os.path.getsize(mp3_filename)
        print(f"MP3 file size: {file_size} bytes")
        audio = AudioSegment.from_file(mp3_filename)
        duration = len(audio) / 1000.0
        print(f"Actual MP3 duration: {duration} seconds")
        if duration < 10:
            raise ValueError(f"[ERROR] MP3 file is too short: {duration} seconds")
        print("MP3 file validated successfully")
        return mp3_filename, video_title
    except Exception as e:
        print(f"[ERROR] in download_youtube_audio: {e}")
        sys.exit(1)

def convert_to_wav(mp3_path, video_title):
    """Convert MP3 audio to WAV format using pydub, skip if valid WAV exists."""
    sanitized_title = sanitize_filename(video_title)
    wav_path = f"{sanitized_title}.wav"
    print(f"\nStarting MP3 to WAV conversion...")
    print(f"Input MP3 file: {mp3_path}")
    print(f"Target WAV file: {wav_path}")
    try:
        if os.path.exists(wav_path):
            print(f"Existing WAV file found: {wav_path}")
            audio = AudioSegment.from_file(wav_path)
            duration = len(audio) / 1000.0
            print(f"Existing WAV duration: {duration} seconds")
            if duration >= 10:
                print("Existing WAV file validated successfully. Skipping conversion.")
                return wav_path
            else:
                print(f"Warning: Existing WAV file too short: {duration} seconds. Re-converting...")

        if not os.path.exists(mp3_path):
            raise FileNotFoundError(f"[ERROR] MP3 file not found: {mp3_path}")
        print("Loading MP3 file with pydub...")
        audio = AudioSegment.from_file(mp3_path)
        duration = len(audio) / 1000.0
        print(f"MP3 duration: {duration} seconds")
        if duration < 10:
            raise ValueError(f"[ERROR] MP3 file is too short: {duration} seconds")
        print("Exporting audio to WAV format...")
        audio.export(wav_path, format="wav")
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            raise ValueError("[ERROR] WAV file creation failed or is empty")
        print(f"WAV file created: {wav_path}")
        print(f"WAV file size: {os.path.getsize(wav_path)} bytes")
        return wav_path
    except Exception as e:
        print(f"[ERROR] in converting to WAV: {e}")
        sys.exit(1)

def diarize_audio(wav_path):
    """Perform speaker diarization using pyannote.audio with GPU support if available."""
    print(f"\nStarting speaker diarization...")
    print(f"Input WAV file: {wav_path}")
    try:
        print("Loading diarization pipeline...")
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("[ERROR] HF_TOKEN environment variable not set. Please set it to your Hugging Face access token.")
        print("Initializing pyannote.audio pipeline with Hugging Face token...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                          use_auth_token=hf_token).to(device)
        print(f"Performing speaker diarization on {device}...")
        diarization = pipeline(wav_path)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment_length = turn.end - turn.start
            print(f"Processing segment: Speaker {speaker}, Start: {turn.start:.2f}s, End: {turn.end:.2f}s, Length: {segment_length:.2f}s")
            if segment_length < 0.1:
                print(f"Warning: Very short segment detected ({segment_length:.2f}s)")
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        unique_speakers = len(set(s['speaker'] for s in segments))
        print(f"Diarization complete. Detected {unique_speakers} unique speakers")
        return segments
    except Exception as e:
        print(f"[ERROR] in diarization: {e}")
        sys.exit(1)

def transcribe_audio(wav_path, model_name="base"):
    """Transcribe WAV audio using Whisper model with timestamps."""
    print(f"\nStarting audio transcription...")
    print(f"Input WAV file: {wav_path}")
    print(f"Whisper model: {model_name}")
    try:
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"[ERROR] WAV file not found: {wav_path}")
        print(f"Loading Whisper model: {model_name}...")
        model = whisper.load_model(model_name, device=device)
        print(f"Whisper model loaded successfully on {device}")
        print("Transcribing audio...")
        result = model.transcribe(wav_path, verbose=True)
        print(f"Transcription complete. Number of segments: {len(result['segments'])}")
        for i, segment in enumerate(result["segments"], 1):
            print(f"Segment {i}: Start: {segment['start']:.2f}s, End: {segment['end']:.2f}s, Text: {segment['text'].strip()}")
        return result["segments"]
    except Exception as e:
        print(f"[ERROR] in transcribing audio: {e}")
        sys.exit(1)

def align_transcript_with_speakers(transcript_segments, diarization_segments):
    """Align Whisper transcript segments with diarization speaker labels."""
    print(f"\nStarting alignment of transcript with speaker labels...")
    print(f"Number of transcript segments: {len(transcript_segments)}")
    print(f"Number of diarization segments: {len(diarization_segments)}")
    aligned_transcript = []
    for i, t_segment in enumerate(transcript_segments, 1):
        t_start = t_segment["start"]
        t_end = t_segment["end"]
        text = t_segment["text"].strip()
        if not text:
            print(f"Skipping empty transcript segment {i} at {t_start:.2f}s")
            continue
        max_overlap = 0
        best_speaker = "Unknown"
        for d_segment in diarization_segments:
            d_start = d_segment["start"]
            d_end = d_segment["end"]
            overlap_start = max(t_start, d_start)
            overlap_end = min(t_end, d_end)
            overlap = max(0, overlap_end - overlap_start)
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = d_segment["speaker"]
        start_time = str(timedelta(seconds=int(t_start)))
        aligned_line = f"[{start_time}] {best_speaker}: {text}"
        print(f"Aligned segment {i}: {aligned_line}")
        aligned_transcript.append(aligned_line)
    print(f"Alignment complete. Total aligned segments: {len(aligned_transcript)}")
    return aligned_transcript

def save_transcript(transcript_lines, video_title):
    """Save the transcript to a text file with a unique name based on video title."""
    print(f"\nStarting transcript save...")
    sanitized_title = sanitize_filename(video_title)
    output_file = f"{sanitized_title}.txt"
    print(f"Target output file: {output_file}")
    try:
        base, ext = os.path.splitext(output_file)
        counter = 1
        unique_output_file = output_file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if os.path.exists(output_file):
            unique_output_file = f"{base}_{timestamp}{ext}"
            print(f"Output file {output_file} exists. Using unique name: {unique_output_file}")
        while os.path.exists(unique_output_file):
            unique_output_file = f"{base}_{timestamp}_{counter}{ext}"
            print(f"File {unique_output_file} exists. Trying next: {unique_output_file}")
            counter += 1
        print(f"Writing transcript to {unique_output_file}...")
        with open(unique_output_file, "w", encoding="utf-8") as f:
            for i, line in enumerate(transcript_lines, 1):
                f.write(line + "\n")
                print(f"Wrote line {i}: {line}")
        file_size = os.path.getsize(unique_output_file)
        print(f"Transcript saved to {unique_output_file}, size: {file_size} bytes")
        return unique_output_file
    except Exception as e:
        print(f"[ERROR] in saving transcript: {e}")
        sys.exit(1)

def main():
    print(f"\nStarting YouTube audio transcription script...")
    parser = argparse.ArgumentParser(description="Transcribe YouTube audio to text with speaker diarization")
    parser.add_argument("url", help="YouTube URL to process")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading if video and MP3 files exist")
    args = parser.parse_args()
    print(f"Arguments: URL={args.url}, Skip Download={args.skip_download}")

    print(f"\nProcessing audio download...")
    mp3_path, video_title = download_youtube_audio(args.url, skip_download=args.skip_download)

    print(f"\nConverting audio to WAV...")
    wav_path = convert_to_wav(mp3_path, video_title)

    print(f"\nPerforming speaker diarization...")
    diarization_segments = diarize_audio(wav_path)

    print(f"\nTranscribing audio...")
    transcript_segments = transcribe_audio(wav_path, model_name="base")

    print(f"\nAligning transcript with speakers...")
    aligned_transcript = align_transcript_with_speakers(transcript_segments, diarization_segments)

    print(f"\nSaving transcript...")
    save_transcript(aligned_transcript, video_title)

    # Calculate and report total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nCleanup skipped for debugging. Check MP3 ({mp3_path}), WAV ({wav_path}), and video files manually.")
    print(f"Script execution completed successfully. Total execution time: {execution_time:.2f} seconds.")

if __name__ == "__main__":
    main()
