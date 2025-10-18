TranscriptExtractor: YouTube Audio Transcription with Speaker Diarization
Overview
TranscriptExtractor.py is a Python script that downloads audio from a YouTube video, converts it to WAV format, performs speaker diarization to identify distinct speakers, transcribes the audio using OpenAI's Whisper model, aligns the transcript with speaker labels, and saves the result to a text file. The script is designed for efficiency, leveraging GPU acceleration for transcription and diarization when available, and includes features like execution time tracking, verbose debugging output, and unique filenames to avoid overwriting files.
This script is ideal for researchers, content creators, or developers needing accurate transcriptions of YouTube videos with speaker attribution, such as for podcasts, interviews, or lectures.
Features

YouTube Audio Download: Downloads audio from a YouTube URL using yt-dlp, retaining the original video file for flexibility.
Skip Redundant Downloads: Checks for existing video and MP3 files to avoid unnecessary downloads when the --skip-download flag is used.
MP3 to WAV Conversion: Converts MP3 audio to WAV format using pydub, skipping conversion if a valid WAV file exists.
Speaker Diarization: Uses pyannote.audio to identify and segment speakers, with debug output for segment lengths to diagnose issues.
Audio Transcription: Transcribes audio using OpenAI's Whisper model with timestamped segments, supporting GPU acceleration.
Transcript Alignment: Aligns transcription segments with speaker labels based on temporal overlap for accurate speaker attribution.
Unique Filenames: Names output files (video, MP3, WAV, transcript) using the sanitized YouTube video title, appending timestamps or counters to transcripts to prevent overwriting.
Execution Time Tracking: Reports total script execution time in seconds for performance monitoring.
GPU Acceleration: Utilizes CUDA for both Whisper transcription and pyannote.audio diarization if an NVIDIA GPU is available, with clear CPU fallback explanations.
Verbose Output: Provides detailed, clean console output for each step (without [INFO] tags, using [ERROR] for errors) to aid debugging.
CUDA Debugging: Performs CUDA availability checks at script start, displaying device details or reasons for CPU fallback.

Prerequisites
To run TranscriptExtractor.py, ensure your system meets the following requirements:

Operating System: Windows, Linux, or macOS.
Python Version: Python 3.8 or higher.
Hardware: NVIDIA GPU with CUDA support (optional, for faster transcription and diarization). A CPU is sufficient if GPU is unavailable.
Disk Space: Sufficient space for video, MP3, WAV, and transcript files (e.g., a 1-hour video may require ~1GB for WAV).
Internet Connection: Required for downloading YouTube videos and Hugging Face models.

Installation
Follow these steps to set up the environment and dependencies for TranscriptExtractor.py.
1. Install Python
Ensure Python 3.8 or higher is installed. Download from python.org or use a package manager:

Linux: sudo apt-get install python3
macOS: brew install python3
Windows: Install via the official installer and add to PATH.

Verify installation:
python --version

2. Install Dependencies
Install the required Python packages using pip. For GPU support, include the PyTorch index for your CUDA version (e.g., cu121 for CUDA 12.1). Replace cu121 with your CUDA version (check with nvcc --version) or omit for CPU-only:
pip install yt_dlp pydub openai-whisper pyannote.audio torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Dependencies:

yt_dlp: Downloads YouTube videos and extracts audio.
pydub: Converts MP3 to WAV format (requires ffmpeg).
openai-whisper: Performs audio transcription.
pyannote.audio: Conducts speaker diarization (requires Hugging Face token).
torch: Enables GPU acceleration and model support.
argparse, os, sys, datetime, re, time: Standard Python libraries (included with Python).

3. Install FFmpeg
pydub requires ffmpeg for audio conversion. Install it based on your OS:

Linux:sudo apt-get update
sudo apt-get install ffmpeg


macOS:brew install ffmpeg


Windows:
Download ffmpeg from ffmpeg.org or a trusted source like gyan.dev.
Extract the archive and add the bin folder (containing ffmpeg.exe) to your system PATH.
Verify installation:ffmpeg -version





4. Set Hugging Face Token
The pyannote.audio library requires a Hugging Face access token to download the diarization model:

Create a Hugging Face account at huggingface.co.
Generate an access token at Settings > Access Tokens.
Set the token as an environment variable:
Linux/macOS:export HF_TOKEN=your_huggingface_token

Add to ~/.bashrc or ~/.zshrc for persistence.
Windows:setx HF_TOKEN your_huggingface_token

Reopen the terminal to apply.


Verify the token is set:echo $HF_TOKEN



Usage
Clone or download this repository to your local machine. Navigate to the directory containing TranscriptExtractor.py and run the script with a YouTube URL:
python TranscriptExtractor.py "https://www.youtube.com/watch?v=video_id" [--skip-download]

Command-Line Arguments

url: The YouTube video URL to process (required).
--skip-download: Optional flag to skip downloading if both the video and MP3 files already exist in the working directory.

Example
python TranscriptExtractor.py "https://www.youtube.com/watch?v=example_id"

This processes the video, generating files like:

Video: Example_Video_Title.m4a
MP3: Example_Video_Title.mp3
WAV: Example_Video_Title.wav
Transcript: Example_Video_Title_20251017_205700.txt

Output Files

Video File: <sanitized_title>.<ext> (e.g., .m4a, .webm), retained after download.
MP3 File: <sanitized_title>.mp3.
WAV File: <sanitized_title>.wav.
Transcript File: <sanitized_title>_<timestamp>.txt or <sanitized_title>_<timestamp>_<counter>.txt if the base file exists.
Files are saved in the script's working directory.

Sample Output
For a video with CUDA enabled:
=========================================
= TRANSCRIPT EXTRACTOR                  =
=========================================

Starting CUDA availability check...
CUDA available: True
SUCCESS: CUDA detected and will be used for transcription and diarization.
Number of CUDA devices: 1
CUDA Device 0: NVIDIA GeForce RTX 3080
Current CUDA device: 0
CUDA device properties: _CudaDeviceProperties(name='NVIDIA GeForce RTX 3080', ...)
Using device: cuda:0

Starting YouTube audio transcription script...
Arguments: URL=https://www.youtube.com/watch?v=example_id, Skip Download=False

Processing audio download...
Extracting video info without download...
Video title: Example Video Title
...

Starting speaker diarization...
Input WAV file: Example_Video_Title.wav
Loading diarization pipeline...
Initializing pyannote.audio pipeline with Hugging Face token...
Performing speaker diarization on cuda:0...
Processing segment: Speaker SPEAKER_00, Start: 0.00s, End: 2.50s, Length: 2.50s
...

Cleanup skipped for debugging. Check MP3, WAV, and video files manually.
Script execution completed successfully. Total execution time: 300.25 seconds.

If an error occurs:
[ERROR] in download_youtube_audio: File not found

GPU Support
The script leverages CUDA for both Whisper transcription and pyannote.audio diarization if an NVIDIA GPU is available. CUDA is checked at script start, with detailed output:

If CUDA is detected: Lists device count, names, and properties.
If CUDA is not available: Falls back to CPU and explains reasons (e.g., no GPU, outdated drivers, non-CUDA PyTorch).

To verify GPU support:
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

If GPU is not detected:

Ensure an NVIDIA GPU is installed.
Update NVIDIA drivers from nvidia.com.
Reinstall PyTorch with CUDA support:pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


Check CUDA version with nvcc --version and match it to the PyTorch index.

Troubleshooting
Common Issues

"[ERROR] Missing files for skip-download":

Ensure video and MP3 files exist in the working directory when using --skip-download.
Run without the flag to re-download.


"[ERROR] HF_TOKEN environment variable not set":

Set the Hugging Face token:export HF_TOKEN=your_huggingface_token




Diarization Warning: std(): degrees of freedom is <= 0:

Indicates short or empty audio segments. Check the diarization output for "Warning: Very short segment detected".
Try a lighter model:pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token).to(device)


Verify the WAV file with:ffprobe your_file.wav




"[ERROR] in converting to WAV: ffmpeg not found":

Install ffmpeg and ensure it's in your PATH (see Installation section).
Test: ffmpeg -version.


Slow Performance:

Use a smaller Whisper model (e.g., tiny instead of base) in the transcribe_audio function:model = whisper.load_model("tiny", device=device)


Ensure GPU is enabled (check CUDA output).



Debugging

Verbose Output: The script provides detailed logs for each step (download, conversion, diarization, transcription, alignment, saving).
Segment Debugging: Diarization output includes segment lengths to identify issues with short segments.
Execution Time: Total runtime is reported to help assess performance.

Notes

Cleanup: The script skips cleanup for debugging. Manually delete video, MP3, and WAV files if no longer needed.
File Size: WAV files can be large (e.g., ~800MB for a 1-hour video). Ensure sufficient disk space.
Model Selection: The base Whisper model balances speed and accuracy. Use tiny for faster processing or large for better accuracy, adjusting in the main function.
Diarization Sensitivity: Short or silent segments may cause warnings. The debug output helps identify these.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a branch: git checkout -b feature/your-feature.
Make changes and test thoroughly.
Commit: git commit -m "Add your feature".
Push: git push origin feature/your-feature.
Open a pull request on GitHub.

Please include tests and update this README if necessary.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Author
Created by Grok, powered by xAI.
Acknowledgments

yt_dlp for robust YouTube downloading.
pydub for audio conversion.
openai-whisper for transcription.
pyannote.audio for speaker diarization.
torch for GPU acceleration.
