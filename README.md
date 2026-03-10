# Desktop Audio Whisper Transcriber

Real-time desktop audio transcription with a simple Streamlit interface powered by OpenAI Whisper.

## Features

- Record **system/desktop audio** using loopback capture.
- Start and stop transcription with one click.
- View transcript updates inside the app.
- Copy the transcript to clipboard from the UI.
- Choose Whisper model size (`tiny`, `base`, `small`, `medium`).

## Tech Stack

| Layer | Package |
|---|---|
| UI | `streamlit` |
| Speech-to-text | `openai-whisper` |
| Desktop loopback audio | `soundcard` |
| Audio file handling | `soundfile` |
| Numeric processing | `numpy<2` |

## Quick Start (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## How To Use

1. Select a Whisper model.
2. Click `Start` to begin desktop audio capture and transcription.
3. Click `Stop` to flush remaining audio and finalize text.
4. Click `Copy Transcript` to copy everything to your clipboard.
5. Use `Clear` to reset the transcript.

## Compatibility Notes

- First run may take longer while Whisper model weights download.
- This app currently expects NumPy 1.x for `soundcard` compatibility.
- If NumPy 2.x is installed, run:

```powershell
pip install "numpy<2"
```

Then restart Streamlit.

## Project Files

- `app.py`: Streamlit app, audio capture threads, Whisper transcription flow.
- `requirements.txt`: Python dependencies.

## Troubleshooting

- `Error 0x800401f0`: COM initialization issue on Windows.
  - The app already initializes COM in the recording thread. If this persists, restart the app and verify your default playback device.
- `fromstring is removed, use frombuffer instead`:
  - Install NumPy 1.x with `pip install "numpy<2"`.
