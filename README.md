# Desktop Audio Whisper Transcriber (Web Deployable)

Web app that captures **desktop/system audio on the host server**, transcribes with Whisper, and exposes controls in a browser UI.

## Important Architecture Note

This app captures audio from the machine where the Python server process runs.

- If you deploy to a cloud VM/container with no audio output device, desktop loopback capture will fail.
- To capture real desktop audio, run on a Windows host/session that has an active playback device.
- Browser clients only control and view transcripts; the audio source is server-side.

## Tech Stack

| Layer | Package |
|---|---|
| Web server/API | `fastapi`, `uvicorn` |
| Server-rendered UI | `jinja2` |
| Speech-to-text | `openai-whisper` |
| Desktop loopback audio | `soundcard` |
| Audio file handling | `soundfile` |
| Numeric processing | `numpy<2` |

## Quick Start (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## How To Use

1. Pick a Whisper model.
2. Click `Start` to begin capture and transcription.
3. Click `Split` to start a new transcript section.
4. Click `Stop` to flush remaining audio and finalize text.
5. Use `Copy Transcript` per section.
6. Click `Clear All Sections` while stopped.

## Deployment Notes

- Keep the process attached to an interactive desktop session if you need loopback audio.
- On Windows Server, ensure audio service/device is enabled.
- First model load can be slow due to Whisper weights download.
- If NumPy 2.x is installed, run:

```powershell
pip install "numpy<2"
```

## Project Files

- `app.py`: FastAPI app and API routes.
- `transcription_engine.py`: audio capture and Whisper worker engine.
- `templates/index.html`: browser UI.
- `static/styles.css`: UI styles.
- `requirements.txt`: Python dependencies.
