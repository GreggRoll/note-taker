import html
import ctypes
import os
import queue
import tempfile
import threading
import time

import numpy as np
import soundcard as sc
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components
import whisper


SAMPLE_RATE = 16_000
CHANNELS = 1
RECORD_CHUNK_SECONDS = 1.5
TRANSCRIBE_WINDOW_SECONDS = 6


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_name: str):
    return whisper.load_model(model_name)


def numpy_major_version() -> int:
    version = np.__version__.split(".", 1)[0]
    try:
        return int(version)
    except ValueError:
        return 0


def record_desktop_audio(
    stop_event: threading.Event, audio_queue: queue.Queue, error_queue: queue.Queue
):
    coinit_result = None
    if os.name == "nt":
        # soundcard uses Windows COM APIs; worker threads must initialize COM explicitly.
        coinit_result = ctypes.windll.ole32.CoInitializeEx(None, 0x2)

    try:
        speaker = sc.default_speaker()
        microphone = sc.get_microphone(speaker.name, include_loopback=True)
    except Exception as exc:
        error_queue.put(
            f"Audio capture setup failed. Check your desktop output device. Details: {exc}"
        )
        return

    frames_per_chunk = int(SAMPLE_RATE * RECORD_CHUNK_SECONDS)

    try:
        with microphone.recorder(samplerate=SAMPLE_RATE, channels=CHANNELS) as recorder:
            while not stop_event.is_set():
                block = recorder.record(numframes=frames_per_chunk)
                mono = np.squeeze(block).astype(np.float32)
                audio_queue.put(mono)
    except Exception as exc:
        error_queue.put(f"Audio recording failed: {exc}")
    finally:
        if os.name == "nt" and coinit_result in (0, 1):
            ctypes.windll.ole32.CoUninitialize()


def _transcribe_audio_array(model, audio_array: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = tmp.name

    try:
        sf.write(temp_path, audio_array, SAMPLE_RATE)
        result = model.transcribe(temp_path, fp16=False)
        return result.get("text", "").strip()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def transcribe_from_queue(
    stop_event: threading.Event,
    audio_queue: queue.Queue,
    text_queue: queue.Queue,
    error_queue: queue.Queue,
    model_name: str,
):
    try:
        model = load_whisper_model(model_name)
    except Exception as exc:
        error_queue.put(f"Whisper model failed to load: {exc}")
        return

    window_samples = int(SAMPLE_RATE * TRANSCRIBE_WINDOW_SECONDS)
    chunks = []
    total_samples = 0

    while not stop_event.is_set() or not audio_queue.empty():
        try:
            audio_chunk = audio_queue.get(timeout=0.25)
        except queue.Empty:
            continue

        chunks.append(audio_chunk)
        total_samples += len(audio_chunk)

        if total_samples >= window_samples:
            merged_audio = np.concatenate(chunks)
            chunks = []
            total_samples = 0

            text = _transcribe_audio_array(model, merged_audio)
            if text:
                text_queue.put(text)

    if chunks:
        merged_audio = np.concatenate(chunks)
        text = _transcribe_audio_array(model, merged_audio)
        if text:
            text_queue.put(text)


def ensure_state():
    defaults = {
        "running": False,
        "worker_error": "",
        "stop_event": None,
        "audio_queue": None,
        "audio_thread": None,
        "transcribe_thread": None,
        "text_queue": None,
        "error_queue": None,
        "transcript_parts": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_capture(model_name: str):
    if st.session_state.running:
        return

    st.session_state.worker_error = ""
    st.session_state.transcript_parts = []
    st.session_state.stop_event = threading.Event()
    st.session_state.audio_queue = queue.Queue()
    st.session_state.text_queue = queue.Queue()
    st.session_state.error_queue = queue.Queue()

    st.session_state.audio_thread = threading.Thread(
        target=record_desktop_audio,
        args=(
            st.session_state.stop_event,
            st.session_state.audio_queue,
            st.session_state.error_queue,
        ),
        daemon=True,
    )
    st.session_state.transcribe_thread = threading.Thread(
        target=transcribe_from_queue,
        args=(
            st.session_state.stop_event,
            st.session_state.audio_queue,
            st.session_state.text_queue,
            st.session_state.error_queue,
            model_name,
        ),
        daemon=True,
    )

    st.session_state.audio_thread.start()
    st.session_state.transcribe_thread.start()
    st.session_state.running = True


def stop_capture():
    if not st.session_state.running:
        return

    st.session_state.stop_event.set()

    if st.session_state.audio_thread is not None:
        st.session_state.audio_thread.join(timeout=3)
    if st.session_state.transcribe_thread is not None:
        st.session_state.transcribe_thread.join(timeout=6)

    st.session_state.running = False


def clear_transcript():
    st.session_state.transcript_parts = []


def drain_queues():
    if st.session_state.text_queue is not None:
        while not st.session_state.text_queue.empty():
            st.session_state.transcript_parts.append(st.session_state.text_queue.get())
    if st.session_state.error_queue is not None:
        while not st.session_state.error_queue.empty():
            st.session_state.worker_error = st.session_state.error_queue.get()


def copy_button_component(text: str):
    escaped = html.escape(text, quote=True)
    components.html(
        f"""
        <div style="display:flex; align-items:center; gap:8px;">
          <input id="transcript-src" type="hidden" value="{escaped}" />
          <button
            style="padding:0.5rem 0.75rem; border:1px solid #ccc; border-radius:6px; cursor:pointer;"
            onclick="
              const text = document.getElementById('transcript-src').value;
              navigator.clipboard.writeText(text);
              this.textContent='Copied';
              setTimeout(() => this.textContent='Copy Transcript', 1200);
            "
          >
            Copy Transcript
          </button>
        </div>
        """,
        height=55,
    )


def main():
    st.set_page_config(page_title="Desktop Audio Transcriber", layout="centered")
    st.title("Desktop Audio to Text")
    st.write("Capture system audio, transcribe with Whisper, then copy the transcript.")

    ensure_state()
    numpy_ok = numpy_major_version() < 2

    if not numpy_ok:
        st.error(
            "Detected NumPy 2.x, which is incompatible with soundcard recording in this app. "
            "Install NumPy < 2 (example: pip install \"numpy<2\") and restart Streamlit."
        )

    model_name = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small", "medium"],
        index=1,
        disabled=st.session_state.running,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(
            "Start",
            use_container_width=True,
            disabled=st.session_state.running or not numpy_ok,
        ):
            start_capture(model_name)
    with col2:
        if st.button("Stop", use_container_width=True, disabled=not st.session_state.running):
            stop_capture()
    with col3:
        if st.button("Clear", use_container_width=True, disabled=st.session_state.running):
            clear_transcript()

    status_text = "Recording..." if st.session_state.running else "Stopped"
    st.caption(f"Status: {status_text}")

    drain_queues()

    if st.session_state.worker_error:
        st.error(st.session_state.worker_error)

    transcript_text = " ".join(st.session_state.transcript_parts).strip()
    st.text_area("Transcript", value=transcript_text, height=280)

    if transcript_text:
        copy_button_component(transcript_text)
    else:
        st.caption("Transcript is empty.")

    if st.session_state.running:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
