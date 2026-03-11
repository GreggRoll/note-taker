import html
import os
import queue
import tempfile
import threading
import time
import warnings

import numpy as np
import wave
import streamlit as st
import streamlit.components.v1 as components
import whisper
from streamlit_webrtc import webrtc_streamer, WebRtcMode
WEBRTC_AVAILABLE = True


SAMPLE_RATE = 16_000
CHANNELS = 1
RECORD_CHUNK_SECONDS = 0.5
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
    # Local desktop recording via soundcard has been removed; use browser
    # capture (streamlit-webrtc) instead. Signal an error if this function
    # is invoked.
    error_queue.put("Local desktop recording is not supported. Use browser capture.")
    return


def _transcribe_audio_array(model, audio_array: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = tmp.name

    try:
        # Write WAV using the standard library to avoid extra deps.
        audio = audio_array.astype(np.float32)
        clipped = np.clip(audio, -1.0, 1.0)
        int16 = (clipped * 32767).astype(np.int16)
        with wave.open(temp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(int16.tobytes())

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
    control: dict,
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
                with control["lock"]:
                    active_section_idx = control["active_section_idx"]
                text_queue.put((active_section_idx, text))

    if chunks:
        merged_audio = np.concatenate(chunks)
        text = _transcribe_audio_array(model, merged_audio)
        if text:
            with control["lock"]:
                active_section_idx = control["active_section_idx"]
            text_queue.put((active_section_idx, text))


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
        "transcript_sections": [{"parts": []}],
        "control": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_capture(model_name: str):
    if st.session_state.running:
        return

    st.session_state.worker_error = ""
    st.session_state.transcript_sections = [{"parts": []}]
    st.session_state.stop_event = threading.Event()
    st.session_state.audio_queue = queue.Queue()
    st.session_state.text_queue = queue.Queue()
    st.session_state.error_queue = queue.Queue()
    st.session_state.control = {
        "active_section_idx": 0,
        "lock": threading.Lock(),
    }

    # Build RTC configuration from environment (allow STUN/TURN overrides)
    def build_rtc_configuration():
        ice_servers = []
        stun_env = os.getenv("STUN_SERVERS", "stun:stun.l.google.com:19302")
        for s in stun_env.split(","):
            s = s.strip()
            if s:
                ice_servers.append({"urls": [s]})

        turn_url = os.getenv("TURN_URL")
        turn_username = os.getenv("TURN_USERNAME")
        turn_password = os.getenv("TURN_PASSWORD")
        if turn_url and turn_username and turn_password:
            ice_servers.append({
                "urls": [turn_url],
                "username": turn_username,
                "credential": turn_password,
            })

        return {"iceServers": ice_servers} if ice_servers else {}

    rtc_config = build_rtc_configuration()

    ctx = webrtc_streamer(
        key="desktop_audio",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration=rtc_config,
    )

    def browser_reader(ctx, stop_event, audio_queue, error_queue):
        try:
            while not stop_event.is_set() and ctx.state.playing:
                try:
                    frame = ctx.audio_receiver.get_frame(timeout=1)
                except Exception:
                    continue
                try:
                    arr = frame.to_ndarray()
                    if arr.ndim == 2:
                        mono = np.mean(arr, axis=0).astype(np.float32)
                    else:
                        mono = arr.astype(np.float32)
                    audio_queue.put(mono)
                except Exception as exc:
                    error_queue.put(f"Audio frame processing error: {exc}")
        except Exception as exc:
            error_queue.put(f"Browser audio reader failed: {exc}")

    st.session_state.audio_thread = threading.Thread(
        target=browser_reader,
        args=(ctx, st.session_state.stop_event, st.session_state.audio_queue, st.session_state.error_queue),
        daemon=True,
    )
    st.session_state.transcribe_thread = threading.Thread(
        target=transcribe_from_queue,
        args=(
            st.session_state.stop_event,
            st.session_state.audio_queue,
            st.session_state.text_queue,
            st.session_state.error_queue,
            st.session_state.control,
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
    st.session_state.transcript_sections = [{"parts": []}]
    if st.session_state.control is not None:
        with st.session_state.control["lock"]:
            st.session_state.control["active_section_idx"] = 0


def split_transcript_section():
    st.session_state.transcript_sections.append({"parts": []})
    if st.session_state.control is not None:
        with st.session_state.control["lock"]:
            st.session_state.control["active_section_idx"] = (
                len(st.session_state.transcript_sections) - 1
            )


def drain_queues():
    if st.session_state.text_queue is not None:
        while not st.session_state.text_queue.empty():
            section_idx, text = st.session_state.text_queue.get()
            if section_idx >= len(st.session_state.transcript_sections):
                section_idx = len(st.session_state.transcript_sections) - 1
            st.session_state.transcript_sections[section_idx]["parts"].append(text)
    if st.session_state.error_queue is not None:
        while not st.session_state.error_queue.empty():
            st.session_state.worker_error = st.session_state.error_queue.get()


def copy_button_component(text: str, section_idx: int):
    escaped = html.escape(text, quote=True)
    components.html(
        f"""
        <div style="display:flex; align-items:center; gap:8px;">
          <input id="transcript-src-{section_idx}" type="hidden" value="{escaped}" />
          <button
            style="padding:0.5rem 0.75rem; border:1px solid #ccc; border-radius:6px; cursor:pointer;"
            onclick="
              const text = document.getElementById('transcript-src-{section_idx}').value;
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
            "Detected NumPy 2.x — some dependencies may be incompatible. "
            "If you encounter issues, consider installing NumPy < 2 (pip install \"numpy<2\")."
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
        if st.button("Split", use_container_width=True, disabled=not st.session_state.running):
            split_transcript_section()

    if st.button("Clear All Sections", use_container_width=True, disabled=st.session_state.running):
        clear_transcript()

    status_text = "Recording..." if st.session_state.running else "Stopped"
    st.caption(f"Status: {status_text}")

    drain_queues()

    if st.session_state.worker_error:
        st.error(st.session_state.worker_error)

    has_any_text = False
    for i, section in enumerate(st.session_state.transcript_sections):
        section_text = " ".join(section["parts"]).strip()
        st.text_area(f"Section {i + 1}", value=section_text, height=200, disabled=True)
        if section_text:
            has_any_text = True
            copy_button_component(section_text, i)
        else:
            st.caption(f"Section {i + 1} is empty.")

    if not has_any_text:
        st.caption("Transcript is empty.")

    if st.session_state.running:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
