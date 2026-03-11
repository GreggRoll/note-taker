import ctypes
import os
import queue
import tempfile
import threading
import warnings
from dataclasses import dataclass, field

import numpy as np
import soundcard as sc
import soundfile as sf
import whisper


SAMPLE_RATE = 16_000
CHANNELS = 1
RECORD_CHUNK_SECONDS = 0.5
TRANSCRIBE_WINDOW_SECONDS = 6


_MODEL_CACHE: dict[str, object] = {}
_MODEL_LOCK = threading.Lock()


def load_whisper_model(model_name: str):
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(model_name)
        if model is None:
            model = whisper.load_model(model_name)
            _MODEL_CACHE[model_name] = model
        return model


def numpy_major_version() -> int:
    version = np.__version__.split(".", 1)[0]
    try:
        return int(version)
    except ValueError:
        return 0


def _record_desktop_audio(
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
            f"Audio capture setup failed. Check the server output device. Details: {exc}"
        )
        return

    frames_per_chunk = int(SAMPLE_RATE * RECORD_CHUNK_SECONDS)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="data discontinuity in recording")
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


def _transcribe_from_queue(
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


@dataclass
class TranscriptionEngine:
    running: bool = False
    worker_error: str = ""
    transcript_sections: list = field(default_factory=lambda: [{"parts": []}])
    stop_event: threading.Event | None = None
    audio_queue: queue.Queue | None = None
    text_queue: queue.Queue | None = None
    error_queue: queue.Queue | None = None
    audio_thread: threading.Thread | None = None
    transcribe_thread: threading.Thread | None = None
    control: dict | None = None
    current_model: str = "base"

    def __post_init__(self):
        self._lock = threading.Lock()

    def start(self, model_name: str):
        with self._lock:
            if self.running:
                return

            self.worker_error = ""
            self.current_model = model_name
            self.transcript_sections = [{"parts": []}]
            self.stop_event = threading.Event()
            self.audio_queue = queue.Queue()
            self.text_queue = queue.Queue()
            self.error_queue = queue.Queue()
            self.control = {"active_section_idx": 0, "lock": threading.Lock()}

            self.audio_thread = threading.Thread(
                target=_record_desktop_audio,
                args=(self.stop_event, self.audio_queue, self.error_queue),
                daemon=True,
            )
            self.transcribe_thread = threading.Thread(
                target=_transcribe_from_queue,
                args=(
                    self.stop_event,
                    self.audio_queue,
                    self.text_queue,
                    self.error_queue,
                    self.control,
                    model_name,
                ),
                daemon=True,
            )

            self.audio_thread.start()
            self.transcribe_thread.start()
            self.running = True

    def stop(self):
        with self._lock:
            if not self.running:
                return
            assert self.stop_event is not None
            self.stop_event.set()

        if self.audio_thread is not None:
            self.audio_thread.join(timeout=3)
        if self.transcribe_thread is not None:
            self.transcribe_thread.join(timeout=6)

        with self._lock:
            self.running = False

    def clear(self):
        with self._lock:
            self.transcript_sections = [{"parts": []}]
            if self.control is not None:
                with self.control["lock"]:
                    self.control["active_section_idx"] = 0

    def split(self):
        with self._lock:
            self.transcript_sections.append({"parts": []})
            if self.control is not None:
                with self.control["lock"]:
                    self.control["active_section_idx"] = len(self.transcript_sections) - 1

    def drain_queues(self):
        with self._lock:
            if self.text_queue is not None:
                while not self.text_queue.empty():
                    section_idx, text = self.text_queue.get()
                    if section_idx >= len(self.transcript_sections):
                        section_idx = len(self.transcript_sections) - 1
                    self.transcript_sections[section_idx]["parts"].append(text)

            if self.error_queue is not None:
                while not self.error_queue.empty():
                    self.worker_error = self.error_queue.get()

    def snapshot(self) -> dict:
        self.drain_queues()
        with self._lock:
            return {
                "running": self.running,
                "error": self.worker_error,
                "model": self.current_model,
                "numpy_ok": numpy_major_version() < 2,
                "sections": [
                    " ".join(section["parts"]).strip() for section in self.transcript_sections
                ],
            }
