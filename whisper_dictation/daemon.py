#!/usr/bin/env python3
"""Whisper daemon - keeps model loaded, toggles recording on signal."""

import atexit
import json
import logging
import os
import signal
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

# Silence the "process forked after parallelism has already been used" warning.
# Set before any heavy import that might initialise HF tokenizers; we fork
# (subprocess.run) on every transcription to type the result.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .text_output import type_text

LOG_FILE = "/tmp/whisper-dictation-daemon.log"
handlers: list[logging.Handler] = [logging.StreamHandler()]
try:
    handlers.append(logging.FileHandler(LOG_FILE, mode="w"))
except OSError:
    pass
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=handlers,
)
log = logging.getLogger(__name__)

DEFAULT_BACKEND = "moonshine-onnx:base"
SAMPLE_RATE = 16000
PID_FILE = "/tmp/whisper-dictation-daemon.pid"
STATUS_FILE = "/tmp/whisper-dictation-status"
LAST_TEXT_FILE = "/tmp/whisper-dictation-last"
CONFIG_PATH = Path.home() / ".config" / "whisper-dictation" / "config.toml"
STATUS_LOADING = '<span color="#fabd2f">● LOAD</span>'
STATUS_REC = '<span color="#fb4934">● REC</span>'
STATUS_TX = '<span color="#458588">● TX</span>'
IDLE_TIMEOUT = 15 * 60  # 15 minutes


def set_status(status: str):
    try:
        with open(STATUS_FILE, "w") as f:
            f.write(status)
    except OSError:
        pass


def cleanup():
    for f in (PID_FILE, STATUS_FILE):
        try:
            os.unlink(f)
        except OSError:
            pass


def main():
    atexit.register(cleanup)
    # Write PID and show loading status before heavy imports
    try:
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))
        set_status(STATUS_LOADING)
    except OSError as e:
        log.error(f"Cannot write to /tmp: {e}")
        sys.exit(1)

    # Load config
    history_file = None
    vocabulary_hints = None
    backend_spec = DEFAULT_BACKEND
    try:
        with open(CONFIG_PATH, "rb") as f:
            config = tomllib.load(f)
        hf = config.get("history_file")
        if hf:
            history_file = Path(hf).expanduser()
        vocabulary_hints = config.get("vocabulary_hints")
        backend_spec = config.get("backend", DEFAULT_BACKEND)
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        pass
    if vocabulary_hints:
        log.info(f"Vocabulary hints: {vocabulary_hints.strip()}")

    # Heavy imports
    import numpy as np
    import sounddevice as sd
    from .backends import FasterWhisper, MoonshineOnnx

    log.info(f"Loading {backend_spec}...")
    kind, _, model_name = backend_spec.partition(":")
    if not model_name:
        log.error(f"backend spec missing model: {backend_spec!r}")
        sys.exit(1)
    match kind:
        case "moonshine-onnx":
            backend = MoonshineOnnx(model_name, vocabulary_hints=vocabulary_hints)
        case "faster-whisper":
            backend = FasterWhisper(model_name, vocabulary_hints=vocabulary_hints)
        case _:
            log.error(f"unknown backend kind: {kind!r}")
            sys.exit(1)
    log.info("Model loaded.")

    # State
    recording = False
    audio_chunks = []

    def audio_callback(indata, frames, time_info, status):
        if recording:
            audio_chunks.append(indata[:, 0].copy())

    # Pre-create stream to eliminate device init latency on toggle
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
        callback=audio_callback,
    )

    def start_recording():
        nonlocal recording, audio_chunks
        signal.alarm(0)  # Cancel any pending timeout
        audio_chunks = []
        stream.start()
        recording = True
        set_status(STATUS_REC)
        log.info("Recording...")

    def stop_recording():
        nonlocal recording
        recording = False
        stream.stop()

        if not audio_chunks:
            set_status("")
            signal.alarm(IDLE_TIMEOUT)
            log.info("No audio.")
            return

        set_status(STATUS_TX)

        audio = np.concatenate(audio_chunks)
        log.info(f"Transcribing {len(audio)/SAMPLE_RATE:.1f}s...")

        t0 = time.monotonic()
        text = backend.transcribe(audio)
        log.info(f"Transcribed in {time.monotonic()-t0:.2f}s")

        if text:
            text_out = text + " "
            try:
                with open(LAST_TEXT_FILE, "w") as f:
                    f.write(text_out)
            except OSError:
                pass
            if history_file:
                try:
                    history_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(history_file, "a") as f:
                        json.dump({
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "text": text,
                        }, f)
                        f.write("\n")
                except OSError:
                    pass
            type_text(text_out)

        set_status("")
        signal.alarm(IDLE_TIMEOUT)

    def toggle(sig, frame):
        nonlocal recording
        if recording:
            stop_recording()
        else:
            start_recording()

    def shutdown(sig, frame):
        nonlocal recording
        if recording:
            stop_recording()
        stream.stop()
        stream.close()
        log.info("Daemon stopped.")
        sys.exit(0)

    def idle_timeout(sig, frame):
        if not recording:
            log.info("Idle timeout, shutting down.")
            shutdown(sig, frame)
        # If recording, ignore - alarm will be reset when recording stops

    signal.signal(signal.SIGUSR1, toggle)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGALRM, idle_timeout)

    # Start recording immediately
    start_recording()

    # Wait forever
    while True:
        signal.pause()


if __name__ == "__main__":
    main()
