#!/usr/bin/env python3
"""Live streaming transcription using Qwen3-ASR + VAD gate.

Captures audio from two sources simultaneously:
  - Mic via sounddevice (your voice)
  - System via capture_system_audio --pipe (other person's voice)

VAD filters silence before feeding to Qwen3-ASR streaming API.
No speaker labels in streaming mode — use batch transcribe.py for that.

Run alongside call-recorder (which records to files independently).

Usage: python3 live_stream_transcribe.py [--language Russian]
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

CAPTURE_BINARY = Path(__file__).parent / "capture_system_audio"
QWEN_MODEL = "/Users/ilya/models/qwen3-asr-1.7b"
SAMPLE_RATE = 16000
CHUNK_SEC = 2
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SEC
VAD_WINDOW = 512  # Silero VAD expects 512 samples at 16kHz


def chunk_has_speech(chunk: np.ndarray, vad_model, threshold: float = 0.3) -> bool:
    for j in range(0, len(chunk) - VAD_WINDOW, VAD_WINDOW):
        prob = vad_model(torch.tensor(chunk[j:j + VAD_WINDOW]), SAMPLE_RATE).item()
        if prob > threshold:
            return True
    return False


def find_mic_device() -> int | None:
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        if "MacBook Air Microphone" in d["name"] and d["max_input_channels"] > 0:
            return i
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="Russian")
    parser.add_argument("--model", default=QWEN_MODEL)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    token_file = Path.home() / ".huggingface_token"
    if token_file.exists() and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = token_file.read_text().strip()

    # Load VAD
    print("Loading VAD...", flush=True)
    vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)

    # Load Qwen3-ASR
    print(f"Loading Qwen3-ASR ({Path(args.model).name})...", flush=True)
    from mlx_qwen3_asr import Session
    session = Session(model=args.model)

    # Mic setup
    import sounddevice as sd
    mic_idx = find_mic_device()
    if mic_idx is None:
        print("WARNING: MacBook Air Microphone not found", flush=True)
    else:
        print(f"Mic: device {mic_idx}", flush=True)

    # System audio setup
    if not CAPTURE_BINARY.exists():
        print(f"ERROR: {CAPTURE_BINARY} not found", flush=True)
        sys.exit(1)

    sys_proc = subprocess.Popen(
        [str(CAPTURE_BINARY), "/dev/null", "--pipe"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    time.sleep(0.5)
    if sys_proc.poll() is not None:
        print("ERROR: capture_system_audio failed to start", flush=True)
        sys.exit(1)
    print("System: ScreenCaptureKit", flush=True)

    # Mic buffer (filled by sounddevice callback thread)
    mic_buf = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    mic_pos = [0]
    mic_lock = threading.Lock()

    def mic_callback(indata, frames, time_info, status):
        with mic_lock:
            n = min(frames, CHUNK_SAMPLES - mic_pos[0])
            if n > 0:
                mic_buf[mic_pos[0]:mic_pos[0] + n] = indata[:n, 0]
                mic_pos[0] += n

    mic_stream = None
    if mic_idx is not None:
        mic_stream = sd.InputStream(
            device=mic_idx, samplerate=SAMPLE_RATE, channels=1,
            blocksize=1024, dtype="float32", callback=mic_callback,
        )
        mic_stream.start()

    # Streaming
    state = session.init_streaming(language=args.language, chunk_size_sec=float(CHUNK_SEC))
    prev_text = ""
    all_lines = []
    stop = False

    def handle_signal(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # System audio: float32 stereo 48kHz from ScreenCaptureKit
    sys_bytes_per_chunk = 48000 * 2 * 4 * CHUNK_SEC  # stereo float32 @ 48kHz

    t_start = time.time()
    audio_pos = 0.0
    fed = 0
    skipped = 0

    print("\nListening... (Ctrl+C to stop)\n", flush=True)

    try:
        while not stop:
            # Read system audio chunk
            raw = sys_proc.stdout.read(sys_bytes_per_chunk)
            if not raw or len(raw) < 100:
                if sys_proc.poll() is not None:
                    break
                time.sleep(0.1)
                continue

            # Convert system: float32 stereo 48kHz → float32 mono 16kHz
            sys_audio = np.frombuffer(raw, dtype=np.float32)
            if len(sys_audio) > 1 and len(sys_audio) % 2 == 0:
                sys_audio = sys_audio.reshape(-1, 2).mean(axis=1)
            sys_16k = sys_audio[::3][:CHUNK_SAMPLES]  # decimate 48k→16k

            # Get mic chunk
            with mic_lock:
                mic_chunk = mic_buf[:mic_pos[0]].copy()
                mic_pos[0] = 0

            # Pad to chunk size
            if len(sys_16k) < CHUNK_SAMPLES:
                sys_16k = np.pad(sys_16k, (0, CHUNK_SAMPLES - len(sys_16k)))
            if len(mic_chunk) < CHUNK_SAMPLES:
                mic_chunk = np.pad(mic_chunk, (0, CHUNK_SAMPLES - len(mic_chunk)))
            else:
                mic_chunk = mic_chunk[:CHUNK_SAMPLES]

            # Mix
            mixed = (mic_chunk * 0.5 + sys_16k * 0.5).astype(np.float32)

            # VAD gate
            if not chunk_has_speech(mixed, vad_model):
                skipped += 1
                audio_pos += CHUNK_SEC
                continue

            # Feed to model
            fed += 1
            state = session.feed_audio(mixed, state)
            audio_pos += CHUNK_SEC

            text = state.text if hasattr(state, "text") else ""
            if text and text != prev_text:
                new = text[len(prev_text):].strip()
                if new:
                    elapsed = time.time() - t_start
                    m, s = divmod(int(elapsed), 60)
                    print(f"[{m:02d}:{s:02d}] {new}", flush=True)
                    all_lines.append(f"[{m:02d}:{s:02d}] {new}")
                prev_text = text

    except Exception as e:
        print(f"Error: {e}", flush=True)

    # Cleanup
    if mic_stream:
        mic_stream.stop()
    sys_proc.kill()
    sys_proc.wait()

    # Finalize
    state = session.finish_streaming(state)
    text = state.text if hasattr(state, "text") else ""
    if text and text != prev_text:
        new = text[len(prev_text):].strip()
        if new:
            print(f"[final] {new}", flush=True)
            all_lines.append(f"[final] {new}")

    total = time.time() - t_start
    print(f"\nTotal: {total:.1f}s | Fed: {fed} chunks, Skipped: {skipped}", flush=True)

    if args.output and all_lines:
        Path(args.output).write_text("\n".join(all_lines), encoding="utf-8")
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
