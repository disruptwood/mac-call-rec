#!/usr/bin/env python3
"""Live streaming transcription from growing WAV files during recording.

Reads mic and system WAV files as they grow during call-recorder session.
Runs VAD on both tracks for speaker labels.
Feeds mixed PCM to Qwen3-ASR streaming API.
Prints text with speaker labels in real-time.

Usage: python3 live_stream_transcribe.py <session_dir> [--language Russian]

Start call-recorder first, then run this pointing to the session directory.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

QWEN_MODEL = "/Users/ilya/models/qwen3-asr-1.7b"
SAMPLE_RATE = 16000
CHUNK_SEC = 2
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SEC
VAD_INTERVAL_SEC = 10  # Re-run VAD every N seconds


def read_wav_tail(path: Path, offset: int) -> tuple[np.ndarray, int]:
    """Read new PCM data from a growing WAV file starting at byte offset.
    Returns (pcm_float32_mono_16k, new_offset)."""
    if not path.exists():
        return np.array([], dtype=np.float32), offset

    size = path.stat().st_size
    if size <= offset + 100:
        return np.array([], dtype=np.float32), offset

    with open(path, "rb") as f:
        # Skip WAV header on first read
        if offset == 0:
            f.seek(0)
            header = f.read(44)
            if len(header) < 44 or header[:4] != b"RIFF":
                return np.array([], dtype=np.float32), 0
            offset = 44

        f.seek(offset)
        data = f.read(size - offset)

    if not data:
        return np.array([], dtype=np.float32), offset

    new_offset = offset + len(data)

    # System WAV from ScreenCaptureKit: PCM int16 stereo 48kHz
    # Try to detect format from header if first read
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    # If stereo, convert to mono
    if len(audio) > 1 and len(audio) % 2 == 0:
        # Check if stereo by looking at WAV header channels
        # For simplicity: system files are stereo 48kHz, mic files are mono
        if "system" in path.name.lower():
            audio = audio.reshape(-1, 2).mean(axis=1)
            # Downsample 48k → 16k (decimate by 3)
            audio = audio[::3]

    return audio.astype(np.float32), new_offset


def run_vad_incremental(mic_pcm: np.ndarray, sys_pcm: np.ndarray, vad_model, get_stamps):
    """Run VAD on accumulated PCM buffers. Returns (mic_stamps, sys_stamps) in seconds."""
    import torch
    mic_stamps = []
    sys_stamps = []

    if len(mic_pcm) > SAMPLE_RATE:  # At least 1 sec
        raw = get_stamps(torch.tensor(mic_pcm), vad_model, sampling_rate=SAMPLE_RATE)
        mic_stamps = [(float(s["start"]) / SAMPLE_RATE, float(s["end"]) / SAMPLE_RATE) for s in raw]

    if len(sys_pcm) > SAMPLE_RATE:
        raw = get_stamps(torch.tensor(sys_pcm), vad_model, sampling_rate=SAMPLE_RATE)
        sys_stamps = [(float(s["start"]) / SAMPLE_RATE, float(s["end"]) / SAMPLE_RATE) for s in raw]

    return mic_stamps, sys_stamps


def get_speaker(audio_sec: float, mic_stamps: list, sys_stamps: list) -> str:
    """Determine speaker at given timestamp based on VAD."""
    in_mic = any(s <= audio_sec <= e for s, e in mic_stamps)
    in_sys = any(s <= audio_sec <= e for s, e in sys_stamps)

    if in_mic and not in_sys:
        return "Я"
    elif in_sys and not in_mic:
        return "Собеседни:ца"
    elif in_mic and in_sys:
        return "Я"  # Both speaking — attribute to user
    return ""


def find_session_files(session_dir: Path) -> tuple[Path | None, Path | None]:
    """Find mic and system WAV files in session directory."""
    mic = None
    sys_f = None
    for f in session_dir.iterdir():
        name = f.name.lower()
        if "mic" in name and (name.endswith(".wav") or name.endswith(".m4a")):
            mic = f
        if "system" in name and name.endswith(".wav"):
            sys_f = f
    return mic, sys_f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir")
    parser.add_argument("--language", default="Russian")
    parser.add_argument("--model", default=QWEN_MODEL)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        print(f"ERROR: {session_dir} not found")
        sys.exit(1)

    # Load VAD
    import torch
    print("Loading VAD...", flush=True)
    vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    get_stamps = utils[0]

    # Load Qwen3-ASR
    print(f"Loading Qwen3-ASR ({Path(args.model).name})...", flush=True)
    from mlx_qwen3_asr import Session
    session = Session(model=args.model)
    print("Models loaded. Waiting for audio files...\n", flush=True)

    # Wait for files to appear
    mic_path = None
    sys_path = None
    for _ in range(30):
        mic_path, sys_path = find_session_files(session_dir)
        if sys_path:
            break
        time.sleep(1)

    if not sys_path:
        print("ERROR: No system WAV found in session directory")
        sys.exit(1)

    print(f"Mic: {mic_path.name if mic_path else 'not found'}", flush=True)
    print(f"System: {sys_path.name}", flush=True)

    # Streaming state
    state = session.init_streaming(language=args.language, chunk_size_sec=float(CHUNK_SEC))
    prev_text = ""
    prev_speaker = ""
    all_lines = []

    # File read offsets
    mic_offset = 0
    sys_offset = 0

    # Accumulated PCM for VAD
    all_mic_pcm = np.array([], dtype=np.float32)
    all_sys_pcm = np.array([], dtype=np.float32)

    # VAD results
    mic_stamps = []
    sys_stamps = []
    last_vad_time = 0
    audio_pos = 0.0

    stop = False

    def handle_signal(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    t_start = time.time()

    while not stop:
        # Read new data from growing files
        sys_new, sys_offset = read_wav_tail(sys_path, sys_offset)

        mic_new = np.array([], dtype=np.float32)
        if mic_path and mic_path.exists():
            mic_new_raw, mic_offset = read_wav_tail(mic_path, mic_offset)
            mic_new = mic_new_raw

        if len(sys_new) < CHUNK_SAMPLES and len(mic_new) < CHUNK_SAMPLES:
            # No new data — check if recording finished
            manifest = session_dir / "manifest.json"
            if manifest.exists():
                try:
                    m = json.loads(manifest.read_text())
                    if "recording" in m:
                        # Recording finished — process remaining
                        time.sleep(1)
                        sys_new, sys_offset = read_wav_tail(sys_path, sys_offset)
                        if mic_path:
                            mic_new, mic_offset = read_wav_tail(mic_path, mic_offset)
                        if len(sys_new) < 100:
                            break
                except Exception:
                    pass
            time.sleep(0.5)
            continue

        # Accumulate for VAD
        if len(mic_new) > 0:
            all_mic_pcm = np.concatenate([all_mic_pcm, mic_new])
        if len(sys_new) > 0:
            all_sys_pcm = np.concatenate([all_sys_pcm, sys_new])

        # Periodic VAD update
        elapsed = time.time() - t_start
        if elapsed - last_vad_time > VAD_INTERVAL_SEC:
            mic_stamps, sys_stamps = run_vad_incremental(all_mic_pcm, all_sys_pcm, vad_model, get_stamps)
            last_vad_time = elapsed

        # Mix and feed to Qwen3-ASR in chunks
        max_len = max(len(sys_new), len(mic_new))
        if max_len < CHUNK_SAMPLES:
            continue

        # Pad shorter to match
        if len(sys_new) < max_len:
            sys_new = np.pad(sys_new, (0, max_len - len(sys_new)))
        if len(mic_new) < max_len:
            mic_new = np.pad(mic_new, (0, max_len - len(mic_new)))

        # Feed in CHUNK_SAMPLES increments
        for i in range(0, max_len, CHUNK_SAMPLES):
            chunk_sys = sys_new[i:i + CHUNK_SAMPLES]
            chunk_mic = mic_new[i:i + CHUNK_SAMPLES]

            if len(chunk_sys) < CHUNK_SAMPLES:
                chunk_sys = np.pad(chunk_sys, (0, CHUNK_SAMPLES - len(chunk_sys)))
            if len(chunk_mic) < CHUNK_SAMPLES:
                chunk_mic = np.pad(chunk_mic, (0, CHUNK_SAMPLES - len(chunk_mic)))

            mixed = (chunk_mic * 0.5 + chunk_sys * 0.5).astype(np.float32)
            state = session.feed_audio(mixed, state)
            audio_pos += CHUNK_SEC

            text = state.text if hasattr(state, "text") else ""
            if text and text != prev_text:
                new = text[len(prev_text):].strip()
                if new:
                    speaker = get_speaker(audio_pos, mic_stamps, sys_stamps)
                    m, s = divmod(int(audio_pos), 60)
                    ts = f"{m:02d}:{s:02d}"

                    if speaker and speaker != prev_speaker:
                        print(f"\n{speaker}  {ts}", flush=True)
                        all_lines.append(f"\n{speaker}  {ts}")
                        prev_speaker = speaker

                    print(new, flush=True)
                    all_lines.append(new)
                prev_text = text

    # Finalize
    state = session.finish_streaming(state)
    text = state.text if hasattr(state, "text") else ""
    if text and text != prev_text:
        new = text[len(prev_text):].strip()
        if new:
            print(f"\n{new}", flush=True)
            all_lines.append(new)

    total = time.time() - t_start
    print(f"\nTotal: {total:.1f}s, audio: {audio_pos:.0f}s ({audio_pos / total:.1f}x real-time)", flush=True)

    if args.output and all_lines:
        Path(args.output).write_text("\n".join(all_lines), encoding="utf-8")
        print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
