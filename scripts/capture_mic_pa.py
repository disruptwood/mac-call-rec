#!/usr/bin/env python3
"""Record default mic to WAV using sounddevice (PortAudio → CoreAudio HAL).

Bypasses AVFoundation/VPIO. Used in parallel with ffmpeg avfoundation as A/B
test against the WebRTC drift bug. PortAudio on macOS talks directly to the
CoreAudio HAL layer (AudioDeviceCreateIOProcID) — the same path used by
Audacity, REAPER, sox. VPIO mode-switch from active WebRTC mic capture in a
browser does not affect this path.

Usage:
    python3 capture_mic_pa.py <output.wav> [duration_seconds]

If duration is omitted, records until SIGTERM/SIGINT.
"""

from __future__ import annotations

import argparse
import queue
import signal
import sys
import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 48000
CHANNELS = 1
SUBTYPE = "PCM_16"
DTYPE = "int16"
BLOCKSIZE = 1024  # ~21 ms at 48 kHz


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def diagnostic_dump() -> None:
    try:
        ver = sd.get_portaudio_version()
        log(f"PortAudio: {ver[1]}")
    except Exception as e:
        log(f"PortAudio version query failed: {e}")
    try:
        apis = [a["name"] for a in sd.query_hostapis()]
        log(f"Host APIs: {apis}")
    except Exception as e:
        log(f"Host APIs query failed: {e}")
    try:
        d = sd.query_devices(kind="input")
        log(
            f"Default input: name={d['name']!r} "
            f"native_sr={d['default_samplerate']:.0f} "
            f"max_in_ch={d['max_input_channels']} "
            f"hostapi={d['hostapi']}"
        )
    except Exception as e:
        log(f"Default input query failed: {e}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="Output WAV path")
    parser.add_argument(
        "duration",
        nargs="?",
        type=float,
        default=None,
        help="Duration in seconds (default: until SIGTERM/SIGINT)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device name substring or index (default: system default input)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    diagnostic_dump()

    stop_flag = {"stop": False}

    def handle_signal(signum, frame):
        log(f"Received signal {signum}")
        stop_flag["stop"] = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    q: queue.Queue = queue.Queue()
    status_events: list[str] = []

    def audio_callback(indata, frames, time_info, status):
        if status:
            status_events.append(str(status))
        q.put(indata.copy())

    log(
        f"Opening InputStream: sr={SAMPLE_RATE} ch={CHANNELS} dtype={DTYPE} "
        f"blocksize={BLOCKSIZE} device={args.device!r}"
    )
    start_t = time.monotonic()
    frames_written = 0

    try:
        with sf.SoundFile(
            str(output_path),
            mode="w",
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            subtype=SUBTYPE,
        ) as wf, sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCKSIZE,
            device=args.device,
            callback=audio_callback,
        ):
            log(f"Stream started; writing to {output_path}")
            while not stop_flag["stop"]:
                if args.duration is not None and (time.monotonic() - start_t) >= args.duration:
                    break
                try:
                    chunk = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                wf.write(chunk)
                frames_written += len(chunk)
            log("Stopping, draining queue...")
            while True:
                try:
                    chunk = q.get_nowait()
                except queue.Empty:
                    break
                wf.write(chunk)
                frames_written += len(chunk)
    except Exception as e:
        log(f"Stream error: {e}")
        return 1

    elapsed = time.monotonic() - start_t
    size_kb = output_path.stat().st_size // 1024 if output_path.exists() else 0
    sample_seconds = frames_written / SAMPLE_RATE
    log(
        f"Done. wall_clock={elapsed:.2f}s "
        f"frames={frames_written} sample_seconds={sample_seconds:.2f}s "
        f"size={size_kb}KB"
    )
    if status_events:
        log(f"Status events ({len(status_events)}): first 5 = {status_events[:5]}")
    if elapsed > 0:
        drift_pct = abs(sample_seconds - elapsed) / max(sample_seconds, elapsed) * 100
        log(f"sample_seconds vs wall_clock drift: {drift_pct:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
