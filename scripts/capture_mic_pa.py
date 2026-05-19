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

Reliability notes:
  - The audio→file queue is bounded so a disk stall can't OOM the process
    during a long session. When the queue is full we drop incoming frames
    and increment a counter rather than block the PortAudio callback.
  - The PortAudio callback is kept tight — no string formatting or list
    allocation. Status flags are accumulated as bitmasks and reported once
    at exit.
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

# Bounded queue between PortAudio callback and the file-writer thread.
# At 48 kHz / 1024 frames per block we see ~47 blocks per second; 200 slots is
# ~4.3 seconds of audio — long enough to ride out fsync hiccups, short enough
# that we don't accumulate hundreds of MB of RAM if the disk stalls for real.
MAX_QUEUE_BLOCKS = 200


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


def resolve_input_device(name_substring: str | None) -> int | None:
    """Resolve a device name substring to a unique input device index.

    sounddevice's default name matching is permissive and can match outputs or
    multiple inputs. We require exactly one INPUT device whose name contains
    the substring (case-insensitive). Returns the device index, or None to
    fall through to PortAudio's default.
    """
    if not name_substring:
        return None
    matches: list[tuple[int, dict]] = []
    try:
        devices = sd.query_devices()
    except Exception as e:
        log(f"query_devices failed: {e}; falling back to default device")
        return None
    needle = name_substring.lower()
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) <= 0:
            continue
        if needle in d.get("name", "").lower():
            matches.append((idx, d))
    if not matches:
        log(f"No input device matching {name_substring!r}; using default")
        return None
    if len(matches) > 1:
        log(
            f"Multiple input devices match {name_substring!r}: "
            f"{[m[1]['name'] for m in matches]}; picking first"
        )
    idx, d = matches[0]
    log(f"Resolved device {name_substring!r} → idx={idx} name={d['name']!r}")
    return idx


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
        help="Device name substring (default: system default input)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    diagnostic_dump()
    device_idx = resolve_input_device(args.device)

    stop_flag = {"stop": False}

    def handle_signal(signum, frame):
        log(f"Received signal {signum}")
        stop_flag["stop"] = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    q: queue.Queue = queue.Queue(maxsize=MAX_QUEUE_BLOCKS)

    # Counters maintained from the PortAudio audio thread. Plain int +=
    # operations are not strictly atomic in CPython, but the worst-case error
    # is one missed increment per pre-emption — totally acceptable for these
    # diagnostic counters.
    counters = {
        "input_overflow": 0,
        "input_underflow": 0,
        "priming_output": 0,
        "queue_full_drops": 0,
    }

    def audio_callback(indata, frames, time_info, status):
        if status:
            if status.input_overflow:
                counters["input_overflow"] += 1
            if status.input_underflow:
                counters["input_underflow"] += 1
            if status.priming_output:
                counters["priming_output"] += 1
        try:
            q.put_nowait(indata.copy())
        except queue.Full:
            counters["queue_full_drops"] += 1

    log(
        f"Opening InputStream: sr={SAMPLE_RATE} ch={CHANNELS} dtype={DTYPE} "
        f"blocksize={BLOCKSIZE} device_idx={device_idx}"
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
            device=device_idx,
            callback=audio_callback,
        ) as stream:
            log(
                f"Stream started; actual_sr={stream.samplerate} "
                f"writing to {output_path}"
            )
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
            # Bound the drain so an out-of-control queue can't keep us past the
            # parent's SIGKILL deadline. At max queue depth this completes in
            # well under a second on any modern SSD.
            drain_deadline = time.monotonic() + 5.0
            while time.monotonic() < drain_deadline:
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
    log(
        f"Counters: input_overflow={counters['input_overflow']} "
        f"input_underflow={counters['input_underflow']} "
        f"priming_output={counters['priming_output']} "
        f"queue_full_drops={counters['queue_full_drops']}"
    )
    if elapsed > 0:
        drift_pct = abs(sample_seconds - elapsed) / max(sample_seconds, elapsed) * 100
        log(f"sample_seconds vs wall_clock drift: {drift_pct:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
