#!/usr/bin/env python3
"""PREFLIGHT diagnostic for recording sync bug (before real session).

Runs a 2-min baseline recording to check if mic/system tracks stay in sync
in current conditions. Reports drift, sample rate, and whether ffmpeg saw
the mic in a non-48kHz format (sign of WebRTC/VPIO interference).

Run this:
  1. WITHOUT any browser call active — expect synced output
  2. THEN optionally open a browser call and run again — compare

Usage:
    python3 diag_preflight.py [--duration 120]

Outputs files in /tmp/preflight_<timestamp>/ and prints a summary.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CAPTURE_BINARY = PROJECT_ROOT / "scripts" / "capture_system_audio"


def find_mic_device_index() -> int | None:
    r = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    )
    text = r.stderr
    # Find audio devices section, look for built-in mic
    in_audio = False
    for line in text.splitlines():
        if "AVFoundation audio devices" in line:
            in_audio = True
            continue
        if not in_audio:
            continue
        m = re.search(r"\[(\d+)\].*?(MacBook.*Microphone|Built-in Microphone|Microphone)", line, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def parse_ffmpeg_input_format(log_path: Path) -> str:
    """Parse ffmpeg log to find 'Input #0 ... Audio:' line."""
    if not log_path.exists():
        return "log missing"
    content = log_path.read_text(errors="replace")
    in_input_section = False
    for line in content.splitlines()[:80]:
        if line.startswith("Input #0"):
            in_input_section = True
            m = re.search(r"Audio:\s*(.+)$", line)
            if m:
                return m.group(1).strip()
            continue
        if line.startswith("Stream mapping:") or line.startswith("Output #0"):
            in_input_section = False
        if in_input_section:
            m = re.search(r"^\s*Stream #0.*?Audio:\s*(.+)$", line)
            if m:
                return m.group(1).strip()
    return "format line not found"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=120, help="Recording duration in seconds")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"/tmp/preflight_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    mic_path = out_dir / "mic.wav"
    sys_path = out_dir / "sys.wav"
    mic_log = out_dir / "mic.ffmpeg.log"
    sys_log = out_dir / "sys.capture.log"

    mic_idx = find_mic_device_index()
    if mic_idx is None:
        print("ERROR: could not find MacBook mic index via ffmpeg")
        print("  Run manually: ffmpeg -f avfoundation -list_devices true -i ''")
        sys.exit(1)

    if not CAPTURE_BINARY.exists():
        print(f"ERROR: {CAPTURE_BINARY} not found. Build it first.")
        sys.exit(1)

    print(f"Mic device index: {mic_idx}")
    print(f"Duration: {args.duration}s")
    print(f"Output: {out_dir}")
    print("")
    print("Starting recordings in parallel. Please speak into mic and play some")
    print("audio (YouTube/music) so both tracks have content.")
    print("")

    # Start mic ffmpeg with -t duration (auto-stop). stdout+stderr → log file.
    mic_cmd = [
        "ffmpeg",
        "-f", "avfoundation",
        "-i", f":{mic_idx}",
        "-af", "aresample=async=1:first_pts=0",
        "-ar", "48000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-t", str(args.duration),
        "-y", str(mic_path),
    ]
    mic_log_file = open(mic_log, "wb")
    mic_proc = subprocess.Popen(
        mic_cmd, stdin=subprocess.PIPE, stdout=mic_log_file, stderr=mic_log_file,
    )

    # Start system capture
    sys_log_file = open(sys_log, "wb")
    sys_proc = subprocess.Popen(
        [str(CAPTURE_BINARY), str(sys_path)],
        stdin=subprocess.PIPE, stdout=sys_log_file, stderr=sys_log_file,
    )

    time.sleep(0.5)
    if mic_proc.poll() is not None:
        print("ERROR: mic ffmpeg died immediately")
        sys.exit(1)
    if sys_proc.poll() is not None:
        print("ERROR: system capture died immediately")
        sys.exit(1)

    print(f"Recording {args.duration}s... (Ctrl+C to abort early)")
    t0 = time.time()
    try:
        while time.time() - t0 < args.duration:
            time.sleep(1)
            elapsed = time.time() - t0
            if int(elapsed) % 10 == 0:
                sys.stdout.write(f"\r  elapsed: {elapsed:.0f}s")
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    print("")

    # Stop both
    if sys_proc.poll() is None:
        sys_proc.send_signal(15)  # SIGTERM
        try:
            sys_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sys_proc.kill()
    # mic ffmpeg should stop on its own via -t flag, but ensure
    try:
        mic_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        mic_proc.kill()

    mic_log_file.close()
    sys_log_file.close()

    # Analyze
    print("\n=== Results ===")
    mic_dur = probe_duration(mic_path)
    sys_dur = probe_duration(sys_path)
    print(f"mic.wav duration: {mic_dur:.2f}s")
    print(f"sys.wav duration: {sys_dur:.2f}s")

    if mic_dur > 0 and sys_dur > 0:
        drift = abs(mic_dur - sys_dur) / max(mic_dur, sys_dur) * 100
        ratio = mic_dur / sys_dur
        print(f"Drift: {drift:.1f}% (mic/sys ratio = {ratio:.3f})")
        if drift < 2:
            print("✓ SYNCHRONIZED — recording pipeline is healthy in current conditions")
        elif drift < 5:
            print("? MARGINAL — small drift, might be clock skew")
        else:
            print("✗ DESYNC CONFIRMED — drift is significant, matches bug seen in real sessions")

    mic_input = parse_ffmpeg_input_format(mic_log)
    print(f"\nffmpeg saw mic input as: {mic_input}")
    if "48000 Hz" not in mic_input and "48 kHz" not in mic_input:
        print("  ⚠ Mic input is NOT 48kHz — likely WebRTC/VPIO is holding mic in reduced mode")

    print(f"\nFull logs: {out_dir}")
    print("  Inspect: cat", mic_log)


if __name__ == "__main__":
    main()
