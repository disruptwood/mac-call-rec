#!/usr/bin/env python3
"""POSTMORTEM diagnostic for a recorded session.

Analyzes a session directory to diagnose recording sync issues:
  - File durations (mic vs system vs recording.m4a)
  - Sample rates from ffprobe
  - ffmpeg log (if present from new recording.py): input format, warnings, drops
  - Drift calculation
  - Identifies whether ffmpeg saw a non-48kHz input (WebRTC/VPIO sign)

Run this after ANY session to see if recording was healthy.

Usage:
    python3 diag_postmortem.py <session_dir>
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def probe(path: Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=sample_rate,channels,codec_name,duration",
         "-show_entries", "format=duration,bit_rate",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        return json.loads(r.stdout)
    except Exception:
        return {}


def fmt_duration(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d} ({sec:.0f}s)"
    return f"{m}:{s:02d} ({sec:.0f}s)"


def analyze_ffmpeg_log(log_path: Path) -> dict:
    """Extract key diagnostics from ffmpeg stderr log."""
    if not log_path.exists():
        return {"present": False}

    content = log_path.read_text(errors="replace")
    info = {"present": True, "size": log_path.stat().st_size}

    # Input format line
    for line in content.splitlines()[:100]:
        m = re.search(r"Input #0.*?Audio:\s*(.+)$", line)
        if m:
            info["input_format"] = m.group(1).strip()
            break
        m = re.search(r"^\s*Stream #0.*?Audio:\s*(.+)$", line)
        if m:
            info["input_format"] = m.group(1).strip()

    # Warnings / errors
    warnings = []
    for line in content.splitlines():
        l = line.lower()
        if any(kw in l for kw in ["warning", "error", "overrun", "drop", "queue size",
                                    "non monotonous", "past duration"]):
            if len(warnings) < 20:
                warnings.append(line.strip())
    info["warnings"] = warnings

    # Final output stats (last N lines)
    info["tail"] = "\n".join(content.splitlines()[-20:])
    return info


def main():
    if len(sys.argv) < 2:
        print("Usage: diag_postmortem.py <session_dir>")
        sys.exit(1)

    session_dir = Path(sys.argv[1])
    if not session_dir.is_dir():
        print(f"ERROR: {session_dir} not a directory")
        sys.exit(1)

    print(f"\n=== Postmortem: {session_dir.name} ===\n")

    # Load manifest
    manifest_path = session_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"Manifest says: {manifest.get('duration_seconds', '?'):.1f}s wall-clock")
        print(f"Started: {manifest.get('started_at', '?')}")
        print(f"Headphones connected: {manifest.get('headphones_connected', '?')}")
        print("")

    # Check files
    mic = session_dir / "_mic.wav"
    sys_f = session_dir / "_system.wav"
    rec = session_dir / "recording.m4a"

    results = {}
    for name, path in [("mic", mic), ("system", sys_f), ("recording.m4a", rec)]:
        if not path.exists():
            print(f"{name:16s} — MISSING")
            continue
        data = probe(path)
        stream = data.get("streams", [{}])[0] if data.get("streams") else {}
        fmt = data.get("format", {})
        dur = float(fmt.get("duration", stream.get("duration", 0)))
        sr = stream.get("sample_rate", "?")
        ch = stream.get("channels", "?")
        codec = stream.get("codec_name", "?")
        print(f"{name:16s} dur={fmt_duration(dur):20s} sr={sr} Hz ch={ch} codec={codec}")
        results[name] = dur

    # Drift analysis
    if "mic" in results and "system" in results:
        mic_dur, sys_dur = results["mic"], results["system"]
        drift_pct = abs(mic_dur - sys_dur) / max(mic_dur, sys_dur) * 100
        ratio = mic_dur / sys_dur
        print(f"\nmic/system ratio: {ratio:.3f}")
        print(f"Drift: {drift_pct:.1f}%")
        if drift_pct < 2:
            print("✓ Tracks are SYNCHRONIZED — recording was healthy")
        elif drift_pct < 5:
            print("? MARGINAL drift")
        else:
            print("✗ DESYNC — known sync bug triggered")
            effective_rate = mic_dur / sys_dur * 48000
            print(f"  Effective mic sample rate: ~{effective_rate:.0f} Hz (expected 48000)")

    # ffmpeg logs
    print("\n--- ffmpeg log (_mic.wav.ffmpeg.log) ---")
    mic_log = session_dir / "_mic.wav.ffmpeg.log"
    info = analyze_ffmpeg_log(mic_log)
    if not info["present"]:
        print("  Log not present — session was recorded with OLD version of recording.py")
        print("  (Upgrade recording.py already done; next session will have the log.)")
    else:
        print(f"  Log size: {info['size']} bytes")
        if "input_format" in info:
            print(f"  ffmpeg saw mic input: {info['input_format']}")
            if "48000" in info["input_format"] or "48 kHz" in info["input_format"]:
                print("  ✓ Input was 48kHz as expected")
            else:
                print("  ⚠ Input was NOT 48kHz — WebRTC/VPIO likely interfered")
        else:
            print("  No Input format line found in log — check log manually")
        if info["warnings"]:
            print(f"  Warnings/errors ({len(info['warnings'])}):")
            for w in info["warnings"][:10]:
                print(f"    {w}")

    # Capture log
    print("\n--- capture_system_audio log (_system.wav.capture.log) ---")
    sys_log = session_dir / "_system.wav.capture.log"
    if not sys_log.exists():
        print("  Log not present — recorded with OLD version; next session will have it")
    else:
        print(f"  Log size: {sys_log.stat().st_size} bytes")
        content = sys_log.read_text(errors="replace")
        # Show first 10 + last 10 lines
        lines = content.splitlines()
        if lines:
            print("  First 10 lines:")
            for l in lines[:10]:
                print(f"    {l}")
            if len(lines) > 20:
                print("  Last 10 lines:")
                for l in lines[-10:]:
                    print(f"    {l}")

    print("\n=== End postmortem ===\n")


if __name__ == "__main__":
    main()
