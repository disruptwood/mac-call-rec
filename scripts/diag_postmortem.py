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


def parse_timecode(value: str) -> float | None:
    """Parse an ffmpeg HH:MM:SS.xx timecode into seconds."""
    m = re.fullmatch(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", value)
    if not m:
        return None
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2))
    seconds = float(m.group(3))
    return hours * 3600 + minutes * 60 + seconds


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

    # Input format line. ffmpeg also prints output stream lines with the same
    # "Stream #0" shape, so only accept stream lines while inside the input
    # section.
    in_input_section = False
    for line in content.splitlines()[:120]:
        if line.startswith("Input #0"):
            in_input_section = True
            m = re.search(r"Audio:\s*(.+)$", line)
            if m:
                info["input_format"] = m.group(1).strip()
                break
            continue
        if line.startswith("Stream mapping:") or line.startswith("Output #0"):
            in_input_section = False
        if in_input_section:
            m = re.search(r"^\s*Stream #0.*?Audio:\s*(.+)$", line)
            if m:
                info["input_format"] = m.group(1).strip()
                break

    times = []
    elapsed_times = []
    for m in re.finditer(r"\btime=(\d+:\d{2}:\d{2}(?:\.\d+)?)", content):
        parsed = parse_timecode(m.group(1))
        if parsed is not None:
            times.append(parsed)
    for m in re.finditer(r"\belapsed=(\d+:\d{2}:\d{2}(?:\.\d+)?)", content):
        parsed = parse_timecode(m.group(1))
        if parsed is not None:
            elapsed_times.append(parsed)
    if times:
        info["ffmpeg_time_seconds"] = times[-1]
    if elapsed_times:
        info["ffmpeg_elapsed_seconds"] = elapsed_times[-1]

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


def first_existing_log(session_dir: Path, final_name: str, segment_glob: str) -> tuple[Path, str | None]:
    final_log = session_dir / final_name
    if final_log.exists():
        return final_log, None

    seg_logs = sorted(session_dir.glob(segment_glob))
    if seg_logs:
        return seg_logs[0], seg_logs[0].name

    return final_log, None


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
    mic_pa = session_dir / "_mic_pa.wav"
    sys_f = session_dir / "_system.wav"
    rec = session_dir / "recording.m4a"

    results = {}
    for name, path in [("mic", mic), ("mic_pa", mic_pa), ("system", sys_f), ("recording.m4a", rec)]:
        if not path.exists():
            if name == "mic_pa":
                continue  # mic_pa is opt-in A/B track, missing is fine
            print(f"{name:16s} — MISSING")
            continue
        data = probe(path)
        stream = data.get("streams", [{}])[0] if data.get("streams") else {}
        fmt = data.get("format", {})
        raw_dur = fmt.get("duration", stream.get("duration", 0))
        try:
            dur = float(raw_dur)
        except (TypeError, ValueError):
            # ffprobe returns "N/A" for corrupt/unfinished WAVs (broken RIFF
            # header from SIGKILL'd writer, etc.).
            print(f"{name:16s} — duration unreadable (raw={raw_dur!r}); skipping in drift calc")
            continue
        sr = stream.get("sample_rate", "?")
        ch = stream.get("channels", "?")
        codec = stream.get("codec_name", "?")
        print(f"{name:16s} dur={fmt_duration(dur):20s} sr={sr} Hz ch={ch} codec={codec}")
        results[name] = dur

    # Drift analysis
    def _drift_report(track_name: str, track_dur: float, sys_dur: float) -> None:
        if track_dur <= 0 or sys_dur <= 0:
            print(f"\n{track_name}/system: one or both durations are zero "
                  f"({track_name}={track_dur:.2f}s, system={sys_dur:.2f}s) — "
                  f"likely a capture crash at session start.")
            return
        drift_pct = abs(track_dur - sys_dur) / max(track_dur, sys_dur) * 100
        ratio = track_dur / sys_dur
        print(f"\n{track_name}/system ratio: {ratio:.3f}")
        print(f"{track_name} drift vs system: {drift_pct:.1f}%")
        if drift_pct < 2:
            print(f"✓ {track_name} is SYNCHRONIZED with system")
        elif drift_pct < 5:
            print(f"? {track_name} has MARGINAL drift")
        else:
            print(f"✗ {track_name} DESYNC — sync bug triggered on this track")
            effective_rate = track_dur / sys_dur * 48000
            print(f"  Effective {track_name} sample rate: ~{effective_rate:.0f} Hz (expected 48000)")

    def _safe_drift(a: float, b: float) -> float | None:
        if a <= 0 or b <= 0:
            return None
        return abs(a - b) / max(a, b) * 100

    if "mic" in results and "system" in results:
        _drift_report("mic", results["mic"], results["system"])

    if "mic_pa" in results and "system" in results:
        _drift_report("mic_pa", results["mic_pa"], results["system"])

    # Side-by-side verdict for the A/B test
    if "mic" in results and "mic_pa" in results and "system" in results:
        mic_drift = _safe_drift(results["mic"], results["system"])
        pa_drift = _safe_drift(results["mic_pa"], results["system"])
        print("\n--- A/B verdict (ffmpeg avfoundation vs PortAudio) ---")
        if mic_drift is None or pa_drift is None:
            print(f"  Cannot compare — durations: mic={results['mic']:.1f}s, "
                  f"mic_pa={results['mic_pa']:.1f}s, system={results['system']:.1f}s. "
                  f"Likely a capture crash. Check the *.log files for errors.")
        else:
            print(f"  ffmpeg   _mic.wav    drift: {mic_drift:.2f}%")
            print(f"  PortAudio _mic_pa.wav drift: {pa_drift:.2f}%")
            if pa_drift < 2 and mic_drift >= 2:
                # PortAudio is clean and ffmpeg is at least marginally broken.
                print("  ✓ PortAudio path is CLEAN, ffmpeg path is drifting — switch over.")
            elif pa_drift < mic_drift - 1:
                print(f"  PortAudio is better by {mic_drift - pa_drift:.1f}pp, but not yet clean.")
            elif abs(pa_drift - mic_drift) < 0.5:
                print("  Both tracks behave the same — VPIO may not be active, or bypass failed.")
            else:
                print("  ⚠ PortAudio is NOT better. Investigate before relying on it.")

    # ffmpeg logs — check both post-concat name and segment name (logs are written
    # with segment name before concat; concat renames the .wav but leaves the .log)
    print("\n--- ffmpeg log (_mic.wav.ffmpeg.log) ---")
    mic_log, mic_segment_log = first_existing_log(
        session_dir, "_mic.wav.ffmpeg.log", "_mic_seg*.wav.ffmpeg.log",
    )
    if mic_segment_log:
        print(f"  (using segment log: {mic_segment_log})")
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
        if "ffmpeg_time_seconds" in info:
            print(f"  ffmpeg timestamp duration: {fmt_duration(info['ffmpeg_time_seconds'])}")
            if "mic" in results:
                sample_dur = results["mic"]
                ts_dur = info["ffmpeg_time_seconds"]
                mismatch_pct = abs(ts_dur - sample_dur) / max(ts_dur, sample_dur) * 100
                if mismatch_pct >= 2:
                    print(f"  ⚠ WAV sample duration differs from ffmpeg timestamps by {mismatch_pct:.1f}%")
                    print("    Mic capture needs timestamp-gap compensation (aresample=async).")

    # PortAudio mic log (if A/B track was recorded)
    if mic_pa.exists():
        print("\n--- capture_mic_pa log (_mic_pa.wav.pa.log) ---")
        pa_log, pa_segment_log = first_existing_log(
            session_dir, "_mic_pa.wav.pa.log", "_mic_pa_seg*.wav.pa.log",
        )
        if pa_segment_log:
            print(f"  (using segment log: {pa_segment_log})")
        if not pa_log.exists():
            print("  No PortAudio log found — script may have crashed before logging")
        else:
            content = pa_log.read_text(errors="replace")
            lines = content.splitlines()
            print(f"  Log size: {pa_log.stat().st_size} bytes")
            if lines:
                # Show header (device info) and tail (final drift report)
                for line in lines[:8]:
                    print(f"    {line}")
                if len(lines) > 16:
                    print("    ...")
                for line in lines[-8:]:
                    print(f"    {line}")
            # Highlight any status events (overflows etc.)
            status_lines = [l for l in lines if "Status events" in l or "Stream error" in l]
            for sl in status_lines:
                print(f"  ⚠ {sl}")

    # Capture log
    print("\n--- capture_system_audio log (_system.wav.capture.log) ---")
    sys_log, sys_segment_log = first_existing_log(
        session_dir, "_system.wav.capture.log", "_system_seg*.wav.capture.log",
    )
    if sys_segment_log:
        print(f"  (using segment log: {sys_segment_log})")
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
