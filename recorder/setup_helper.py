"""Helpers for setting up macOS audio routing (Multi-Output Device via coreaudio)."""

from __future__ import annotations

import subprocess


def check_multi_output_device() -> bool:
    """Check if a Multi-Output Device exists that includes BlackHole."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPAudioDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "multi-output" in result.stdout.lower()
    except Exception:
        return False


def print_setup_instructions(headphones: bool) -> None:
    """Print instructions for setting up audio routing."""
    output_device = "your headphones" if headphones else "MacBook Air Speakers"

    print("""
=== Audio Setup Instructions ===

To record the other person's audio during a call, you need a
Multi-Output Device that sends sound to both {output} AND BlackHole 2ch.

Steps:
1. Open "Audio MIDI Setup" (Spotlight → "Audio MIDI Setup")
2. Click "+" at the bottom-left → "Create Multi-Output Device"
3. Check the boxes for:
   - {output}
   - BlackHole 2ch
4. Make sure "Drift Correction" is checked for BlackHole 2ch
5. Right-click the Multi-Output Device → "Use This Device For Sound Output"

When done, your call app will send audio through both {output} (so you hear it)
and BlackHole (so the recorder captures it).

Your microphone stays as "MacBook Air Microphone" — no changes needed there.

TIP: You can rename the Multi-Output Device by double-clicking its name.
     Call it something like "Record + {short_output}".

=== End Setup ===
""".format(
        output=output_device,
        short_output="Headphones" if headphones else "Speakers",
    ))


def check_blackhole_loaded() -> bool:
    """Check if BlackHole appears as a usable audio device."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "blackhole" in result.stderr.lower()
    except Exception:
        return False


def check_ffmpeg_installed() -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except FileNotFoundError:
        return False
