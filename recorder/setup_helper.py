"""Helpers for checking that the macOS environment is ready to record.

The recorder needs:
  - ffmpeg (for the avfoundation mic track)
  - Screen Recording permission for the terminal/IDE that launches the recorder
    (ScreenCaptureKit refuses to start otherwise)
  - Microphone permission for the same binary, plus for the Python interpreter
    that runs the PortAudio mic track

These checks are observational only — they print guidance and never modify
system state.
"""

from __future__ import annotations

import subprocess


def check_ffmpeg_installed() -> bool:
    """Check if ffmpeg is available on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except FileNotFoundError:
        return False


def print_setup_instructions(headphones: bool) -> None:
    """Print a short setup checklist for a fresh machine."""
    output_desc = "your headphones" if headphones else "MacBook speakers"
    print(f"""
=== Recording setup ===

System audio (the other person's voice) is captured via ScreenCaptureKit.
No virtual audio device (BlackHole) or Multi-Output Device is needed.

Checklist:
  1. ffmpeg installed: brew install ffmpeg
  2. Screen Recording permission granted to your terminal:
     System Settings → Privacy & Security → Screen Recording → enable Terminal
  3. Microphone permission granted to your terminal and to Python (for the
     PortAudio A/B mic track):
     System Settings → Privacy & Security → Microphone
  4. Headphones recommended ({output_desc} currently detected as default).
     Without headphones the mic picks up bleed from the speakers, so the two
     tracks are no longer cleanly separated.

To record:
  call-recorder

To add a therapist to the arrow-key menu:
  call-recorder therapist add "Name"
=== End setup ===
""")
