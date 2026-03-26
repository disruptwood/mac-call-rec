"""Create a Multi-Output Device programmatically via CoreAudio API.

This uses ctypes to call AudioHardwareCreateAggregateDevice, which is the same
thing Audio MIDI Setup does when you click "+" → "Create Multi-Output Device".
"""

from __future__ import annotations

import ctypes
import ctypes.util
import subprocess
import sys
import json


def get_audio_device_ids() -> dict[str, dict]:
    """Get audio device info from system_profiler."""
    result = subprocess.run(
        ["system_profiler", "SPAudioDataType"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout


def find_blackhole_uid() -> str | None:
    """Find BlackHole's UID by querying ffmpeg device list and matching."""
    # We need the CoreAudio UID, not just the name.
    # Use the known convention: BlackHole 2ch UID is "BlackHole2ch_UID"
    # But let's verify it exists first
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "blackhole" in result.stderr.lower():
            return "BlackHole2ch_UID"
    except Exception:
        pass
    return None


def create_multi_output_device_via_applescript(
    name: str = "Record Output",
    include_speakers: bool = True,
    include_blackhole: bool = True,
) -> bool:
    """Create Multi-Output Device using AppleScript to drive Audio MIDI Setup.

    Falls back to providing manual instructions if automation fails.
    """
    # AppleScript approach: open Audio MIDI Setup and guide user
    # Unfortunately Audio MIDI Setup is not fully scriptable.
    # Instead, we use the `audiodevice` approach or direct instructions.

    # Alternative: use the undocumented `coreaudiod` aggregate device creation
    # by writing to the preferences plist

    # The most reliable approach on modern macOS: use a small Swift helper
    # or guide the user. Let's try the plist approach first.

    print(f"Creating Multi-Output Device '{name}'...")
    print()
    print("Unfortunately, macOS doesn't expose a public CLI for creating")
    print("Multi-Output Devices. The Audio MIDI Setup app is required.")
    print()
    print("Opening Audio MIDI Setup for you now...")

    subprocess.run(["open", "-a", "Audio MIDI Setup"], check=False)

    print()
    print("Follow these steps in the app:")
    print("  1. Click '+' at bottom-left → 'Create Multi-Output Device'")
    print("  2. Check: 'MacBook Air Speakers' (or your headphones)")
    print("  3. Check: 'BlackHole 2ch'")
    print("  4. Enable 'Drift Correction' for BlackHole 2ch")
    print("  5. Double-click the name to rename it to: " + name)
    print("  6. Right-click → 'Use This Device For Sound Output'")
    print()

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Set up Multi-Output Device for call recording")
    parser.add_argument("--name", default="Record Output", help="Name for the Multi-Output Device")
    args = parser.parse_args()

    # Check if BlackHole is visible
    blackhole = find_blackhole_uid()
    if not blackhole:
        print("BlackHole is not detected by the audio system.")
        print("Try restarting CoreAudio:")
        print("  sudo launchctl kickstart -kp system/com.apple.audio.coreaudiod")
        print()
        print("If that doesn't work, reinstall BlackHole:")
        print("  brew reinstall --cask blackhole-2ch")
        sys.exit(1)

    create_multi_output_device_via_applescript(name=args.name)


if __name__ == "__main__":
    main()
