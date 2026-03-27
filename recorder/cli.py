"""CLI entry point for call recorder."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

from .audio import detect_audio_setup
from .config import RecordingConfig
from .recording import RecordingEngine
from .setup_helper import (
    check_blackhole_loaded,
    check_ffmpeg_installed,
    print_setup_instructions,
)

PID_FILE = Path.home() / ".call-recorder" / "active.pid"


def cmd_start(args: argparse.Namespace) -> None:
    """Start recording."""
    config = RecordingConfig.load()

    profile_name = args.profile or config.active_profile
    profile = config.get_profile(profile_name)

    print(f"Profile: {profile.name} — {profile.description}")
    print(f"Preferred mic: {profile.preferred_mic or '(auto — external/headphone mic)'}")

    audio = detect_audio_setup(profile)

    if not audio.can_record_mic:
        print("ERROR: No microphone detected.")
        sys.exit(1)

    print(f"Microphone: {audio.microphone.name}")
    print(f"System audio: ScreenCaptureKit")
    if audio.headphones_connected:
        print("Headphones: detected")

    engine = RecordingEngine(config, audio)
    session = engine.start(label=args.label)

    # Save state so 'stop' can find us
    state = {
        "pid": os.getpid(),
        "session_id": session.session_id,
        "profile": profile_name,
        "tracks": [
            {"name": t.name, "device_name": t.device_name, "path": str(t.output_path)}
            for t in session.tracks
        ],
    }
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(json.dumps(state, indent=2))

    print(f"\nRecording started: {session.session_id}")
    print(f"Tracks: {', '.join(t.name for t in session.tracks)}")
    print("Press Ctrl+C to stop recording.\n")

    def handle_signal(sig, frame):
        print("\nStopping recording...")
        result = engine.stop()
        PID_FILE.unlink(missing_ok=True)
        _print_result(result)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Keep alive
    try:
        while True:
            status = engine.status()
            duration = status["duration_seconds"]
            mins, secs = divmod(int(duration), 60)
            hrs, mins = divmod(mins, 60)
            sys.stdout.write(f"\rRecording... {hrs:02d}:{mins:02d}:{secs:02d}")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        handle_signal(None, None)


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop a recording started in the background."""
    if not PID_FILE.exists():
        print("No active recording found.")
        sys.exit(1)

    state = json.loads(PID_FILE.read_text())
    pid = state["pid"]

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent stop signal to recording process (PID {pid})")
        print(f"Session: {state['session_id']}")
    except ProcessLookupError:
        print(f"Recording process (PID {pid}) not found — may have already stopped.")
        PID_FILE.unlink(missing_ok=True)


def cmd_status(args: argparse.Namespace) -> None:
    """Show recording status."""
    if not PID_FILE.exists():
        print("No active recording.")
        return

    state = json.loads(PID_FILE.read_text())
    pid = state["pid"]

    try:
        os.kill(pid, 0)  # Check if alive
        print(f"Recording active (PID {pid})")
        print(f"Session: {state['session_id']}")
        print(f"Profile: {state.get('profile', 'unknown')}")
        for t in state.get("tracks", []):
            print(f"  Track: {t['name']} — {t['device_name']} → {t['path']}")
    except ProcessLookupError:
        print("Recording process not running (stale state).")
        PID_FILE.unlink(missing_ok=True)


def cmd_devices(args: argparse.Namespace) -> None:
    """List detected audio devices."""
    audio = detect_audio_setup()
    print("Detected audio devices:\n")
    for d in audio.all_devices:
        marker = ""
        if d.is_builtin_mic:
            marker = " [default mic]"
        elif d.is_blackhole:
            marker = " [system capture]"
        print(f"  [{d.index}] {d.name} ({d.device_type.value}){marker}")

    print(f"\nHeadphones connected: {'yes' if audio.headphones_connected else 'no'}")
    print(f"BlackHole available: {'yes' if audio.blackhole_available else 'no'}")


def cmd_profiles(args: argparse.Namespace) -> None:
    """List available profiles."""
    config = RecordingConfig.load()
    profiles = config.list_profiles()
    print("Available profiles:\n")
    for p in profiles:
        active = " (active)" if p.name == config.active_profile else ""
        mic_desc = p.preferred_mic or "(auto — external/headphone mic)"
        print(f"  {p.name}{active}")
        print(f"    {p.description}")
        print(f"    Mic: {mic_desc}")
        print()


def cmd_use_profile(args: argparse.Namespace) -> None:
    """Set the active profile."""
    config = RecordingConfig.load()
    config.get_profile(args.name)  # Validate it exists
    config.active_profile = args.name
    config.save()
    print(f"Active profile set to: {args.name}")


def cmd_setup(args: argparse.Namespace) -> None:
    """Check system setup and print instructions."""
    if not check_ffmpeg_installed():
        print("ERROR: ffmpeg not found. Install with: brew install ffmpeg")
        return

    print("ffmpeg: OK")

    if check_blackhole_loaded():
        print("BlackHole: OK (visible to ffmpeg)")
    else:
        print("BlackHole: NOT detected by ffmpeg")
        print("Try: sudo launchctl kickstart -kp system/com.apple.audio.coreaudiod")
        print()

    audio = detect_audio_setup()
    headphones = audio.headphones_connected
    print_setup_instructions(headphones)


def cmd_config(args: argparse.Namespace) -> None:
    """Show or edit config."""
    config = RecordingConfig.load()
    if args.key and args.value:
        if hasattr(config, args.key):
            field_type = type(getattr(config, args.key))
            if field_type == bool:
                setattr(config, args.key, args.value.lower() in ("true", "1", "yes"))
            elif field_type == int:
                setattr(config, args.key, int(args.value))
            else:
                setattr(config, args.key, args.value)
            config.save()
            print(f"Set {args.key} = {getattr(config, args.key)}")
        else:
            print(f"Unknown config key: {args.key}")
    else:
        from dataclasses import asdict
        print(json.dumps(asdict(config), indent=2))


def _print_result(session) -> None:
    print(f"\nSession: {session.session_id}")
    duration = session.duration_seconds
    mins, secs = divmod(int(duration), 60)
    hrs, mins = divmod(mins, 60)
    print(f"Duration: {hrs:02d}:{mins:02d}:{secs:02d}")
    for t in session.tracks:
        size = t.output_path.stat().st_size if t.output_path.exists() else 0
        print(f"  {t.name}: {t.output_path} ({size / 1024:.1f} KB)")
    if session.mixed_path and session.mixed_path.exists():
        size = session.mixed_path.stat().st_size
        print(f"  mixed: {session.mixed_path} ({size / 1024:.1f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="call-recorder",
        description="Record both sides of a voice call on macOS",
    )
    sub = parser.add_subparsers(dest="command")

    # start
    p_start = sub.add_parser("start", help="Start recording")
    p_start.add_argument("-l", "--label", default=None, help="Label for this session")
    p_start.add_argument("-p", "--profile", default=None, help="Audio profile to use")
    p_start.set_defaults(func=cmd_start)

    # stop
    p_stop = sub.add_parser("stop", help="Stop active recording")
    p_stop.set_defaults(func=cmd_stop)

    # status
    p_status = sub.add_parser("status", help="Show recording status")
    p_status.set_defaults(func=cmd_status)

    # devices
    p_devices = sub.add_parser("devices", help="List audio devices")
    p_devices.set_defaults(func=cmd_devices)

    # profiles
    p_profiles = sub.add_parser("profiles", help="List audio profiles")
    p_profiles.set_defaults(func=cmd_profiles)

    # use-profile
    p_use = sub.add_parser("use-profile", help="Set active audio profile")
    p_use.add_argument("name", help="Profile name")
    p_use.set_defaults(func=cmd_use_profile)

    # setup
    p_setup = sub.add_parser("setup", help="Check system setup and show instructions")
    p_setup.set_defaults(func=cmd_setup)

    # config
    p_config = sub.add_parser("config", help="Show or set config values")
    p_config.add_argument("key", nargs="?", help="Config key")
    p_config.add_argument("value", nargs="?", help="Config value")
    p_config.set_defaults(func=cmd_config)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
