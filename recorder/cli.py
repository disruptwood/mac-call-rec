"""CLI entry point for call recorder."""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import signal
import subprocess
import sys
import termios
import time
import tty
from dataclasses import asdict
from pathlib import Path

from .audio import detect_audio_setup
from .config import RecordingConfig, SessionType, slugify_label
from .recording import RecordingEngine
from .setup_helper import (
    check_ffmpeg_installed,
    print_setup_instructions,
)

# Resolve once so tests can monkeypatch fake paths before invoking cmd_start.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GEMINI_TRANSCRIBE_SCRIPT = _PROJECT_ROOT / "scripts" / "transcribe_gemini.py"
_LOCAL_TRANSCRIBE_SCRIPT = _PROJECT_ROOT / "scripts" / "transcribe.py"
_TRANSCRIPTION_BACKENDS = {"none", "local", "gemini"}

# Whitelist for filenames going into session directory names. Anything outside
# this set in a custom label gets rejected — the label is stitched directly
# into a path so we can't trust user input here.
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

PID_FILE = Path.home() / ".call-recorder" / "active.pid"


# Sentinel returned from the picker for the "custom — type a label" choice.
# Using a class attribute on SessionType would tangle UI concerns with config.
_CUSTOM_LABEL = "__custom__"


def _render_menu(options: list[tuple[str, str]], selected: int, *, redraw: bool) -> None:
    """Render the session picker. With redraw=True, move cursor up by the
    number of lines the previous render printed so the new state overwrites
    in place. Total printed lines (including the title and footer) = len + 3.
    """
    total_lines = len(options) + 3
    if redraw:
        # Move up to top of previous render, then clear-to-end-of-screen.
        sys.stdout.write(f"\x1b[{total_lines}A\x1b[J")
    sys.stdout.write("Выбери тип сессии:\r\n")
    for i, (display, _label) in enumerate(options):
        marker = "\x1b[36m▶\x1b[0m" if i == selected else " "
        num = f"{i + 1}."
        line = f"  {marker} {num} {display}"
        if i == selected:
            line = f"\x1b[1m{line}\x1b[0m"
        sys.stdout.write(line + "\r\n")
    sys.stdout.write("\r\n")
    sys.stdout.write("(стрелки или цифра, Enter — выбор, q/Esc — отмена)\r\n")
    sys.stdout.flush()


def _choose_session_label(session_types: list[SessionType]) -> str | None:
    """Interactive picker for session label.

    Returns the chosen label string, or None if the user aborted. For the
    "custom" option, drops out of raw mode and prompts with input(); validated
    against _LABEL_RE before being returned.

    Falls back to plain numbered prompt when stdin isn't a TTY (e.g. piped
    input in CI / scripts) — keeps the same shape but no arrow keys.
    """
    options: list[tuple[str, str]] = [(s.name, s.label) for s in session_types]
    options.append(("Custom (ввести свой label)", _CUSTOM_LABEL))

    if not sys.stdin.isatty():
        # Non-interactive fallback. Print menu and read one line.
        print("Выбери тип сессии:")
        for i, (display, _label) in enumerate(options):
            print(f"  {i + 1}. {display}")
        try:
            raw = input("Номер: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not raw.isdigit():
            return None
        idx = int(raw) - 1
        if not 0 <= idx < len(options):
            return None
        chosen_label = options[idx][1]
        if chosen_label == _CUSTOM_LABEL:
            return _prompt_custom_label()
        return chosen_label

    selected = 0
    _render_menu(options, selected, redraw=False)

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            # read1 returns whatever is buffered; arrow keys are 3 bytes
            # (ESC, '[', 'A'/'B') so we ask for 4 to be safe.
            key = sys.stdin.buffer.read1(4).decode("utf-8", errors="replace")
            if key in ("\x1b[A", "\x1bOA"):  # up arrow (xterm + screen modes)
                selected = (selected - 1) % len(options)
            elif key in ("\x1b[B", "\x1bOB"):  # down arrow
                selected = (selected + 1) % len(options)
            elif key in ("\r", "\n"):
                break
            elif key in ("\x03", "q", "\x1b"):
                # Ctrl-C, q, bare Esc — abort. Bare Esc is also the prefix of
                # arrow sequences, but read1(4) gets them in one call so a
                # standalone Esc is a real Esc.
                return None
            elif key.isdigit():
                idx = int(key) - 1
                if 0 <= idx < len(options):
                    selected = idx
                    break
            else:
                # Unknown key — ignore without redrawing.
                continue
            _render_menu(options, selected, redraw=True)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    sys.stdout.write("\r\n")
    sys.stdout.flush()

    chosen_label = options[selected][1]
    if chosen_label == _CUSTOM_LABEL:
        return _prompt_custom_label()
    return chosen_label


def _prompt_custom_label() -> str | None:
    """Prompt for a custom session label. Validated and length-bounded since
    the result ends up in a directory name."""
    try:
        raw = input("Label (латиница/цифры/-_., до 64 символов): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not raw:
        return None
    if not _LABEL_RE.match(raw):
        print(f"Невалидный label: {raw!r}. Разрешено: A-Za-z0-9._- (без пробелов).")
        return None
    return raw


def _transcription_command(session, backend: str) -> list[str] | None:
    """Build the subprocess command for the selected transcription backend."""
    if backend == "none":
        return None

    if backend == "local":
        if not session.output_dir or not session.output_dir.exists():
            return None
        script = _LOCAL_TRANSCRIBE_SCRIPT
        if not script.exists():
            print(f"WARN: not found: {script}")
            return None
        return [sys.executable, str(script), str(session.output_dir)]

    if backend == "gemini":
        recording_path = session.recording_path
        if not recording_path or not recording_path.exists():
            return None
        script = _GEMINI_TRANSCRIBE_SCRIPT
        if not script.exists():
            print(f"WARN: not found: {script}")
            return None
        return [sys.executable, str(script), str(recording_path)]

    print(f"WARN: unknown transcription backend: {backend}")
    return None


def _maybe_run_transcription(session, backend: str, ask: bool) -> None:
    """Optionally invoke a configured transcription backend after recording.

    `none` is the safe default for shared repos: no audio leaves the machine
    unless the user explicitly chooses a cloud backend.
    """
    backend = backend.strip().lower()
    if backend not in _TRANSCRIPTION_BACKENDS:
        print(f"WARN: invalid transcription_backend={backend!r}; skipping")
        return
    if backend == "none":
        return

    if ask:
        backend_name = "локальную транскрипцию" if backend == "local" else "Gemini-транскрипцию"
        try:
            ans = input(f"\nЗапустить {backend_name}? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if ans not in ("y", "yes", "д", "да"):
            return

    cmd = _transcription_command(session, backend)
    if cmd is None:
        return

    print(f"\nЗапуск транскрипции ({backend})...")
    try:
        rc = subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\nТранскрипция прервана пользователем.")
        return
    if rc != 0:
        print(f"WARN: транскрипция завершилась с кодом {rc}")


def _label_exists(config: RecordingConfig, label: str) -> bool:
    return any(s.label == label for s in config.list_session_types())


def _unique_label(config: RecordingConfig, base_label: str) -> str:
    """Return a label not currently used by the visible session menu."""
    label = base_label
    i = 2
    while _label_exists(config, label):
        label = f"{base_label}-{i}"
        i += 1
    return label


def _persist_session_type(config: RecordingConfig, session: SessionType) -> bool:
    """Add/update a session type and persist defaults if this is first edit.

    Returns True when a new menu item was added, False when an existing label
    was updated or an identical item already existed.
    """
    sessions = config.list_session_types()
    for i, existing in enumerate(sessions):
        if existing.label == session.label:
            changed = existing.name != session.name
            sessions[i] = session
            config.session_types = [asdict(s) for s in sessions]
            return changed
        if existing.name.casefold() == session.name.casefold():
            return False

    sessions.append(session)
    config.session_types = [asdict(s) for s in sessions]
    return True


def _build_therapy_session(config: RecordingConfig, therapist_name: str) -> SessionType:
    name = therapist_name.strip()
    if not name:
        raise ValueError("Therapist name cannot be empty")
    display = f"Терапия {name}"
    base_label = slugify_label(f"therapy-{name}", fallback="therapy")
    return SessionType(name=display, label=_unique_label(config, base_label))


def cmd_start(args: argparse.Namespace) -> None:
    """Start recording."""
    config = RecordingConfig.load()

    # Resolve label first — without a label there's no point launching the
    # capture pipeline since the session directory needs a name.
    label = args.label
    if label is None:
        label = _choose_session_label(config.list_session_types())
        if label is None:
            print("Отменено.")
            return

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
    session = engine.start(label=label)

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
    print("Controls: [space] pause/resume, [q] or Ctrl+C to stop\n")

    stopping = False

    def handle_signal(sig, frame):
        nonlocal stopping
        if stopping:
            return
        stopping = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Raw terminal mode for keypress detection
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())

        while not stopping:
            status = engine.status()
            duration = status["duration_seconds"]
            mins, secs = divmod(int(duration), 60)
            hrs, mins = divmod(mins, 60)
            state = status["state"]

            if state == "paused":
                label = f"PAUSED  {hrs:02d}:{mins:02d}:{secs:02d}"
            else:
                label = f"REC     {hrs:02d}:{mins:02d}:{secs:02d}"

            sys.stdout.write(f"\r\x1b[K{label}")
            sys.stdout.flush()

            # Check for keypress (non-blocking)
            if select.select([sys.stdin], [], [], 0.5)[0]:
                key = sys.stdin.buffer.read1(4).decode("utf-8", errors="replace")
                if key == " ":
                    if state == "recording":
                        engine.pause()
                    elif state == "paused":
                        engine.resume()
                elif key.lower() in ("q", "й", "\x03"):
                    stopping = True
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    sys.stdout.write("\r\x1b[K")
    print("Stopping recording...")
    result = engine.stop()
    PID_FILE.unlink(missing_ok=True)
    _print_result(result)

    if args.no_transcribe:
        backend = "none"
        ask_transcription = False
    else:
        backend = args.transcribe or config.transcription_backend
        ask_transcription = args.transcribe is None

    _maybe_run_transcription(result, backend=backend, ask=ask_transcription)


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
        marker = " [default mic]" if d.is_builtin_mic else ""
        print(f"  [{d.index}] {d.name} ({d.device_type.value}){marker}")
    print(f"\nHeadphones connected: {'yes' if audio.headphones_connected else 'no'}")
    print("System audio capture: ScreenCaptureKit (no virtual device needed)")


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


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize local user config without writing anything into the repo."""
    config = RecordingConfig.load()

    if args.transcription_backend:
        config.transcription_backend = args.transcription_backend

    therapist_name = args.therapist
    if therapist_name is None and sys.stdin.isatty():
        try:
            therapist_name = input("Имя психолога для меню (Enter — пропустить): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            therapist_name = None

    if therapist_name:
        session = _build_therapy_session(config, therapist_name)
        added = _persist_session_type(config, session)
        verb = "Добавлено" if added else "Уже есть"
        print(f"{verb}: {session.name} ({session.label})")

    config.save()
    print(f"Config: {Path.home() / '.call-recorder' / 'config.json'}")
    print(f"Transcription backend: {config.transcription_backend}")
    print("Next: call-recorder")


def cmd_therapist_add(args: argparse.Namespace) -> None:
    """Add a therapist session preset to the local config."""
    config = RecordingConfig.load()
    session = _build_therapy_session(config, args.name)
    added = _persist_session_type(config, session)
    config.save()
    verb = "Added" if added else "Already present"
    print(f"{verb}: {session.name} ({session.label})")


def cmd_sessions(args: argparse.Namespace) -> None:
    """List session presets shown in the start menu."""
    config = RecordingConfig.load()
    print("Session menu:")
    for session in config.list_session_types():
        print(f"  {session.label:20s} {session.name}")


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
    capture_binary = _PROJECT_ROOT / "scripts" / "capture_system_audio"
    if capture_binary.exists():
        print("ScreenCaptureKit helper: OK")
    else:
        print("ScreenCaptureKit helper: missing")
        print("Build it with: ./scripts/bootstrap_macos.sh")

    audio = detect_audio_setup()
    print_setup_instructions(audio.headphones_connected)


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
    if session.recording_path and session.recording_path.exists():
        size = session.recording_path.stat().st_size
        print(f"Recording: {session.recording_path} ({size / 1024 / 1024:.1f} MB)")


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
    p_start.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Skip post-recording transcription",
    )
    p_start.add_argument(
        "--transcribe",
        choices=sorted(_TRANSCRIPTION_BACKENDS - {"none"}),
        default=None,
        help="Run a transcription backend after recording (local or gemini)",
    )
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

    # init
    p_init = sub.add_parser("init", help="Initialize local config")
    p_init.add_argument("--therapist", default=None, help="Therapist name to add to the menu")
    p_init.add_argument(
        "--transcription-backend",
        choices=sorted(_TRANSCRIPTION_BACKENDS),
        default=None,
        help="Default post-recording transcription backend",
    )
    p_init.set_defaults(func=cmd_init)

    # therapist
    p_therapist = sub.add_parser("therapist", help="Manage therapist presets")
    therapist_sub = p_therapist.add_subparsers(dest="therapist_command")
    p_therapist_add = therapist_sub.add_parser("add", help="Add therapist to session menu")
    p_therapist_add.add_argument("name", help="Therapist name")
    p_therapist_add.set_defaults(func=cmd_therapist_add)

    # sessions
    p_sessions = sub.add_parser("sessions", help="List session menu presets")
    p_sessions.set_defaults(func=cmd_sessions)

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
        # No subcommand → behave like `start` with all defaults. Keeps the
        # short form `python3 -m recorder` working as the primary entry point.
        args = parser.parse_args(["start"])

    if args.command == "therapist" and not getattr(args, "therapist_command", None):
        p_therapist.print_help()
        sys.exit(2)

    args.func(args)


if __name__ == "__main__":
    main()
