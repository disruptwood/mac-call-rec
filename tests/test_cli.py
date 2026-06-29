"""Tests for the interactive CLI helpers — session picker and the
post-recording transcription prompt.

All tests run in non-TTY mode (sys.stdin.isatty() patched to False) so we
hit the input()-based branch of _choose_session_label. The arrow-key branch
relies on termios raw mode and stdin.buffer.read1 — exercising it from
pytest would require a real PTY and ergonomically isn't worth it.

Hard rules:
  - NEVER spawn a real transcribe_gemini.py subprocess. subprocess.call is
    always mocked.
  - input() is always mocked to a fixed sequence — never let a test block
    on real stdin.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from recorder import cli
from recorder.config import SessionType


SESSION_TYPES = [
    SessionType(name="Терапия", label="therapy"),
    SessionType(name="Созвон", label="meeting"),
]


@pytest.fixture
def non_tty():
    """Force the picker into its non-interactive (input()-based) branch."""
    with patch("recorder.cli.sys.stdin.isatty", return_value=False):
        yield


class TestChooseSessionLabelNonTTY:
    def test_picks_first_option(self, non_tty):
        with patch("builtins.input", side_effect=["1"]):
            assert cli._choose_session_label(SESSION_TYPES) == "therapy"

    def test_picks_second_option(self, non_tty):
        with patch("builtins.input", side_effect=["2"]):
            assert cli._choose_session_label(SESSION_TYPES) == "meeting"

    def test_custom_with_valid_label(self, non_tty):
        # "3" is the implicit Custom entry the picker appends.
        with patch("builtins.input", side_effect=["3", "meeting-2026"]):
            assert cli._choose_session_label(SESSION_TYPES) == "meeting-2026"

    def test_custom_with_empty_label_returns_none(self, non_tty):
        with patch("builtins.input", side_effect=["3", ""]):
            assert cli._choose_session_label(SESSION_TYPES) is None

    def test_custom_with_invalid_label_returns_none(self, non_tty, capsys):
        # Whitespace, slashes, and non-ASCII are blocked because the value
        # gets stitched directly into a directory name.
        with patch("builtins.input", side_effect=["3", "has space"]):
            assert cli._choose_session_label(SESSION_TYPES) is None

    def test_non_digit_input_returns_none(self, non_tty):
        with patch("builtins.input", side_effect=["foo"]):
            assert cli._choose_session_label(SESSION_TYPES) is None

    def test_out_of_range_returns_none(self, non_tty):
        with patch("builtins.input", side_effect=["99"]):
            assert cli._choose_session_label(SESSION_TYPES) is None

    def test_zero_is_out_of_range(self, non_tty):
        # Display is 1-indexed. "0" would map to options[-1] without a guard.
        with patch("builtins.input", side_effect=["0"]):
            assert cli._choose_session_label(SESSION_TYPES) is None

    def test_eof_returns_none(self, non_tty):
        with patch("builtins.input", side_effect=EOFError):
            assert cli._choose_session_label(SESSION_TYPES) is None

    def test_keyboard_interrupt_returns_none(self, non_tty):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            assert cli._choose_session_label(SESSION_TYPES) is None


class TestPromptCustomLabel:
    def test_valid_label_passes(self):
        with patch("builtins.input", return_value="therapy-2026"):
            assert cli._prompt_custom_label() == "therapy-2026"

    def test_alphanum_only_passes(self):
        with patch("builtins.input", return_value="meeting42"):
            assert cli._prompt_custom_label() == "meeting42"

    def test_dot_underscore_dash_allowed(self):
        with patch("builtins.input", return_value="call_2026.05-22"):
            assert cli._prompt_custom_label() == "call_2026.05-22"

    @pytest.mark.parametrize("bad", [
        "",                       # empty
        " ",                      # whitespace only
        "has space",              # spaces
        "терапия",                # cyrillic
        "../escape",              # path traversal attempt
        "/abs",                   # absolute path
        "-leading-dash",          # must start alphanum
        "x" * 65,                 # too long
    ])
    def test_invalid_labels_rejected(self, bad):
        with patch("builtins.input", return_value=bad):
            assert cli._prompt_custom_label() is None

    def test_eof_returns_none(self):
        with patch("builtins.input", side_effect=EOFError):
            assert cli._prompt_custom_label() is None


def _fake_session(tmp_path: Path, *, recording_exists: bool = True):
    rec = tmp_path / "recording.m4a"
    if recording_exists:
        rec.write_bytes(b"fake")
    return SimpleNamespace(output_dir=tmp_path, recording_path=rec)


class TestMaybeRunTranscription:
    """The transcribe helper never spawns real transcription processes."""

    def test_none_backend_skips(self, tmp_path):
        session = _fake_session(tmp_path)
        with patch("recorder.cli.subprocess.call") as mock_call:
            cli._maybe_run_transcription(session, backend="none", ask=False)
        mock_call.assert_not_called()

    def test_gemini_skips_when_file_missing(self, tmp_path):
        session = _fake_session(tmp_path, recording_exists=False)
        with patch("recorder.cli.subprocess.call") as mock_call:
            cli._maybe_run_transcription(session, backend="gemini", ask=False)
        mock_call.assert_not_called()

    def test_gemini_ask_false_runs_without_prompt(self, tmp_path):
        session = _fake_session(tmp_path)
        script = tmp_path / "transcribe_gemini.py"
        script.write_text("")
        with patch("recorder.cli.subprocess.call", return_value=0) as mock_call, \
             patch("recorder.cli._GEMINI_TRANSCRIBE_SCRIPT", script), \
             patch("builtins.input") as mock_input:
            cli._maybe_run_transcription(session, backend="gemini", ask=False)
        mock_input.assert_not_called()
        mock_call.assert_called_once()
        assert str(session.recording_path) in mock_call.call_args[0][0]

    def test_yes_runs_gemini_transcription(self, tmp_path):
        session = _fake_session(tmp_path)
        script = tmp_path / "transcribe_gemini.py"
        script.write_text("")
        with patch("recorder.cli.subprocess.call", return_value=0) as mock_call, \
             patch("recorder.cli._GEMINI_TRANSCRIBE_SCRIPT", script), \
             patch("builtins.input", return_value="y"):
            cli._maybe_run_transcription(session, backend="gemini", ask=True)
        mock_call.assert_called_once()

    def test_empty_response_defaults_to_no(self, tmp_path):
        session = _fake_session(tmp_path)
        script = tmp_path / "transcribe_gemini.py"
        script.write_text("")
        with patch("recorder.cli.subprocess.call") as mock_call, \
             patch("recorder.cli._GEMINI_TRANSCRIBE_SCRIPT", script), \
             patch("builtins.input", return_value=""):
            cli._maybe_run_transcription(session, backend="gemini", ask=True)
        mock_call.assert_not_called()

    @pytest.mark.parametrize("answer", ["n", "N", "no", "NO", "н", "нет"])
    def test_no_skips_transcription(self, tmp_path, answer):
        session = _fake_session(tmp_path)
        script = tmp_path / "transcribe_gemini.py"
        script.write_text("")
        with patch("recorder.cli.subprocess.call") as mock_call, \
             patch("recorder.cli._GEMINI_TRANSCRIBE_SCRIPT", script), \
             patch("builtins.input", return_value=answer):
            cli._maybe_run_transcription(session, backend="gemini", ask=True)
        mock_call.assert_not_called()

    def test_eof_on_prompt_skips(self, tmp_path):
        # Don't crash if stdin closes between recording and the prompt
        # (e.g. user piped recording without -l and lost interactivity).
        session = _fake_session(tmp_path)
        with patch("recorder.cli.subprocess.call") as mock_call, \
             patch("builtins.input", side_effect=EOFError):
            cli._maybe_run_transcription(session, backend="gemini", ask=True)
        mock_call.assert_not_called()

    def test_missing_transcribe_script_warns_and_skips(self, tmp_path, capsys):
        session = _fake_session(tmp_path)
        with patch("recorder.cli.subprocess.call") as mock_call, \
             patch("recorder.cli._GEMINI_TRANSCRIBE_SCRIPT", tmp_path / "missing.py"):
            cli._maybe_run_transcription(session, backend="gemini", ask=False)
        mock_call.assert_not_called()
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_nonzero_exit_prints_warning(self, tmp_path, capsys):
        session = _fake_session(tmp_path)
        script = tmp_path / "transcribe_gemini.py"
        script.write_text("")
        with patch("recorder.cli.subprocess.call", return_value=2), \
             patch("recorder.cli._GEMINI_TRANSCRIBE_SCRIPT", script):
            cli._maybe_run_transcription(session, backend="gemini", ask=False)
        captured = capsys.readouterr()
        assert "кодом 2" in captured.out

    def test_keyboard_interrupt_during_transcription_is_caught(self, tmp_path, capsys):
        session = _fake_session(tmp_path)
        script = tmp_path / "transcribe_gemini.py"
        script.write_text("")
        with patch("recorder.cli.subprocess.call", side_effect=KeyboardInterrupt), \
             patch("recorder.cli._GEMINI_TRANSCRIBE_SCRIPT", script):
            # Must NOT propagate — user hitting Ctrl-C during a long Gemini
            # call should drop back to the shell cleanly, not raise.
            cli._maybe_run_transcription(session, backend="gemini", ask=False)
        captured = capsys.readouterr()
        assert "прервана" in captured.out

    def test_local_backend_uses_session_dir(self, tmp_path):
        session = _fake_session(tmp_path)
        script = tmp_path / "transcribe.py"
        script.write_text("")
        with patch("recorder.cli.subprocess.call", return_value=0) as mock_call, \
             patch("recorder.cli._LOCAL_TRANSCRIBE_SCRIPT", script):
            cli._maybe_run_transcription(session, backend="local", ask=False)
        mock_call.assert_called_once()
        assert str(tmp_path) in mock_call.call_args[0][0]

    def test_invalid_backend_skips(self, tmp_path, capsys):
        session = _fake_session(tmp_path)
        with patch("recorder.cli.subprocess.call") as mock_call:
            cli._maybe_run_transcription(session, backend="bogus", ask=False)
        mock_call.assert_not_called()
        assert "invalid" in capsys.readouterr().out


class TestTherapistSessions:
    def test_build_therapy_session_slugifies_cyrillic_name(self):
        config = cli.RecordingConfig()
        session = cli._build_therapy_session(config, "Анна")
        assert session.name == "Терапия Анна"
        assert session.label == "therapy-anna"

    def test_build_therapy_session_makes_label_unique(self):
        config = cli.RecordingConfig()
        config.session_types = [{"name": "Терапия Анна", "label": "therapy-anna"}]
        session = cli._build_therapy_session(config, "Анна")
        assert session.label == "therapy-anna-2"

    def test_persist_session_type_keeps_default_menu_on_first_edit(self):
        config = cli.RecordingConfig()
        session = cli._build_therapy_session(config, "Анна")
        assert cli._persist_session_type(config, session) is True
        labels = [s["label"] for s in config.session_types]
        assert "therapy" in labels
        assert "therapy-anna" in labels


class TestDefaultSubcommand:
    """The main() entry point routes `python3 -m recorder` (no args) through
    cmd_start with all defaults — that's what makes the picker the default
    front door."""

    def test_no_args_invokes_cmd_start(self):
        fake_cmd_start = MagicMock()
        with patch("recorder.cli.cmd_start", fake_cmd_start), \
             patch("recorder.cli.sys.argv", ["recorder"]):
            cli.main()
        assert fake_cmd_start.called
        args = fake_cmd_start.call_args[0][0]
        assert args.command == "start"
        assert args.label is None
        assert args.profile is None
        assert args.no_transcribe is False
        assert args.transcribe is None

    def test_no_transcribe_flag_is_propagated(self):
        fake_cmd_start = MagicMock()
        with patch("recorder.cli.cmd_start", fake_cmd_start), \
             patch("recorder.cli.sys.argv", ["recorder", "start", "--no-transcribe"]):
            cli.main()
        args = fake_cmd_start.call_args[0][0]
        assert args.no_transcribe is True

    def test_explicit_label_skips_picker(self):
        fake_cmd_start = MagicMock()
        with patch("recorder.cli.cmd_start", fake_cmd_start), \
             patch("recorder.cli.sys.argv", ["recorder", "start", "-l", "therapy"]):
            cli.main()
        args = fake_cmd_start.call_args[0][0]
        assert args.label == "therapy"

    def test_transcribe_flag_is_propagated(self):
        fake_cmd_start = MagicMock()
        with patch("recorder.cli.cmd_start", fake_cmd_start), \
             patch("recorder.cli.sys.argv", ["recorder", "start", "--transcribe", "local"]):
            cli.main()
        args = fake_cmd_start.call_args[0][0]
        assert args.transcribe == "local"
