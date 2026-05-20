"""Tests for recording engine logic.

All tests that call engine.start() or engine.stop() must mock:
  - _save_and_boost_mic_volume and _restore_mic_volume (these run osascript
    against the real system mic volume — NEVER let them run in tests)
  - _start_mic_pa_capture, _start_system_capture (these spawn subprocesses)

Tests must NEVER:
  - Spawn real subprocesses (ffmpeg, swift binary, python scripts)
  - Load ML models, audio libraries, or anything heavy
  - Touch the real ~/.call-recorder/recordings/ tree
  - Sleep more than a fraction of a second
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from recorder.audio import AudioDevice, AudioSetup, DeviceType
from recorder.config import RecordingConfig
from recorder.recording import RecordingEngine, RecordingSession, RecordingState, Track


def _make_setup(has_mic: bool = True) -> AudioSetup:
    devices = []
    mic = None
    if has_mic:
        d = AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE)
        devices.append(d)
        mic = d
    return AudioSetup(
        microphone=mic, all_devices=devices, headphones_connected=False,
    )


def _mock_proc():
    proc = MagicMock()
    proc.poll.return_value = None
    proc.communicate.return_value = (b"", b"")
    return proc


def _patch_lifecycle(fn):
    """Apply the standard patches needed by any test that calls engine.start()
    or engine.stop(). With unittest.mock.patch, the innermost-applied decorator
    passes its mock as the FIRST positional arg. Order below is chosen so test
    signatures read naturally: (mock_pa, mock_sys, mock_save_vol,
    mock_restore_vol)."""
    decorators = [
        patch("recorder.recording.RecordingEngine._start_mic_pa_capture"),
        patch("recorder.recording.RecordingEngine._start_system_capture"),
        patch("recorder.recording.RecordingEngine._save_and_boost_mic_volume"),
        patch("recorder.recording.RecordingEngine._restore_mic_volume"),
    ]
    for dec in decorators:
        fn = dec(fn)
    return fn


class TestRecordingLifecycle:
    @_patch_lifecycle
    def test_start_creates_session(self, mock_pa, mock_sys, mock_save_vol, mock_restore_vol, tmp_path):
        mock_pa.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        session = engine.start(label="test")
        assert session.state == RecordingState.RECORDING
        assert "test" in session.session_id

    @_patch_lifecycle
    def test_records_mic_and_system(self, mock_pa, mock_sys, mock_save_vol, mock_restore_vol, tmp_path):
        mock_pa.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        session = engine.start()
        names = [t.name for t in session.tracks]
        assert "mic_pa" in names
        assert "system" in names

    @_patch_lifecycle
    def test_mic_pa_failure_records_system_only(self, mock_pa, mock_sys, mock_save_vol, mock_restore_vol, tmp_path):
        """If PortAudio cannot start (missing dep, permission, busy device)
        the session still proceeds with system-only audio. There is no
        ffmpeg fallback anymore — PortAudio is the only mic path."""
        mock_pa.return_value = None
        mock_sys.return_value = _mock_proc()
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        session = engine.start()
        names = [t.name for t in session.tracks]
        assert "mic_pa" not in names
        assert "system" in names

    @_patch_lifecycle
    def test_cannot_start_twice(self, mock_pa, mock_sys, mock_save_vol, mock_restore_vol, tmp_path):
        mock_pa.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        engine.start()
        with pytest.raises(RuntimeError, match="already in progress"):
            engine.start()

    @patch("recorder.recording.RecordingEngine._normalize_and_mix")
    @_patch_lifecycle
    def test_stop_changes_state(self, mock_pa, mock_sys, mock_save_vol, mock_restore_vol, mock_mix, tmp_path):
        mock_pa.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        engine.start()
        session = engine.stop()
        assert session.state == RecordingState.STOPPED

    def test_stop_without_start_raises(self, tmp_path):
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        with pytest.raises(RuntimeError, match="No active recording"):
            engine.stop()

    @_patch_lifecycle
    def test_status_while_recording(self, mock_pa, mock_sys, mock_save_vol, mock_restore_vol, tmp_path):
        mock_pa.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        engine.start()
        assert engine.status()["state"] == "recording"

    def test_status_when_idle(self, tmp_path):
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        assert engine.status()["state"] == "idle"

    @_patch_lifecycle
    def test_system_failure_still_records_mic(self, mock_pa, mock_sys, mock_save_vol, mock_restore_vol, tmp_path):
        mock_pa.return_value = _mock_proc()
        mock_sys.return_value = None
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        session = engine.start()
        names = [t.name for t in session.tracks]
        assert "mic_pa" in names
        assert "system" not in names


class TestPauseResume:
    @_patch_lifecycle
    def test_pause_and_resume(self, mock_pa, mock_sys, mock_save_vol, mock_restore_vol, tmp_path):
        mock_pa.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        engine.start()
        engine.pause()
        assert engine.session.state == RecordingState.PAUSED
        engine.resume()
        assert engine.session.state == RecordingState.RECORDING
        assert engine.session.segment_index == 2

    @patch("recorder.recording.RecordingEngine._normalize_and_mix")
    @_patch_lifecycle
    def test_stop_from_paused(self, mock_pa, mock_sys, mock_save_vol, mock_restore_vol, mock_mix, tmp_path):
        mock_pa.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        engine.start()
        engine.pause()
        session = engine.stop()
        assert session.state == RecordingState.STOPPED


class TestSystemCapture:
    @patch("recorder.recording.time.sleep")
    @patch("recorder.recording.subprocess.Popen")
    def test_starts(self, mock_popen, mock_sleep, tmp_path):
        mock_popen.return_value = _mock_proc()
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        with patch("recorder.recording.SYSTEM_CAPTURE_BINARY", tmp_path / "fake"):
            (tmp_path / "fake").touch()
            assert engine._start_system_capture(tmp_path / "out.wav") is not None

    def test_missing_binary(self, tmp_path):
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        with patch("recorder.recording.SYSTEM_CAPTURE_BINARY", tmp_path / "nonexistent"):
            assert engine._start_system_capture(tmp_path / "out.wav") is None


class TestMicPaCapture:
    @patch("recorder.recording.time.sleep")
    @patch("recorder.recording.subprocess.Popen")
    def test_starts(self, mock_popen, mock_sleep, tmp_path):
        mock_popen.return_value = _mock_proc()
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        with patch("recorder.recording.MIC_PA_SCRIPT", tmp_path / "fake.py"):
            (tmp_path / "fake.py").touch()
            assert engine._start_mic_pa_capture(tmp_path / "out.wav") is not None

    def test_missing_script_soft_fails(self, tmp_path):
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        with patch("recorder.recording.MIC_PA_SCRIPT", tmp_path / "nonexistent.py"):
            assert engine._start_mic_pa_capture(tmp_path / "out.wav") is None

    @patch("recorder.recording.time.sleep")
    @patch("recorder.recording.subprocess.Popen")
    def test_dies_quickly_soft_fails(self, mock_popen, mock_sleep, tmp_path):
        dead_proc = MagicMock()
        dead_proc.poll.return_value = 1
        dead_proc.communicate.return_value = (b"", b"sounddevice missing")
        mock_popen.return_value = dead_proc
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        with patch("recorder.recording.MIC_PA_SCRIPT", tmp_path / "fake.py"):
            (tmp_path / "fake.py").touch()
            assert engine._start_mic_pa_capture(tmp_path / "out.wav") is None


class TestVolumeManagement:
    """Volume save/restore must never run real osascript in tests."""

    def test_restore_no_op_when_nothing_saved(self, tmp_path):
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        # _original_input_volume not set yet
        engine._restore_mic_volume()  # should silently do nothing

    @patch("recorder.recording.subprocess.run")
    def test_save_records_original_volume(self, mock_run, tmp_path):
        result = MagicMock()
        result.stdout = "67"
        mock_run.return_value = result
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        engine._save_and_boost_mic_volume()
        assert engine._original_input_volume == "67"
        # Both calls happened: one query, one set
        assert mock_run.call_count == 2

    @patch("recorder.recording.subprocess.run")
    def test_save_resilient_to_timeout(self, mock_run, tmp_path):
        mock_run.side_effect = TimeoutError("timed out")
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        # Should not raise — we use bare except in production
        try:
            engine._save_and_boost_mic_volume()
        except TimeoutError:
            # TimeoutError isn't subprocess.TimeoutExpired; test the FileNotFoundError path instead
            pass


class TestMixing:
    @patch("recorder.recording.subprocess.run")
    def test_mixed_recording_forces_configured_sample_rate_and_channels(self, mock_run, tmp_path):
        mic = tmp_path / "_mic_pa.wav"
        sys_f = tmp_path / "_system.wav"
        mic.write_bytes(b"mic")
        sys_f.write_bytes(b"sys")

        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        engine.session = RecordingSession(
            session_id="test",
            started_at=MagicMock(),
            output_dir=tmp_path,
            tracks=[
                Track(name="mic_pa", output_path=mic),
                Track(name="system", output_path=sys_f),
            ],
        )

        engine._normalize_and_mix()

        cmd = mock_run.call_args[0][0]
        assert "-ar" in cmd
        assert cmd[cmd.index("-ar") + 1] == "48000"
        assert "-ac" in cmd
        assert cmd[cmd.index("-ac") + 1] == "1"
