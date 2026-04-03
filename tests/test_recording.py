"""Tests for recording engine logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from recorder.audio import AudioDevice, AudioSetup, DeviceType
from recorder.config import RecordingConfig
from recorder.recording import RecordingEngine, RecordingState


def _make_setup(has_mic: bool = True) -> AudioSetup:
    devices = []
    mic = None
    if has_mic:
        d = AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE)
        devices.append(d)
        mic = d
    return AudioSetup(
        microphone=mic, system_capture=None, all_devices=devices,
        headphones_connected=False, blackhole_available=False,
    )


def _mock_proc():
    proc = MagicMock()
    proc.poll.return_value = None
    proc.communicate.return_value = (b"", b"")
    return proc


class TestRecordingLifecycle:
    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg_wav")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_start_creates_session(self, mock_ffmpeg, mock_ffmpeg_wav, mock_sys, tmp_path):
        mock_ffmpeg_wav.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        session = engine.start(label="test")
        assert session.state == RecordingState.RECORDING
        assert "test" in session.session_id

    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg_wav")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_records_mic_and_system(self, mock_ffmpeg, mock_ffmpeg_wav, mock_sys, tmp_path):
        mock_ffmpeg_wav.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        session = engine.start()
        names = [t.name for t in session.tracks]
        assert "mic" in names
        assert "system" in names

    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg_wav")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_cannot_start_twice(self, mock_ffmpeg, mock_ffmpeg_wav, mock_sys, tmp_path):
        mock_ffmpeg_wav.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        engine.start()
        with pytest.raises(RuntimeError, match="already in progress"):
            engine.start()

    @patch("recorder.recording.RecordingEngine._normalize_and_mix")
    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg_wav")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_stop_changes_state(self, mock_ffmpeg, mock_ffmpeg_wav, mock_sys, mock_mix, tmp_path):
        mock_ffmpeg_wav.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        engine.start()
        session = engine.stop()
        assert session.state == RecordingState.STOPPED

    def test_stop_without_start_raises(self, tmp_path):
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        with pytest.raises(RuntimeError, match="No active recording"):
            engine.stop()

    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg_wav")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_status_while_recording(self, mock_ffmpeg, mock_ffmpeg_wav, mock_sys, tmp_path):
        mock_ffmpeg_wav.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        engine.start()
        assert engine.status()["state"] == "recording"

    def test_status_when_idle(self, tmp_path):
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        assert engine.status()["state"] == "idle"

    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg_wav")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_system_failure_still_records_mic(self, mock_ffmpeg, mock_ffmpeg_wav, mock_sys, tmp_path):
        mock_ffmpeg_wav.return_value = _mock_proc()
        mock_sys.return_value = None
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        session = engine.start()
        names = [t.name for t in session.tracks]
        assert "mic" in names
        assert "system" not in names


class TestPauseResume:
    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg_wav")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_pause_and_resume(self, mock_ffmpeg, mock_ffmpeg_wav, mock_sys, tmp_path):
        mock_ffmpeg_wav.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        engine.start()
        engine.pause()
        assert engine.session.state == RecordingState.PAUSED
        engine.resume()
        assert engine.session.state == RecordingState.RECORDING
        assert engine.session.segment_index == 2

    @patch("recorder.recording.RecordingEngine._normalize_and_mix")
    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg_wav")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_stop_from_paused(self, mock_ffmpeg, mock_ffmpeg_wav, mock_sys, mock_mix, tmp_path):
        mock_ffmpeg_wav.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        engine.start()
        engine.pause()
        session = engine.stop()
        assert session.state == RecordingState.STOPPED


class TestFfmpegCommand:
    @patch("recorder.recording.time.sleep")
    @patch("recorder.recording.subprocess.Popen")
    def test_ffmpeg_command_structure(self, mock_popen, mock_sleep, tmp_path):
        mock_popen.return_value = _mock_proc()
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path), sample_rate=44100), _make_setup())
        device = AudioDevice(1, "MacBook Air Microphone", DeviceType.MICROPHONE)
        engine._start_ffmpeg(device, tmp_path / "test.wav")
        cmd = mock_popen.call_args[0][0]
        assert "ffmpeg" in cmd[0]
        assert "avfoundation" in cmd
        assert ":1" in cmd
        assert "44100" in cmd

    @patch("recorder.recording.time.sleep")
    @patch("recorder.recording.subprocess.Popen")
    def test_ffmpeg_failure_raises(self, mock_popen, mock_sleep, tmp_path):
        proc = MagicMock()
        proc.poll.return_value = 1
        proc.communicate.return_value = (b"", b"device not found")
        mock_popen.return_value = proc
        engine = RecordingEngine(RecordingConfig(recordings_dir=str(tmp_path)), _make_setup())
        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            engine._start_ffmpeg(AudioDevice(99, "X", DeviceType.MICROPHONE), tmp_path / "x.m4a")


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
