"""Tests for recording engine logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from recorder.audio import AudioDevice, AudioSetup, DeviceType
from recorder.config import RecordingConfig
from recorder.recording import RecordingEngine, RecordingState


def _make_setup(
    has_mic: bool = True,
    has_blackhole: bool = False,
    has_external_mic: bool = False,
) -> AudioSetup:
    devices = []
    mic = None

    if has_mic:
        d = AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE)
        devices.append(d)
        mic = d

    if has_external_mic:
        d = AudioDevice(3, "External Microphone", DeviceType.MICROPHONE)
        devices.append(d)
        mic = d

    return AudioSetup(
        microphone=mic,
        system_capture=None,
        all_devices=devices,
        headphones_connected=False,
        blackhole_available=has_blackhole,
    )


def _mock_proc():
    proc = MagicMock()
    proc.poll.return_value = None
    proc.communicate.return_value = (b"", b"")
    return proc


class TestRecordingLifecycle:
    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_start_creates_session(self, mock_ffmpeg, mock_sys, tmp_path):
        mock_ffmpeg.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        config = RecordingConfig(recordings_dir=str(tmp_path))
        engine = RecordingEngine(config, _make_setup())

        session = engine.start(label="test")
        assert session.state == RecordingState.RECORDING
        assert "test" in session.session_id
        assert len(session.tracks) >= 1

    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_records_both_mic_and_system(self, mock_ffmpeg, mock_sys, tmp_path):
        mock_ffmpeg.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        config = RecordingConfig(recordings_dir=str(tmp_path))
        engine = RecordingEngine(config, _make_setup())

        session = engine.start()
        names = [t.name for t in session.tracks]
        assert "mic" in names
        assert "system" in names

    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_cannot_start_twice(self, mock_ffmpeg, mock_sys, tmp_path):
        mock_ffmpeg.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        config = RecordingConfig(recordings_dir=str(tmp_path))
        engine = RecordingEngine(config, _make_setup())

        engine.start()
        with pytest.raises(RuntimeError, match="already in progress"):
            engine.start()

    @patch("recorder.recording.RecordingEngine._concat_segments")
    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_stop_changes_state(self, mock_ffmpeg, mock_sys, mock_concat, tmp_path):
        mock_ffmpeg.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        config = RecordingConfig(recordings_dir=str(tmp_path))
        engine = RecordingEngine(config, _make_setup())

        engine.start()
        session = engine.stop()
        assert session.state == RecordingState.STOPPED

    def test_stop_without_start_raises(self, tmp_path):
        config = RecordingConfig(recordings_dir=str(tmp_path))
        engine = RecordingEngine(config, _make_setup())

        with pytest.raises(RuntimeError, match="No active recording"):
            engine.stop()

    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_status_while_recording(self, mock_ffmpeg, mock_sys, tmp_path):
        mock_ffmpeg.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        config = RecordingConfig(recordings_dir=str(tmp_path))
        engine = RecordingEngine(config, _make_setup())

        engine.start()
        status = engine.status()
        assert status["state"] == "recording"

    def test_status_when_idle(self, tmp_path):
        config = RecordingConfig(recordings_dir=str(tmp_path))
        engine = RecordingEngine(config, _make_setup())
        assert engine.status()["state"] == "idle"

    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_system_capture_failure_is_non_fatal(self, mock_ffmpeg, mock_sys, tmp_path):
        mock_ffmpeg.return_value = _mock_proc()
        mock_sys.return_value = None

        config = RecordingConfig(recordings_dir=str(tmp_path))
        engine = RecordingEngine(config, _make_setup())

        session = engine.start()
        names = [t.name for t in session.tracks]
        assert "mic" in names
        assert "system" not in names


class TestPauseResume:
    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_pause_stops_processes(self, mock_ffmpeg, mock_sys, tmp_path):
        proc = _mock_proc()
        mock_ffmpeg.return_value = proc
        mock_sys.return_value = _mock_proc()
        config = RecordingConfig(recordings_dir=str(tmp_path))
        engine = RecordingEngine(config, _make_setup())

        engine.start()
        engine.pause()
        assert engine.session.state == RecordingState.PAUSED

    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_resume_starts_new_segment(self, mock_ffmpeg, mock_sys, tmp_path):
        mock_ffmpeg.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        config = RecordingConfig(recordings_dir=str(tmp_path))
        engine = RecordingEngine(config, _make_setup())

        engine.start()
        assert engine.session.segment_index == 1

        engine.pause()
        engine.resume()
        assert engine.session.state == RecordingState.RECORDING
        assert engine.session.segment_index == 2

    @patch("recorder.recording.RecordingEngine._concat_segments")
    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_stop_from_paused(self, mock_ffmpeg, mock_sys, mock_concat, tmp_path):
        mock_ffmpeg.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        config = RecordingConfig(recordings_dir=str(tmp_path))
        engine = RecordingEngine(config, _make_setup())

        engine.start()
        engine.pause()
        session = engine.stop()
        assert session.state == RecordingState.STOPPED

    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_segments_tracked(self, mock_ffmpeg, mock_sys, tmp_path):
        mock_ffmpeg.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        config = RecordingConfig(recordings_dir=str(tmp_path))
        engine = RecordingEngine(config, _make_setup())

        engine.start()
        engine.pause()
        engine.resume()
        engine.pause()

        # Should have 2 segments per track
        assert len(engine.session.segments.get("mic", [])) == 2
        assert len(engine.session.segments.get("system", [])) == 2

    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_pause_when_not_recording_is_noop(self, mock_ffmpeg, mock_sys, tmp_path):
        mock_ffmpeg.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        config = RecordingConfig(recordings_dir=str(tmp_path))
        engine = RecordingEngine(config, _make_setup())

        engine.start()
        engine.pause()
        engine.pause()  # Should not crash
        assert engine.session.state == RecordingState.PAUSED


class TestPostHooks:
    @patch("recorder.recording.subprocess.run")
    @patch("recorder.recording.RecordingEngine._mix_tracks")
    @patch("recorder.recording.RecordingEngine._concat_segments")
    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_post_hooks_called_on_stop(self, mock_ffmpeg, mock_sys, mock_concat, mock_mix, mock_hook_run, tmp_path):
        mock_ffmpeg.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        config = RecordingConfig(
            recordings_dir=str(tmp_path),
            post_hooks=["echo {session_id}"],
        )
        engine = RecordingEngine(config, _make_setup())

        engine.start(label="hooktest")
        engine.stop()

        mock_hook_run.assert_called_once()
        assert "hooktest" in mock_hook_run.call_args[0][0]

    @patch("recorder.recording.subprocess.run", side_effect=Exception("hook failed"))
    @patch("recorder.recording.RecordingEngine._mix_tracks")
    @patch("recorder.recording.RecordingEngine._concat_segments")
    @patch("recorder.recording.RecordingEngine._start_system_capture")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_post_hook_failure_is_non_fatal(self, mock_ffmpeg, mock_sys, mock_concat, mock_mix, mock_hook_run, tmp_path):
        mock_ffmpeg.return_value = _mock_proc()
        mock_sys.return_value = _mock_proc()
        config = RecordingConfig(
            recordings_dir=str(tmp_path),
            post_hooks=["bad-command"],
        )
        engine = RecordingEngine(config, _make_setup())

        engine.start()
        session = engine.stop()
        assert session.state == RecordingState.STOPPED


class TestFfmpegCommand:
    @patch("recorder.recording.time.sleep")
    @patch("recorder.recording.subprocess.Popen")
    def test_ffmpeg_command_structure(self, mock_popen, mock_sleep, tmp_path):
        mock_popen.return_value = _mock_proc()
        config = RecordingConfig(
            recordings_dir=str(tmp_path),
            sample_rate=44100,
            channels=2,
            codec="pcm_s16le",
            bitrate="192k",
        )
        device = AudioDevice(1, "MacBook Air Microphone", DeviceType.MICROPHONE)
        engine = RecordingEngine(config, _make_setup())

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
        proc.communicate.return_value = (b"", b"Error: device not found")
        mock_popen.return_value = proc

        config = RecordingConfig(recordings_dir=str(tmp_path))
        engine = RecordingEngine(config, _make_setup())

        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            engine._start_ffmpeg(
                AudioDevice(99, "Nonexistent", DeviceType.MICROPHONE),
                tmp_path / "test.m4a",
            )


class TestSystemCapture:
    @patch("recorder.recording.time.sleep")
    @patch("recorder.recording.subprocess.Popen")
    def test_system_capture_starts(self, mock_popen, mock_sleep, tmp_path):
        mock_popen.return_value = _mock_proc()
        config = RecordingConfig(recordings_dir=str(tmp_path))
        engine = RecordingEngine(config, _make_setup())

        with patch("recorder.recording.SYSTEM_CAPTURE_BINARY", tmp_path / "fake"):
            (tmp_path / "fake").touch()
            assert engine._start_system_capture(tmp_path / "out.wav") is not None

    def test_missing_binary_returns_none(self, tmp_path):
        config = RecordingConfig(recordings_dir=str(tmp_path))
        engine = RecordingEngine(config, _make_setup())

        with patch("recorder.recording.SYSTEM_CAPTURE_BINARY", tmp_path / "nonexistent"):
            assert engine._start_system_capture(tmp_path / "out.wav") is None


class TestConcatSegments:
    def test_single_segment_renamed(self, tmp_path):
        config = RecordingConfig(recordings_dir=str(tmp_path))
        engine = RecordingEngine(config, _make_setup())
        engine.session = MagicMock()
        engine.session.output_dir = tmp_path
        engine.session.segments = {"mic": [tmp_path / "mic_seg000.m4a"]}
        engine.session.tracks = [MagicMock(name="mic")]
        engine.config = config

        # Create the segment file
        (tmp_path / "mic_seg000.m4a").write_bytes(b"fake audio data")

        engine._concat_segments()
        assert (tmp_path / "mic.m4a").exists()
        assert not (tmp_path / "mic_seg000.m4a").exists()
