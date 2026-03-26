"""Tests for recording engine logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from recorder.audio import AudioDevice, AudioSetup, DeviceType
from recorder.config import RecordingConfig
from recorder.recording import RecordingEngine, RecordingState


def _make_setup(
    has_mic: bool = True,
    has_blackhole: bool = True,
    has_external_mic: bool = False,
) -> AudioSetup:
    devices = []
    mic = None
    system_capture = None

    if has_mic:
        d = AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE)
        devices.append(d)
        mic = d

    if has_external_mic:
        d = AudioDevice(3, "External Microphone", DeviceType.MICROPHONE)
        devices.append(d)
        # If external mic is present and we want to use it, override mic
        mic = d

    if has_blackhole:
        d = AudioDevice(2, "BlackHole 2ch", DeviceType.VIRTUAL, is_input=True, is_output=True)
        devices.append(d)
        system_capture = d

    return AudioSetup(
        microphone=mic,
        system_capture=system_capture,
        all_devices=devices,
        headphones_connected=False,
        blackhole_available=has_blackhole,
    )


class TestTrackPlanning:
    def test_both_tracks_when_full_setup(self, tmp_path):
        config = RecordingConfig(recordings_dir=str(tmp_path))
        audio = _make_setup(has_mic=True, has_blackhole=True)
        engine = RecordingEngine(config, audio)

        tracks = engine._plan_tracks(tmp_path)
        names = [t.name for t in tracks]
        assert "mic" in names
        assert "system" in names

    def test_mic_only_when_no_blackhole(self, tmp_path):
        config = RecordingConfig(recordings_dir=str(tmp_path))
        audio = _make_setup(has_mic=True, has_blackhole=False)
        engine = RecordingEngine(config, audio)

        tracks = engine._plan_tracks(tmp_path)
        assert len(tracks) == 1
        assert tracks[0].name == "mic"

    def test_raises_when_no_devices(self, tmp_path):
        config = RecordingConfig(recordings_dir=str(tmp_path))
        audio = _make_setup(has_mic=False, has_blackhole=False)
        engine = RecordingEngine(config, audio)

        with pytest.raises(RuntimeError, match="No audio devices"):
            engine._plan_tracks(tmp_path)

    def test_output_paths_use_config_format(self, tmp_path):
        config = RecordingConfig(recordings_dir=str(tmp_path), format="wav")
        audio = _make_setup()
        engine = RecordingEngine(config, audio)

        tracks = engine._plan_tracks(tmp_path)
        for t in tracks:
            assert str(t.output_path).endswith(".wav")


class TestRecordingLifecycle:
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_start_creates_session(self, mock_ffmpeg, tmp_path):
        mock_ffmpeg.return_value = MagicMock(poll=MagicMock(return_value=None))
        config = RecordingConfig(recordings_dir=str(tmp_path))
        audio = _make_setup()
        engine = RecordingEngine(config, audio)

        session = engine.start(label="test")
        assert session.state == RecordingState.RECORDING
        assert "test" in session.session_id
        assert len(session.tracks) >= 1

    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_cannot_start_twice(self, mock_ffmpeg, tmp_path):
        mock_ffmpeg.return_value = MagicMock(poll=MagicMock(return_value=None))
        config = RecordingConfig(recordings_dir=str(tmp_path))
        audio = _make_setup()
        engine = RecordingEngine(config, audio)

        engine.start()
        with pytest.raises(RuntimeError, match="already in progress"):
            engine.start()

    @patch("recorder.recording.RecordingEngine._mix_tracks")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_stop_changes_state(self, mock_ffmpeg, mock_mix, tmp_path):
        proc = MagicMock()
        proc.poll.return_value = None
        proc.communicate.return_value = (b"", b"")
        mock_ffmpeg.return_value = proc

        config = RecordingConfig(recordings_dir=str(tmp_path))
        audio = _make_setup()
        engine = RecordingEngine(config, audio)

        engine.start()
        session = engine.stop()
        assert session.state == RecordingState.STOPPED

    def test_stop_without_start_raises(self, tmp_path):
        config = RecordingConfig(recordings_dir=str(tmp_path))
        audio = _make_setup()
        engine = RecordingEngine(config, audio)

        with pytest.raises(RuntimeError, match="No active recording"):
            engine.stop()

    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_status_while_recording(self, mock_ffmpeg, tmp_path):
        mock_ffmpeg.return_value = MagicMock(poll=MagicMock(return_value=None))
        config = RecordingConfig(recordings_dir=str(tmp_path))
        audio = _make_setup()
        engine = RecordingEngine(config, audio)

        engine.start()
        status = engine.status()
        assert status["state"] == "recording"
        assert len(status["tracks"]) >= 1

    def test_status_when_idle(self, tmp_path):
        config = RecordingConfig(recordings_dir=str(tmp_path))
        audio = _make_setup()
        engine = RecordingEngine(config, audio)

        status = engine.status()
        assert status["state"] == "idle"


class TestPostHooks:
    @patch("recorder.recording.subprocess.run")
    @patch("recorder.recording.RecordingEngine._mix_tracks")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_post_hooks_called_on_stop(self, mock_ffmpeg, mock_mix, mock_hook_run, tmp_path):
        proc = MagicMock()
        proc.poll.return_value = None
        proc.communicate.return_value = (b"", b"")
        mock_ffmpeg.return_value = proc

        config = RecordingConfig(
            recordings_dir=str(tmp_path),
            post_hooks=["echo {session_id}"],
        )
        audio = _make_setup()
        engine = RecordingEngine(config, audio)

        engine.start(label="hooktest")
        engine.stop()

        mock_hook_run.assert_called_once()
        call_cmd = mock_hook_run.call_args[0][0]
        assert "hooktest" in call_cmd

    @patch("recorder.recording.subprocess.run", side_effect=Exception("hook failed"))
    @patch("recorder.recording.RecordingEngine._mix_tracks")
    @patch("recorder.recording.RecordingEngine._start_ffmpeg")
    def test_post_hook_failure_is_non_fatal(self, mock_ffmpeg, mock_mix, mock_hook_run, tmp_path):
        proc = MagicMock()
        proc.poll.return_value = None
        proc.communicate.return_value = (b"", b"")
        mock_ffmpeg.return_value = proc

        config = RecordingConfig(
            recordings_dir=str(tmp_path),
            post_hooks=["bad-command"],
        )
        audio = _make_setup()
        engine = RecordingEngine(config, audio)

        engine.start()
        # Should not raise even if hook fails
        session = engine.stop()
        assert session.state == RecordingState.STOPPED


class TestFfmpegCommand:
    @patch("recorder.recording.time.sleep")
    @patch("recorder.recording.subprocess.Popen")
    def test_ffmpeg_command_structure(self, mock_popen, mock_sleep, tmp_path):
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc

        config = RecordingConfig(
            recordings_dir=str(tmp_path),
            sample_rate=44100,
            channels=2,
            codec="pcm_s16le",
            bitrate="192k",
        )
        device = AudioDevice(1, "MacBook Air Microphone", DeviceType.MICROPHONE)
        audio = _make_setup()
        engine = RecordingEngine(config, audio)

        output_path = tmp_path / "test.wav"
        engine._start_ffmpeg(device, output_path)

        cmd = mock_popen.call_args[0][0]
        assert "ffmpeg" in cmd[0]
        assert "-f" in cmd
        assert "avfoundation" in cmd
        assert ":1" in cmd  # device index
        assert "44100" in cmd
        assert "2" in [str(c) for c in cmd]  # channels

    @patch("recorder.recording.time.sleep")
    @patch("recorder.recording.subprocess.Popen")
    def test_ffmpeg_failure_raises(self, mock_popen, mock_sleep, tmp_path):
        proc = MagicMock()
        proc.poll.return_value = 1  # Exited with error
        proc.communicate.return_value = (b"", b"Error: device not found")
        mock_popen.return_value = proc

        config = RecordingConfig(recordings_dir=str(tmp_path))
        device = AudioDevice(99, "Nonexistent", DeviceType.MICROPHONE)
        audio = _make_setup()
        engine = RecordingEngine(config, audio)

        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            engine._start_ffmpeg(device, tmp_path / "test.m4a")
