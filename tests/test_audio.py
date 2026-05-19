"""Tests for audio device detection logic."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from recorder.audio import (
    AudioDevice,
    DeviceType,
    _classify_device,
    _find_device_by_name,
    detect_audio_setup,
    detect_ffmpeg_devices,
)
from recorder.config import AudioProfile


FFMPEG_OUTPUT_FULL = """\
[AVFoundation indev @ 0x1234] AVFoundation video devices:
[AVFoundation indev @ 0x1234] [0] MacBook Air Camera
[AVFoundation indev @ 0x1234] AVFoundation audio devices:
[AVFoundation indev @ 0x1234] [0] iPhone Microphone
[AVFoundation indev @ 0x1234] [1] MacBook Air Microphone
[AVFoundation indev @ 0x1234] [2] External Microphone
"""

FFMPEG_OUTPUT_ONE_MIC = """\
[AVFoundation indev @ 0x1234] AVFoundation audio devices:
[AVFoundation indev @ 0x1234] [0] MacBook Air Microphone
"""

FFMPEG_OUTPUT_NO_DEVICES = """\
[AVFoundation indev @ 0x1234] AVFoundation video devices:
[AVFoundation indev @ 0x1234] [0] MacBook Air Camera
[AVFoundation indev @ 0x1234] AVFoundation audio devices:
"""


def _mock_ffmpeg_run(stderr_output: str):
    result = MagicMock()
    result.stderr = stderr_output
    return result


class TestClassifyDevice:
    def test_builtin_mic(self):
        assert _classify_device("MacBook Air Microphone") == DeviceType.MICROPHONE

    def test_iphone_mic(self):
        assert _classify_device("iPhone Microphone") == DeviceType.MICROPHONE

    def test_external_mic(self):
        assert _classify_device("External Microphone") == DeviceType.MICROPHONE

    def test_unknown_defaults_to_speaker(self):
        assert _classify_device("Some Random Device") == DeviceType.SPEAKER


class TestDetectDevices:
    @patch("recorder.audio.subprocess.run")
    def test_full_device_list(self, mock_run):
        mock_run.return_value = _mock_ffmpeg_run(FFMPEG_OUTPUT_FULL)
        devices = detect_ffmpeg_devices()
        assert len(devices) == 3
        names = [d.name for d in devices]
        assert "iPhone Microphone" in names
        assert "MacBook Air Microphone" in names
        assert "External Microphone" in names

    @patch("recorder.audio.subprocess.run")
    def test_no_audio_devices(self, mock_run):
        mock_run.return_value = _mock_ffmpeg_run(FFMPEG_OUTPUT_NO_DEVICES)
        devices = detect_ffmpeg_devices()
        assert len(devices) == 0

    @patch("recorder.audio.subprocess.run", side_effect=FileNotFoundError)
    def test_ffmpeg_not_found(self, mock_run):
        with pytest.raises(RuntimeError, match="ffmpeg not found"):
            detect_ffmpeg_devices()

    @patch("recorder.audio.subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 10))
    def test_ffmpeg_timeout(self, mock_run):
        with pytest.raises(RuntimeError, match="timed out"):
            detect_ffmpeg_devices()


class TestAudioDeviceProperties:
    def test_is_builtin_mic(self):
        d = AudioDevice(1, "MacBook Air Microphone", DeviceType.MICROPHONE)
        assert d.is_builtin_mic

    def test_is_not_builtin_mic(self):
        d = AudioDevice(0, "iPhone Microphone", DeviceType.MICROPHONE)
        assert not d.is_builtin_mic


class TestFindDeviceByName:
    def test_finds_by_substring(self):
        devices = [
            AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE),
            AudioDevice(1, "External Microphone", DeviceType.MICROPHONE),
        ]
        result = _find_device_by_name(devices, "External")
        assert result is not None
        assert result.name == "External Microphone"

    def test_case_insensitive(self):
        devices = [AudioDevice(0, "External Microphone", DeviceType.MICROPHONE)]
        result = _find_device_by_name(devices, "external")
        assert result is not None

    def test_not_found(self):
        devices = [AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE)]
        result = _find_device_by_name(devices, "Nonexistent")
        assert result is None

    def test_empty_name(self):
        devices = [AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE)]
        result = _find_device_by_name(devices, "")
        assert result is None


class TestDetectAudioSetup:
    @patch("recorder.audio.detect_headphones", return_value=False)
    @patch("recorder.audio.detect_ffmpeg_devices")
    def test_default_setup_picks_builtin_mic(self, mock_devices, mock_hp):
        mock_devices.return_value = [
            AudioDevice(0, "iPhone Microphone", DeviceType.MICROPHONE),
            AudioDevice(1, "MacBook Air Microphone", DeviceType.MICROPHONE),
        ]
        setup = detect_audio_setup()
        assert setup.microphone.name == "MacBook Air Microphone"
        assert setup.can_record_mic

    @patch("recorder.audio.detect_headphones", return_value=True)
    @patch("recorder.audio.detect_ffmpeg_devices")
    def test_headphones_broken_mic_profile(self, mock_devices, mock_hp):
        mock_devices.return_value = [
            AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE),
        ]
        profile = AudioProfile(
            name="headphones-broken-mic",
            description="test",
            preferred_mic="MacBook Air Microphone",
        )
        setup = detect_audio_setup(profile)
        assert setup.microphone.name == "MacBook Air Microphone"

    @patch("recorder.audio.detect_headphones", return_value=True)
    @patch("recorder.audio.detect_ffmpeg_devices")
    def test_headphones_profile_picks_external_mic(self, mock_devices, mock_hp):
        mock_devices.return_value = [
            AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE),
            AudioDevice(1, "External Microphone", DeviceType.MICROPHONE),
        ]
        profile = AudioProfile(
            name="headphones", description="test", preferred_mic="",
        )
        setup = detect_audio_setup(profile)
        assert setup.microphone.name == "External Microphone"

    @patch("recorder.audio.detect_headphones", return_value=True)
    @patch("recorder.audio.detect_ffmpeg_devices")
    def test_headphones_profile_fallback_to_builtin(self, mock_devices, mock_hp):
        mock_devices.return_value = [
            AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE),
        ]
        profile = AudioProfile(name="headphones", description="test", preferred_mic="")
        setup = detect_audio_setup(profile)
        assert setup.microphone.name == "MacBook Air Microphone"

    @patch("recorder.audio.detect_headphones", return_value=False)
    @patch("recorder.audio.detect_ffmpeg_devices")
    def test_headphones_flag_propagated(self, mock_devices, mock_hp):
        mock_devices.return_value = [
            AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE),
        ]
        setup = detect_audio_setup()
        assert setup.headphones_connected is False
