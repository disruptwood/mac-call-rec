"""Tests for audio device detection logic."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import subprocess

import pytest

from recorder.audio import (
    AudioDevice,
    AudioSetup,
    DeviceType,
    _classify_device,
    _find_device_by_name,
    detect_audio_setup,
    detect_ffmpeg_devices,
    detect_headphones,
)
from recorder.config import AudioProfile


# --- Fixtures ---

FFMPEG_OUTPUT_FULL = """\
[AVFoundation indev @ 0x1234] AVFoundation video devices:
[AVFoundation indev @ 0x1234] [0] MacBook Air Camera
[AVFoundation indev @ 0x1234] AVFoundation audio devices:
[AVFoundation indev @ 0x1234] [0] iPhone Microphone
[AVFoundation indev @ 0x1234] [1] MacBook Air Microphone
[AVFoundation indev @ 0x1234] [2] BlackHole 2ch
[AVFoundation indev @ 0x1234] [3] ZoomAudioDevice
"""

FFMPEG_OUTPUT_NO_BLACKHOLE = """\
[AVFoundation indev @ 0x1234] AVFoundation audio devices:
[AVFoundation indev @ 0x1234] [0] MacBook Air Microphone
"""

FFMPEG_OUTPUT_HEADPHONE_MIC = """\
[AVFoundation indev @ 0x1234] AVFoundation audio devices:
[AVFoundation indev @ 0x1234] [0] MacBook Air Microphone
[AVFoundation indev @ 0x1234] [1] External Microphone
[AVFoundation indev @ 0x1234] [2] BlackHole 2ch
"""

FFMPEG_OUTPUT_NO_DEVICES = """\
[AVFoundation indev @ 0x1234] AVFoundation video devices:
[AVFoundation indev @ 0x1234] [0] MacBook Air Camera
[AVFoundation indev @ 0x1234] AVFoundation audio devices:
"""


def _mock_ffmpeg_run(stderr_output: str):
    """Create a mock for subprocess.run that returns given ffmpeg output."""
    result = MagicMock()
    result.stderr = stderr_output
    return result


# --- Device classification ---

class TestClassifyDevice:
    def test_builtin_mic(self):
        assert _classify_device("MacBook Air Microphone") == DeviceType.MICROPHONE

    def test_iphone_mic(self):
        assert _classify_device("iPhone Microphone") == DeviceType.MICROPHONE

    def test_external_mic(self):
        assert _classify_device("External Microphone") == DeviceType.MICROPHONE

    def test_blackhole(self):
        assert _classify_device("BlackHole 2ch") == DeviceType.VIRTUAL

    def test_zoom(self):
        assert _classify_device("ZoomAudioDevice") == DeviceType.VIRTUAL

    def test_soundflower(self):
        assert _classify_device("Soundflower (2ch)") == DeviceType.VIRTUAL

    def test_unknown_defaults_to_speaker(self):
        assert _classify_device("Some Random Device") == DeviceType.SPEAKER


# --- Device detection ---

class TestDetectDevices:
    @patch("recorder.audio.subprocess.run")
    def test_full_device_list(self, mock_run):
        mock_run.return_value = _mock_ffmpeg_run(FFMPEG_OUTPUT_FULL)
        devices = detect_ffmpeg_devices()

        assert len(devices) == 4
        names = [d.name for d in devices]
        assert "iPhone Microphone" in names
        assert "MacBook Air Microphone" in names
        assert "BlackHole 2ch" in names
        assert "ZoomAudioDevice" in names

    @patch("recorder.audio.subprocess.run")
    def test_no_blackhole(self, mock_run):
        mock_run.return_value = _mock_ffmpeg_run(FFMPEG_OUTPUT_NO_BLACKHOLE)
        devices = detect_ffmpeg_devices()

        assert len(devices) == 1
        assert devices[0].name == "MacBook Air Microphone"
        assert not any(d.is_blackhole for d in devices)

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


# --- AudioDevice properties ---

class TestAudioDeviceProperties:
    def test_is_blackhole(self):
        d = AudioDevice(2, "BlackHole 2ch", DeviceType.VIRTUAL)
        assert d.is_blackhole

    def test_is_not_blackhole(self):
        d = AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE)
        assert not d.is_blackhole

    def test_is_builtin_mic(self):
        d = AudioDevice(1, "MacBook Air Microphone", DeviceType.MICROPHONE)
        assert d.is_builtin_mic

    def test_is_not_builtin_mic(self):
        d = AudioDevice(0, "iPhone Microphone", DeviceType.MICROPHONE)
        assert not d.is_builtin_mic

    def test_is_builtin_speaker(self):
        d = AudioDevice(0, "MacBook Air Speakers", DeviceType.SPEAKER)
        assert d.is_builtin_speaker


# --- Find device by name ---

class TestFindDeviceByName:
    def test_finds_by_substring(self):
        devices = [
            AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE),
            AudioDevice(1, "BlackHole 2ch", DeviceType.VIRTUAL),
        ]
        result = _find_device_by_name(devices, "BlackHole")
        assert result is not None
        assert result.name == "BlackHole 2ch"

    def test_case_insensitive(self):
        devices = [AudioDevice(0, "BlackHole 2ch", DeviceType.VIRTUAL)]
        result = _find_device_by_name(devices, "blackhole")
        assert result is not None

    def test_not_found(self):
        devices = [AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE)]
        result = _find_device_by_name(devices, "BlackHole")
        assert result is None

    def test_empty_name(self):
        devices = [AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE)]
        result = _find_device_by_name(devices, "")
        assert result is None


# --- Audio setup with profiles ---

class TestDetectAudioSetup:
    @patch("recorder.audio.detect_headphones", return_value=False)
    @patch("recorder.audio.detect_ffmpeg_devices")
    def test_default_setup_picks_builtin_mic(self, mock_devices, mock_hp):
        mock_devices.return_value = [
            AudioDevice(0, "iPhone Microphone", DeviceType.MICROPHONE),
            AudioDevice(1, "MacBook Air Microphone", DeviceType.MICROPHONE),
            AudioDevice(2, "BlackHole 2ch", DeviceType.VIRTUAL),
        ]
        setup = detect_audio_setup()
        assert setup.microphone.name == "MacBook Air Microphone"
        assert setup.system_capture.name == "BlackHole 2ch"
        assert setup.blackhole_available

    @patch("recorder.audio.detect_headphones", return_value=True)
    @patch("recorder.audio.detect_ffmpeg_devices")
    def test_headphones_broken_mic_profile(self, mock_devices, mock_hp):
        """With broken headphone mic, still use MacBook mic."""
        mock_devices.return_value = [
            AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE),
            AudioDevice(1, "BlackHole 2ch", DeviceType.VIRTUAL),
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
        """Headphones profile with empty preferred_mic picks external mic."""
        mock_devices.return_value = [
            AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE),
            AudioDevice(1, "External Microphone", DeviceType.MICROPHONE),
            AudioDevice(2, "BlackHole 2ch", DeviceType.VIRTUAL),
        ]
        profile = AudioProfile(
            name="headphones",
            description="test",
            preferred_mic="",  # empty = prefer external
        )
        setup = detect_audio_setup(profile)
        assert setup.microphone.name == "External Microphone"

    @patch("recorder.audio.detect_headphones", return_value=False)
    @patch("recorder.audio.detect_ffmpeg_devices")
    def test_no_blackhole_still_has_mic(self, mock_devices, mock_hp):
        mock_devices.return_value = [
            AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE),
        ]
        setup = detect_audio_setup()
        assert setup.can_record_mic
        assert not setup.can_record_system
        assert not setup.can_record_both

    @patch("recorder.audio.detect_headphones", return_value=False)
    @patch("recorder.audio.detect_ffmpeg_devices")
    def test_full_setup_can_record_both(self, mock_devices, mock_hp):
        mock_devices.return_value = [
            AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE),
            AudioDevice(1, "BlackHole 2ch", DeviceType.VIRTUAL),
        ]
        setup = detect_audio_setup()
        assert setup.can_record_both

    @patch("recorder.audio.detect_headphones", return_value=True)
    @patch("recorder.audio.detect_ffmpeg_devices")
    def test_headphones_profile_fallback_to_builtin(self, mock_devices, mock_hp):
        """If no external mic found, headphones profile falls back to builtin."""
        mock_devices.return_value = [
            AudioDevice(0, "MacBook Air Microphone", DeviceType.MICROPHONE),
            AudioDevice(1, "BlackHole 2ch", DeviceType.VIRTUAL),
        ]
        profile = AudioProfile(name="headphones", description="test", preferred_mic="")
        setup = detect_audio_setup(profile)
        # Falls back to builtin since no external mic
        assert setup.microphone.name == "MacBook Air Microphone"
