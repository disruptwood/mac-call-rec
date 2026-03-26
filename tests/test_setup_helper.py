"""Tests for setup helper utilities."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from recorder.setup_helper import (
    check_blackhole_loaded,
    check_ffmpeg_installed,
    check_multi_output_device,
)


class TestCheckBlackholeLoaded:
    @patch("recorder.setup_helper.subprocess.run")
    def test_blackhole_present(self, mock_run):
        result = MagicMock()
        result.stderr = "[AVFoundation indev] [2] BlackHole 2ch"
        mock_run.return_value = result
        assert check_blackhole_loaded() is True

    @patch("recorder.setup_helper.subprocess.run")
    def test_blackhole_absent(self, mock_run):
        result = MagicMock()
        result.stderr = "[AVFoundation indev] [0] MacBook Air Microphone"
        mock_run.return_value = result
        assert check_blackhole_loaded() is False

    @patch("recorder.setup_helper.subprocess.run", side_effect=Exception("boom"))
    def test_error_returns_false(self, mock_run):
        assert check_blackhole_loaded() is False


class TestCheckFfmpegInstalled:
    @patch("recorder.setup_helper.subprocess.run")
    def test_ffmpeg_found(self, mock_run):
        assert check_ffmpeg_installed() is True

    @patch("recorder.setup_helper.subprocess.run", side_effect=FileNotFoundError)
    def test_ffmpeg_not_found(self, mock_run):
        assert check_ffmpeg_installed() is False


class TestCheckMultiOutputDevice:
    @patch("recorder.setup_helper.subprocess.run")
    def test_multi_output_exists(self, mock_run):
        result = MagicMock()
        result.stdout = "Multi-Output Device:\n  Output Channels: 2"
        mock_run.return_value = result
        assert check_multi_output_device() is True

    @patch("recorder.setup_helper.subprocess.run")
    def test_no_multi_output(self, mock_run):
        result = MagicMock()
        result.stdout = "MacBook Air Speakers:\n  Output Channels: 2"
        mock_run.return_value = result
        assert check_multi_output_device() is False
