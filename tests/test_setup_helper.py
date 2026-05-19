"""Tests for setup helper utilities."""

from __future__ import annotations

from unittest.mock import patch

from recorder.setup_helper import check_ffmpeg_installed, print_setup_instructions


class TestCheckFfmpegInstalled:
    @patch("recorder.setup_helper.subprocess.run")
    def test_ffmpeg_found(self, mock_run):
        assert check_ffmpeg_installed() is True

    @patch("recorder.setup_helper.subprocess.run", side_effect=FileNotFoundError)
    def test_ffmpeg_not_found(self, mock_run):
        assert check_ffmpeg_installed() is False


class TestPrintSetupInstructions:
    def test_with_headphones_runs_clean(self, capsys):
        print_setup_instructions(headphones=True)
        out = capsys.readouterr().out
        assert "ScreenCaptureKit" in out
        assert "headphones" in out

    def test_without_headphones_runs_clean(self, capsys):
        print_setup_instructions(headphones=False)
        out = capsys.readouterr().out
        assert "ScreenCaptureKit" in out
        assert "MacBook speakers" in out
