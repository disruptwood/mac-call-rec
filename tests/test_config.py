"""Tests for configuration management."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recorder.config import AudioProfile, RecordingConfig


class TestAudioProfile:
    def test_defaults_has_three_profiles(self):
        defaults = AudioProfile.defaults()
        assert "headphones-broken-mic" in defaults
        assert "headphones" in defaults
        assert "speaker" in defaults

    def test_broken_mic_uses_macbook_mic(self):
        profile = AudioProfile.defaults()["headphones-broken-mic"]
        assert "MacBook" in profile.preferred_mic

    def test_headphones_has_empty_mic(self):
        """Empty preferred_mic means auto-detect external mic."""
        profile = AudioProfile.defaults()["headphones"]
        assert profile.preferred_mic == ""

    def test_speaker_uses_macbook_mic(self):
        profile = AudioProfile.defaults()["speaker"]
        assert "MacBook" in profile.preferred_mic


class TestRecordingConfig:
    def test_defaults(self):
        config = RecordingConfig()
        assert config.active_profile == "headphones-broken-mic"
        assert config.sample_rate == 48000
        assert config.format == "m4a"

    def test_save_and_load(self, tmp_path):
        config_path = tmp_path / "config.json"
        config = RecordingConfig()
        config.bitrate = "256k"
        config.save(config_path)

        loaded = RecordingConfig.load(config_path)
        assert loaded.bitrate == "256k"
        assert loaded.active_profile == "headphones-broken-mic"

    def test_load_creates_default_if_missing(self, tmp_path):
        config_path = tmp_path / "config.json"
        assert not config_path.exists()

        config = RecordingConfig.load(config_path)
        assert config_path.exists()
        assert config.active_profile == "headphones-broken-mic"

    def test_load_ignores_unknown_keys(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "sample_rate": 44100,
            "unknown_future_key": "whatever",
        }))
        config = RecordingConfig.load(config_path)
        assert config.sample_rate == 44100

    def test_get_profile_default(self):
        config = RecordingConfig()
        profile = config.get_profile()
        assert profile.name == "headphones-broken-mic"

    def test_get_profile_by_name(self):
        config = RecordingConfig()
        profile = config.get_profile("speaker")
        assert profile.name == "speaker"

    def test_get_profile_unknown_raises(self):
        config = RecordingConfig()
        with pytest.raises(ValueError, match="Unknown profile"):
            config.get_profile("nonexistent")

    def test_custom_profile(self):
        config = RecordingConfig()
        config.profiles["studio"] = {
            "name": "studio",
            "description": "Studio mic setup",
            "preferred_mic": "Blue Yeti",
        }
        profile = config.get_profile("studio")
        assert profile.preferred_mic == "Blue Yeti"

    def test_list_profiles_includes_defaults_and_custom(self):
        config = RecordingConfig()
        config.profiles["studio"] = {
            "name": "studio",
            "description": "Studio",
            "preferred_mic": "Blue Yeti",
        }
        profiles = config.list_profiles()
        names = [p.name for p in profiles]
        assert "headphones-broken-mic" in names
        assert "headphones" in names
        assert "speaker" in names
        assert "studio" in names

    def test_output_dir_creates_directory(self, tmp_path):
        config = RecordingConfig(recordings_dir=str(tmp_path / "recs"))
        output = config.output_dir
        assert output.exists()
