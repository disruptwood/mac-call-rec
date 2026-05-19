"""Configuration management for call recorder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".call-recorder" / "config.json"
DEFAULT_RECORDINGS_DIR = Path.home() / ".call-recorder" / "recordings"


@dataclass
class AudioProfile:
    """A named audio configuration for a specific setup.

    `preferred_mic` is a substring matched against device names returned by
    `detect_ffmpeg_devices`. Empty string means "auto: pick first non-builtin
    mic, fall back to builtin".
    """
    name: str
    description: str
    preferred_mic: str

    @staticmethod
    def defaults() -> dict[str, AudioProfile]:
        return {
            "headphones-broken-mic": AudioProfile(
                name="headphones-broken-mic",
                description="Headphones for listening, Mac mic for voice (broken headphone mic)",
                preferred_mic="MacBook Air Microphone",
            ),
            "headphones": AudioProfile(
                name="headphones",
                description="Headphones with working mic — use headphone mic",
                preferred_mic="",
            ),
            "speaker": AudioProfile(
                name="speaker",
                description="No headphones — Mac speakers and Mac mic",
                preferred_mic="MacBook Air Microphone",
            ),
        }


@dataclass
class RecordingConfig:
    # Audio format for the mixed recording.m4a output. Source tracks (_mic.wav,
    # _system.wav, _mic_pa.wav) are always raw PCM regardless of these values.
    sample_rate: int = 48000
    channels: int = 1
    codec: str = "aac"
    format: str = "m4a"
    bitrate: str = "128k"

    active_profile: str = "headphones-broken-mic"
    profiles: dict[str, dict] = field(default_factory=dict)
    recordings_dir: str = str(DEFAULT_RECORDINGS_DIR)

    def get_profile(self, name: str | None = None) -> AudioProfile:
        name = name or self.active_profile
        defaults = AudioProfile.defaults()

        if name in self.profiles:
            d = self.profiles[name]
            return AudioProfile(**d)

        if name in defaults:
            return defaults[name]

        raise ValueError(
            f"Unknown profile '{name}'. "
            f"Available: {', '.join(list(defaults.keys()) + list(self.profiles.keys()))}"
        )

    def list_profiles(self) -> list[AudioProfile]:
        defaults = AudioProfile.defaults()
        all_profiles = list(defaults.values())
        for name, d in self.profiles.items():
            if name not in defaults:
                all_profiles.append(AudioProfile(**d))
        return all_profiles

    def save(self, path: Path | None = None) -> None:
        path = path or DEFAULT_CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path | None = None) -> RecordingConfig:
        path = path or DEFAULT_CONFIG_PATH
        if not path.exists():
            config = cls()
            config.save(path)
            return config
        data = json.loads(path.read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def output_dir(self) -> Path:
        p = Path(self.recordings_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
