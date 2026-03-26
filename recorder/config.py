"""Configuration management for call recorder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".call-recorder" / "config.json"
DEFAULT_RECORDINGS_DIR = Path.home() / ".call-recorder" / "recordings"


@dataclass
class AudioProfile:
    """A named audio configuration for a specific setup."""
    name: str
    description: str
    preferred_mic: str  # Substring match against device name
    preferred_system_capture: str = "BlackHole"

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
                preferred_mic="",  # Will be resolved to headphone mic at runtime
            ),
            "speaker": AudioProfile(
                name="speaker",
                description="No headphones — Mac speakers and Mac mic",
                preferred_mic="MacBook Air Microphone",
            ),
        }


@dataclass
class RecordingConfig:
    # Audio format
    sample_rate: int = 48000
    channels: int = 1
    codec: str = "aac"
    format: str = "m4a"
    bitrate: str = "128k"

    # Active profile name
    active_profile: str = "headphones-broken-mic"

    # Custom profiles (serialized as dicts)
    profiles: dict[str, dict] = field(default_factory=dict)

    # Recording behavior
    separate_tracks: bool = True  # Record mic and system as separate files
    mix_tracks: bool = True       # Also produce a mixed file
    recordings_dir: str = str(DEFAULT_RECORDINGS_DIR)

    # Post-recording hooks (commands to run after recording finishes)
    post_hooks: list[str] = field(default_factory=list)

    def get_profile(self, name: str | None = None) -> AudioProfile:
        name = name or self.active_profile
        defaults = AudioProfile.defaults()

        # Check custom profiles first
        if name in self.profiles:
            d = self.profiles[name]
            return AudioProfile(**d)

        # Then defaults
        if name in defaults:
            return defaults[name]

        raise ValueError(
            f"Unknown profile '{name}'. "
            f"Available: {', '.join(list(defaults.keys()) + list(self.profiles.keys()))}"
        )

    def list_profiles(self) -> list[AudioProfile]:
        defaults = AudioProfile.defaults()
        all_profiles = []
        for p in defaults.values():
            all_profiles.append(p)
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
