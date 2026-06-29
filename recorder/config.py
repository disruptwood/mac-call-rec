"""Configuration management for call recorder."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".call-recorder" / "config.json"
DEFAULT_RECORDINGS_DIR = Path.home() / ".call-recorder" / "recordings"

_SLUG_SAFE_RE = re.compile(r"[^a-z0-9._-]+")
_SLUG_DASH_RE = re.compile(r"[-_.]{2,}")
_CYRILLIC_TRANSLIT = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


def slugify_label(value: str, *, fallback: str = "session", max_length: int = 64) -> str:
    """Convert a display name into an ASCII label safe for directory names."""
    value = value.strip().lower()
    transliterated = "".join(_CYRILLIC_TRANSLIT.get(ch, ch) for ch in value)
    normalized = unicodedata.normalize("NFKD", transliterated)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_SAFE_RE.sub("-", ascii_text)
    slug = _SLUG_DASH_RE.sub("-", slug).strip("-_.")
    if not slug:
        slug = fallback
    return slug[:max_length].strip("-_.") or fallback


@dataclass
class SessionType:
    """A predefined session preset shown in the interactive start menu.

    `name` is what the user sees in the picker (e.g. "Терапия").
    `label` is the filename-safe slug stitched into the session directory
    name (e.g. "therapy_20260522_140000"). Keep `label` ASCII / lowercase /
    hyphenated — it ends up in paths and filenames.
    """
    name: str
    label: str

    @staticmethod
    def defaults() -> list[SessionType]:
        return [
            SessionType(name="Терапия", label="therapy"),
            SessionType(name="Созвон", label="meeting"),
            SessionType(name="Интервью", label="interview"),
        ]


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
                preferred_mic="MacBook",
            ),
            "headphones": AudioProfile(
                name="headphones",
                description="Headphones with working mic — use headphone mic",
                preferred_mic="",
            ),
            "speaker": AudioProfile(
                name="speaker",
                description="No headphones — Mac speakers and Mac mic",
                preferred_mic="MacBook",
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
    transcription_backend: str = "none"

    # Session presets shown in the interactive picker. Stored as raw dicts in
    # JSON so the user can edit them by hand without re-encoding a custom
    # dataclass. Empty list = use SessionType.defaults().
    session_types: list[dict] = field(default_factory=list)

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

    def list_session_types(self) -> list[SessionType]:
        """Return configured session presets, falling back to defaults
        when nothing is configured. Caller is responsible for appending the
        "custom" option in the UI layer — it's not stored as a SessionType
        because it has no fixed label."""
        if not self.session_types:
            return SessionType.defaults()
        return [SessionType(**d) for d in self.session_types]

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
