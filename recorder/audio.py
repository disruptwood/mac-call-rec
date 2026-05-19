"""Audio device detection for macOS via AVFoundation/ffmpeg.

We use ffmpeg's avfoundation device listing to enumerate inputs (microphones
and virtual outputs), pick the preferred mic for recording, and detect whether
headphones are currently the default output (so we can warn the user).

System audio capture happens through Swift + ScreenCaptureKit
(scripts/capture_system_audio), not through a virtual audio device, so we no
longer need to track BlackHole or Multi-Output Device state.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import AudioProfile


class DeviceType(Enum):
    MICROPHONE = "microphone"
    SPEAKER = "speaker"


@dataclass
class AudioDevice:
    index: int
    name: str
    device_type: DeviceType
    is_input: bool = True

    @property
    def is_builtin_mic(self) -> bool:
        return "macbook" in self.name.lower() and "microphone" in self.name.lower()


@dataclass
class AudioSetup:
    """Detected audio configuration."""
    microphone: AudioDevice | None = None
    all_devices: list[AudioDevice] = field(default_factory=list)
    headphones_connected: bool = False

    @property
    def can_record_mic(self) -> bool:
        return self.microphone is not None


def detect_ffmpeg_devices() -> list[AudioDevice]:
    """Query ffmpeg for available AVFoundation audio devices."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg device listing timed out")

    output = result.stderr
    devices = []
    in_audio_section = False

    for line in output.splitlines():
        if "AVFoundation audio devices:" in line:
            in_audio_section = True
            continue
        if not in_audio_section:
            continue

        match = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if not match:
            continue

        index = int(match.group(1))
        name = match.group(2).strip()
        device_type = _classify_device(name)

        devices.append(AudioDevice(
            index=index,
            name=name,
            device_type=device_type,
            is_input=device_type == DeviceType.MICROPHONE,
        ))

    return devices


def _classify_device(name: str) -> DeviceType:
    lower = name.lower()
    if "microphone" in lower or "mic" in lower:
        return DeviceType.MICROPHONE
    return DeviceType.SPEAKER


def detect_headphones() -> bool:
    """Check if headphones/external audio output is connected.

    Used only for an informational warning at session start — the recording
    pipeline does not branch on headphone state. The user controls the physical
    setup themselves.
    """
    try:
        result = subprocess.run(
            ["system_profiler", "SPAudioDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        data = json.loads(result.stdout)
        items = data.get("SPAudioDataType", [])
        for item in items:
            for device in item.get("_items", []):
                name = device.get("_name", "").lower()
                transport = device.get("coreaudio_device_transport", "").lower()
                if transport in ("usb", "bluetooth", "wireless") and "output" not in name:
                    continue
                if any(kw in name for kw in ["headphone", "airpod", "external", "usb"]):
                    return True
                if transport in ("usb", "bluetooth") and device.get("coreaudio_output_source"):
                    return True
    except Exception:
        pass

    return False


def _find_device_by_name(devices: list[AudioDevice], name_substring: str) -> AudioDevice | None:
    if not name_substring:
        return None
    lower = name_substring.lower()
    for d in devices:
        if lower in d.name.lower():
            return d
    return None


def detect_audio_setup(profile: AudioProfile | None = None) -> AudioSetup:
    """Detect mic + headphones state, optionally guided by a profile.

    If profile.preferred_mic is set: pick that device by name substring.
    If empty: prefer the first non-builtin mic (typically a headphone mic).
    Fallback: built-in MacBook mic, then any mic.
    """
    devices = detect_ffmpeg_devices()
    headphones = detect_headphones()

    mic = None
    if profile and profile.preferred_mic:
        mic = _find_device_by_name(devices, profile.preferred_mic)

    if mic is None and profile and not profile.preferred_mic:
        for d in devices:
            if d.device_type == DeviceType.MICROPHONE and not d.is_builtin_mic:
                mic = d
                break

    if mic is None:
        for d in devices:
            if d.is_builtin_mic:
                mic = d
                break

    if mic is None:
        for d in devices:
            if d.device_type == DeviceType.MICROPHONE:
                mic = d
                break

    return AudioSetup(
        microphone=mic,
        all_devices=devices,
        headphones_connected=headphones,
    )
