"""Audio device detection and management for macOS via AVFoundation/ffmpeg."""

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
    VIRTUAL = "virtual"


@dataclass
class AudioDevice:
    index: int
    name: str
    device_type: DeviceType
    is_input: bool = True
    is_output: bool = False

    @property
    def is_blackhole(self) -> bool:
        return "blackhole" in self.name.lower()

    @property
    def is_builtin_mic(self) -> bool:
        return "macbook" in self.name.lower() and "microphone" in self.name.lower()

    @property
    def is_builtin_speaker(self) -> bool:
        return "macbook" in self.name.lower() and "speaker" in self.name.lower()


@dataclass
class AudioSetup:
    """Detected audio configuration."""
    microphone: AudioDevice | None = None
    system_capture: AudioDevice | None = None
    all_devices: list[AudioDevice] = field(default_factory=list)
    headphones_connected: bool = False
    blackhole_available: bool = False

    @property
    def can_record_mic(self) -> bool:
        return self.microphone is not None

    @property
    def can_record_system(self) -> bool:
        return self.system_capture is not None and self.blackhole_available

    @property
    def can_record_both(self) -> bool:
        return self.can_record_mic and self.can_record_system


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
        is_input = device_type in (DeviceType.MICROPHONE, DeviceType.VIRTUAL)

        devices.append(AudioDevice(
            index=index,
            name=name,
            device_type=device_type,
            is_input=is_input,
            is_output=device_type == DeviceType.VIRTUAL,
        ))

    return devices


def _classify_device(name: str) -> DeviceType:
    lower = name.lower()
    if "blackhole" in lower or "zoom" in lower or "soundflower" in lower:
        return DeviceType.VIRTUAL
    if "microphone" in lower or "mic" in lower:
        return DeviceType.MICROPHONE
    return DeviceType.SPEAKER


def detect_headphones() -> bool:
    """Check if headphones/external audio output is connected."""
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

    # Fallback: check if default output is not built-in speakers
    try:
        result = subprocess.run(
            ["system_profiler", "SPAudioDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = result.stdout.splitlines()
        for i, line in enumerate(lines):
            if "Default Output Device: Yes" in line:
                # Look backwards for device name
                for j in range(i - 1, max(i - 10, -1), -1):
                    if lines[j].strip().endswith(":") and not lines[j].strip().startswith(("Default", "Output", "Input", "Manufacturer")):
                        device_name = lines[j].strip().rstrip(":")
                        if "macbook" not in device_name.lower() or "speaker" not in device_name.lower():
                            return True
                        return False
    except Exception:
        pass

    return False


def _find_device_by_name(devices: list[AudioDevice], name_substring: str) -> AudioDevice | None:
    """Find a device whose name contains the given substring (case-insensitive)."""
    if not name_substring:
        return None
    lower = name_substring.lower()
    for d in devices:
        if lower in d.name.lower():
            return d
    return None


def detect_audio_setup(profile: AudioProfile | None = None) -> AudioSetup:
    """Detect full audio setup, optionally guided by a profile.

    If profile is given, mic selection follows profile.preferred_mic.
    If profile.preferred_mic is empty (e.g. "headphones" profile), we pick
    any non-builtin microphone first, falling back to builtin.
    """
    devices = detect_ffmpeg_devices()
    headphones = detect_headphones()
    blackhole = any(d.is_blackhole for d in devices)

    # --- Microphone selection ---
    mic = None
    if profile and profile.preferred_mic:
        mic = _find_device_by_name(devices, profile.preferred_mic)

    if mic is None and profile and not profile.preferred_mic:
        # Empty preferred_mic means "use headphone/external mic if available"
        for d in devices:
            if d.device_type == DeviceType.MICROPHONE and not d.is_builtin_mic:
                mic = d
                break

    if mic is None:
        # Fallback: built-in Mac mic
        for d in devices:
            if d.is_builtin_mic:
                mic = d
                break

    if mic is None:
        # Last resort: any mic
        for d in devices:
            if d.device_type == DeviceType.MICROPHONE:
                mic = d
                break

    # --- System capture: BlackHole ---
    system_capture = None
    capture_name = profile.preferred_system_capture if profile else "BlackHole"
    system_capture = _find_device_by_name(devices, capture_name)

    return AudioSetup(
        microphone=mic,
        system_capture=system_capture,
        all_devices=devices,
        headphones_connected=headphones,
        blackhole_available=blackhole,
    )
