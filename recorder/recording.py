"""Recording engine — manages ffmpeg processes for audio capture."""

from __future__ import annotations

import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from .audio import AudioDevice, AudioSetup
from .config import RecordingConfig


class RecordingState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class Track:
    name: str
    device: AudioDevice
    output_path: Path
    process: subprocess.Popen | None = None


@dataclass
class RecordingSession:
    session_id: str
    started_at: datetime
    tracks: list[Track] = field(default_factory=list)
    state: RecordingState = RecordingState.IDLE
    mixed_path: Path | None = None

    @property
    def duration_seconds(self) -> float:
        if self.state == RecordingState.IDLE:
            return 0.0
        return (datetime.now() - self.started_at).total_seconds()


class RecordingEngine:
    def __init__(self, config: RecordingConfig, audio_setup: AudioSetup):
        self.config = config
        self.audio = audio_setup
        self.session: RecordingSession | None = None

    def start(self, label: str | None = None) -> RecordingSession:
        if self.session and self.session.state == RecordingState.RECORDING:
            raise RuntimeError("Recording already in progress")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{label or 'call'}_{timestamp}"
        output_dir = self.config.output_dir / session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        self.session = RecordingSession(
            session_id=session_id,
            started_at=datetime.now(),
        )

        tracks_to_record = self._plan_tracks(output_dir)

        for track in tracks_to_record:
            proc = self._start_ffmpeg(track.device, track.output_path)
            track.process = proc
            self.session.tracks.append(track)

        self.session.state = RecordingState.RECORDING
        return self.session

    def stop(self) -> RecordingSession:
        if not self.session or self.session.state != RecordingState.RECORDING:
            raise RuntimeError("No active recording to stop")

        for track in self.session.tracks:
            if track.process and track.process.poll() is None:
                # Send 'q' to ffmpeg stdin for graceful stop
                try:
                    track.process.communicate(input=b"q", timeout=5)
                except subprocess.TimeoutExpired:
                    track.process.send_signal(signal.SIGINT)
                    try:
                        track.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        track.process.kill()

        self.session.state = RecordingState.STOPPED

        if self.config.mix_tracks and len(self.session.tracks) > 1:
            self._mix_tracks()

        self._run_post_hooks()

        return self.session

    def status(self) -> dict:
        if not self.session:
            return {"state": "idle"}

        alive_tracks = []
        for t in self.session.tracks:
            alive = t.process is not None and t.process.poll() is None
            alive_tracks.append({
                "name": t.name,
                "device": t.device.name,
                "file": str(t.output_path),
                "alive": alive,
            })

        return {
            "state": self.session.state.value,
            "session_id": self.session.session_id,
            "duration_seconds": round(self.session.duration_seconds, 1),
            "tracks": alive_tracks,
            "mixed_file": str(self.session.mixed_path) if self.session.mixed_path else None,
        }

    def _plan_tracks(self, output_dir: Path) -> list[Track]:
        tracks = []
        ext = self.config.format

        if self.audio.can_record_mic and self.audio.microphone:
            tracks.append(Track(
                name="mic",
                device=self.audio.microphone,
                output_path=output_dir / f"mic.{ext}",
            ))

        if self.audio.can_record_system and self.audio.system_capture:
            tracks.append(Track(
                name="system",
                device=self.audio.system_capture,
                output_path=output_dir / f"system.{ext}",
            ))

        if not tracks:
            # Fallback: record whatever mic is available
            if self.audio.microphone:
                tracks.append(Track(
                    name="mic",
                    device=self.audio.microphone,
                    output_path=output_dir / f"mic.{ext}",
                ))
            else:
                raise RuntimeError(
                    "No audio devices available for recording. "
                    "Check your microphone and BlackHole setup."
                )

        return tracks

    def _start_ffmpeg(self, device: AudioDevice, output_path: Path) -> subprocess.Popen:
        cmd = [
            "ffmpeg",
            "-f", "avfoundation",
            "-i", f":{device.index}",
            "-ar", str(self.config.sample_rate),
            "-ac", str(self.config.channels),
            "-c:a", self.config.codec,
            "-b:a", self.config.bitrate,
            "-y",
            str(output_path),
        ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Brief check that ffmpeg started OK
        time.sleep(0.3)
        if proc.poll() is not None:
            _, stderr = proc.communicate()
            raise RuntimeError(
                f"ffmpeg failed to start recording from {device.name}: "
                f"{stderr.decode(errors='replace')[-500:]}"
            )

        return proc

    def _mix_tracks(self) -> None:
        if not self.session or len(self.session.tracks) < 2:
            return

        inputs = []
        for track in self.session.tracks:
            if track.output_path.exists() and track.output_path.stat().st_size > 0:
                inputs.extend(["-i", str(track.output_path)])

        if len(inputs) < 4:  # Need at least 2 inputs (2 args each)
            return

        n_inputs = len(inputs) // 2
        mix_path = self.session.tracks[0].output_path.parent / f"mixed.{self.config.format}"

        cmd = [
            "ffmpeg",
            *inputs,
            "-filter_complex", f"amix=inputs={n_inputs}:duration=longest",
            "-c:a", self.config.codec,
            "-b:a", self.config.bitrate,
            "-y",
            str(mix_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, timeout=120, check=True)
            self.session.mixed_path = mix_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            # Non-fatal: we still have individual tracks
            print(f"Warning: mixing failed: {e}")

    def _run_post_hooks(self) -> None:
        if not self.session or not self.config.post_hooks:
            return

        session_dir = str(self.session.tracks[0].output_path.parent)
        for hook in self.config.post_hooks:
            cmd = hook.replace("{session_dir}", session_dir)
            cmd = cmd.replace("{session_id}", self.session.session_id)
            try:
                subprocess.run(cmd, shell=True, timeout=300)
            except Exception as e:
                print(f"Warning: post-hook failed: {hook}: {e}")
