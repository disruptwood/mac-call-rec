"""Recording engine — manages ffmpeg (mic) and ScreenCaptureKit (system audio).

Records mic and system as separate tracks. On stop:
1. Normalizes mic + mixes into recording.m4a
2. Keeps source tracks (_mic, _system) for VAD-based speaker diarization
3. Writes manifest.json

Transcription is done separately via scripts/transcribe.py (Qwen3-ASR + Silero VAD).
"""

from __future__ import annotations

import json
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from .audio import AudioDevice, AudioSetup
from .config import RecordingConfig

SYSTEM_CAPTURE_BINARY = Path(__file__).parent.parent / "scripts" / "capture_system_audio"


class RecordingState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class Track:
    name: str
    output_path: Path
    process: subprocess.Popen | None = None
    device_name: str = ""


@dataclass
class RecordingSession:
    session_id: str
    started_at: datetime
    output_dir: Path = field(default_factory=lambda: Path("."))
    tracks: list[Track] = field(default_factory=list)
    state: RecordingState = RecordingState.IDLE
    recording_path: Path | None = None
    segment_index: int = 0
    segments: dict[str, list[Path]] = field(default_factory=dict)
    _paused_total: float = 0.0
    _pause_started: datetime | None = None

    @property
    def duration_seconds(self) -> float:
        if self.state == RecordingState.IDLE:
            return 0.0
        elapsed = (datetime.now() - self.started_at).total_seconds()
        paused = self._paused_total
        if self._pause_started:
            paused += (datetime.now() - self._pause_started).total_seconds()
        return elapsed - paused


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
            output_dir=output_dir,
        )

        self._start_segment()

        if not self.session.tracks:
            raise RuntimeError("No audio sources available. Check microphone permissions.")

        self.session.state = RecordingState.RECORDING
        return self.session

    def pause(self) -> None:
        if not self.session or self.session.state != RecordingState.RECORDING:
            return
        self._stop_processes()
        self.session._pause_started = datetime.now()
        self.session.state = RecordingState.PAUSED

    def resume(self) -> None:
        if not self.session or self.session.state != RecordingState.PAUSED:
            return
        if self.session._pause_started:
            self.session._paused_total += (datetime.now() - self.session._pause_started).total_seconds()
            self.session._pause_started = None
        self._start_segment()
        self.session.state = RecordingState.RECORDING

    def stop(self) -> RecordingSession:
        if not self.session or self.session.state not in (RecordingState.RECORDING, RecordingState.PAUSED):
            raise RuntimeError("No active recording to stop")

        if self.session.state == RecordingState.PAUSED:
            if self.session._pause_started:
                self.session._paused_total += (datetime.now() - self.session._pause_started).total_seconds()
                self.session._pause_started = None
        else:
            self._stop_processes()

        self.session.state = RecordingState.STOPPED

        self._concat_segments()
        self._normalize_and_mix()
        self._write_manifest()

        return self.session

    def status(self) -> dict:
        if not self.session:
            return {"state": "idle"}

        alive_tracks = []
        for t in self.session.tracks:
            alive = t.process is not None and t.process.poll() is None
            alive_tracks.append({
                "name": t.name,
                "device": t.device_name,
                "file": str(t.output_path),
                "alive": alive,
            })

        return {
            "state": self.session.state.value,
            "session_id": self.session.session_id,
            "duration_seconds": round(self.session.duration_seconds, 1),
            "tracks": alive_tracks,
            "recording": str(self.session.recording_path) if self.session.recording_path else None,
        }

    # --- Segment management (for pause/resume) ---

    def _start_segment(self) -> None:
        seg = self.session.segment_index
        output_dir = self.session.output_dir
        self.session.tracks.clear()

        if self.audio.can_record_mic and self.audio.microphone:
            seg_path = output_dir / f"_mic_seg{seg:03d}.{self.config.format}"
            proc = self._start_ffmpeg(self.audio.microphone, seg_path)
            self.session.tracks.append(Track(
                name="mic", output_path=seg_path, process=proc,
                device_name=self.audio.microphone.name,
            ))
            self.session.segments.setdefault("mic", []).append(seg_path)

        seg_path = output_dir / f"_system_seg{seg:03d}.wav"
        proc = self._start_system_capture(seg_path)
        if proc:
            self.session.tracks.append(Track(
                name="system", output_path=seg_path, process=proc,
                device_name="ScreenCaptureKit",
            ))
            self.session.segments.setdefault("system", []).append(seg_path)

        self.session.segment_index += 1

    def _stop_processes(self) -> None:
        for track in self.session.tracks:
            if track.process and track.process.poll() is None:
                if track.name == "mic":
                    try:
                        track.process.communicate(input=b"q", timeout=5)
                    except subprocess.TimeoutExpired:
                        track.process.send_signal(signal.SIGINT)
                        try:
                            track.process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            track.process.kill()
                else:
                    track.process.send_signal(signal.SIGTERM)
                    try:
                        track.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        track.process.kill()

    def _concat_segments(self) -> None:
        """Concatenate pause/resume segments into single track files."""
        output_dir = self.session.output_dir

        for track_name, seg_paths in self.session.segments.items():
            valid = [p for p in seg_paths if p.exists() and p.stat().st_size > 0]
            if not valid:
                continue

            ext = self.config.format if track_name == "mic" else "wav"
            final_path = output_dir / f"_{track_name}.{ext}"

            if len(valid) == 1:
                valid[0].rename(final_path)
            else:
                concat_list = output_dir / f"_{track_name}_concat.txt"
                concat_list.write_text("\n".join(f"file '{p.name}'" for p in valid))
                cmd = [
                    "ffmpeg", "-f", "concat", "-safe", "0",
                    "-i", str(concat_list), "-c", "copy", "-y", str(final_path),
                ]
                try:
                    subprocess.run(cmd, capture_output=True, timeout=120, check=True)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    if valid:
                        valid[0].rename(final_path)
                finally:
                    concat_list.unlink(missing_ok=True)
                for p in valid:
                    p.unlink(missing_ok=True)

            for t in self.session.tracks:
                if t.name == track_name:
                    t.output_path = final_path

    def _normalize_and_mix(self) -> None:
        """Normalize mic volume + mix with system → recording.m4a. Keep source tracks."""
        output_dir = self.session.output_dir
        mic_track = next((t for t in self.session.tracks if t.name == "mic"), None)
        sys_track = next((t for t in self.session.tracks if t.name == "system"), None)

        recording_path = output_dir / f"recording.{self.config.format}"

        if mic_track and sys_track and mic_track.output_path.exists() and sys_track.output_path.exists():
            cmd = [
                "ffmpeg",
                "-i", str(mic_track.output_path),
                "-i", str(sys_track.output_path),
                "-filter_complex",
                "[0:a]loudnorm=I=-16:TP=-1.5:LRA=11[mic];"
                "[mic][1:a]amix=inputs=2:duration=longest[out]",
                "-map", "[out]",
                "-c:a", self.config.codec, "-b:a", self.config.bitrate,
                "-y", str(recording_path),
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=300, check=True)
                self.session.recording_path = recording_path
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"Warning: normalize+mix failed: {e}")

        elif sys_track and sys_track.output_path.exists():
            cmd = [
                "ffmpeg", "-i", str(sys_track.output_path),
                "-c:a", self.config.codec, "-b:a", self.config.bitrate,
                "-y", str(recording_path),
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=300, check=True)
                self.session.recording_path = recording_path
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass

        elif mic_track and mic_track.output_path.exists():
            mic_track.output_path.rename(recording_path)
            self.session.recording_path = recording_path

    def _write_manifest(self) -> None:
        if not self.session:
            return

        session_dir = self.session.output_dir
        manifest = {
            "session_id": self.session.session_id,
            "started_at": self.session.started_at.isoformat(),
            "duration_seconds": round(self.session.duration_seconds, 1),
            "headphones_connected": self.audio.headphones_connected,
        }

        if self.session.recording_path and self.session.recording_path.exists():
            manifest["recording"] = {
                "file": self.session.recording_path.name,
                "size_bytes": self.session.recording_path.stat().st_size,
            }

        manifest_path = session_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # --- Process launchers ---

    def _start_ffmpeg(self, device: AudioDevice, output_path: Path) -> subprocess.Popen:
        cmd = [
            "ffmpeg",
            "-f", "avfoundation",
            "-i", f":{device.index}",
            "-ar", str(self.config.sample_rate),
            "-ac", str(self.config.channels),
            "-c:a", self.config.codec,
            "-b:a", self.config.bitrate,
            "-y", str(output_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(0.3)
        if proc.poll() is not None:
            _, stderr = proc.communicate()
            raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')[-500:]}")
        return proc

    def _start_system_capture(self, output_path: Path) -> subprocess.Popen | None:
        binary = SYSTEM_CAPTURE_BINARY
        if not binary.exists():
            print(f"Warning: system audio capture binary not found at {binary}")
            return None
        cmd = [str(binary), str(output_path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(0.5)
        if proc.poll() is not None:
            _, stderr = proc.communicate()
            print(f"Warning: system audio capture failed: {stderr.decode(errors='replace')[-500:]}")
            return None
        return proc
