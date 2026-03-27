"""Recording engine — manages ffmpeg (mic) and ScreenCaptureKit (system audio) processes.

Pause/resume uses segmented recording: each segment is a properly finalized file.
If the computer crashes, all completed segments are safe.
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

# Compiled Swift binary for system audio capture via ScreenCaptureKit
SYSTEM_CAPTURE_BINARY = Path(__file__).parent.parent / "scripts" / "capture_system_audio"


class RecordingState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


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
    mixed_path: Path | None = None
    segment_index: int = 0
    # Each entry: list of segment file paths per track name
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
            raise RuntimeError(
                "No audio sources available for recording. "
                "Check your microphone permissions."
            )

        self.session.state = RecordingState.RECORDING
        return self.session

    def pause(self) -> None:
        if not self.session or self.session.state != RecordingState.RECORDING:
            return

        # Gracefully stop all processes — finalizes current segment files
        self._stop_processes()
        self.session._pause_started = datetime.now()
        self.session.state = RecordingState.PAUSED

    def resume(self) -> None:
        if not self.session or self.session.state != RecordingState.PAUSED:
            return

        if self.session._pause_started:
            self.session._paused_total += (datetime.now() - self.session._pause_started).total_seconds()
            self.session._pause_started = None

        # Start new segment
        self._start_segment()
        self.session.state = RecordingState.RECORDING

    def stop(self) -> RecordingSession:
        if not self.session or self.session.state not in (RecordingState.RECORDING, RecordingState.PAUSED):
            raise RuntimeError("No active recording to stop")

        if self.session.state == RecordingState.PAUSED:
            # Already stopped processes on pause, just update pause time
            if self.session._pause_started:
                self.session._paused_total += (datetime.now() - self.session._pause_started).total_seconds()
                self.session._pause_started = None
        else:
            self._stop_processes()

        self.session.state = RecordingState.STOPPED

        # Concatenate segments into final files
        self._concat_segments()

        # Echo cancellation: clean mic track when using speakers
        if self.audio.headphones_connected is False and len(self.session.tracks) > 1:
            self._echo_cancel()

        self._write_manifest()

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
                "device": t.device_name,
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

    # --- Segment management ---

    def _start_segment(self) -> None:
        """Start a new recording segment for all tracks."""
        seg = self.session.segment_index
        output_dir = self.session.output_dir
        self.session.tracks.clear()

        # Mic
        if self.audio.can_record_mic and self.audio.microphone:
            seg_path = output_dir / f"mic_seg{seg:03d}.{self.config.format}"
            proc = self._start_ffmpeg(self.audio.microphone, seg_path)
            self.session.tracks.append(Track(
                name="mic",
                output_path=seg_path,
                process=proc,
                device_name=self.audio.microphone.name,
            ))
            self.session.segments.setdefault("mic", []).append(seg_path)

        # System audio
        seg_path = output_dir / f"system_seg{seg:03d}.wav"
        proc = self._start_system_capture(seg_path)
        if proc:
            self.session.tracks.append(Track(
                name="system",
                output_path=seg_path,
                process=proc,
                device_name="ScreenCaptureKit",
            ))
            self.session.segments.setdefault("system", []).append(seg_path)

        self.session.segment_index += 1

    def _stop_processes(self) -> None:
        """Gracefully stop all running processes, finalizing their files."""
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
        """Concatenate segments into final track files."""
        output_dir = self.session.output_dir

        for track_name, seg_paths in self.session.segments.items():
            # Filter to existing, non-empty files
            valid = [p for p in seg_paths if p.exists() and p.stat().st_size > 0]

            if not valid:
                continue

            ext = self.config.format if track_name == "mic" else "wav"
            final_path = output_dir / f"{track_name}.{ext}"

            if len(valid) == 1:
                # Single segment — just rename
                valid[0].rename(final_path)
            else:
                # Multiple segments — concatenate with ffmpeg
                concat_list = output_dir / f"_{track_name}_concat.txt"
                concat_list.write_text(
                    "\n".join(f"file '{p.name}'" for p in valid)
                )
                cmd = [
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_list),
                    "-c", "copy",
                    "-y",
                    str(final_path),
                ]
                try:
                    subprocess.run(cmd, capture_output=True, timeout=120, check=True)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    print(f"Warning: concat failed for {track_name}: {e}")
                    # Keep segments as-is
                    continue
                finally:
                    concat_list.unlink(missing_ok=True)

                # Clean up segments
                for p in valid:
                    p.unlink(missing_ok=True)

            # Update track to point to final file
            for t in self.session.tracks:
                if t.name == track_name:
                    t.output_path = final_path

    # --- Manifest ---

    def _write_manifest(self) -> None:
        """Write session metadata for downstream processing (Whisper, diarization)."""
        if not self.session or not self.session.tracks:
            return

        session_dir = self.session.output_dir
        manifest = {
            "session_id": self.session.session_id,
            "started_at": self.session.started_at.isoformat(),
            "duration_seconds": round(self.session.duration_seconds, 1),
            "headphones_connected": self.audio.headphones_connected,
            "segments_count": self.session.segment_index,
            "tracks": {},
        }

        for track in self.session.tracks:
            size = track.output_path.stat().st_size if track.output_path.exists() else 0
            manifest["tracks"][track.name] = {
                "file": track.output_path.name,
                "device": track.device_name,
                "role": "user" if track.name == "mic" else "remote",
                "size_bytes": size,
            }

        mic_clean = session_dir / "mic_clean.wav"
        if mic_clean.exists():
            manifest["tracks"]["mic_clean"] = {
                "file": "mic_clean.wav",
                "device": "echo-cancelled",
                "role": "user",
                "size_bytes": mic_clean.stat().st_size,
            }

        if self.session.mixed_path and self.session.mixed_path.exists():
            manifest["mixed"] = {
                "file": self.session.mixed_path.name,
                "size_bytes": self.session.mixed_path.stat().st_size,
            }

        manifest_path = session_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # --- Echo cancellation ---

    def _echo_cancel(self) -> None:
        """Remove speaker bleed from mic track using system track as reference."""
        mic_track = next((t for t in self.session.tracks if t.name == "mic"), None)
        sys_track = next((t for t in self.session.tracks if t.name == "system"), None)

        if not mic_track or not sys_track:
            return
        if not mic_track.output_path.exists() or not sys_track.output_path.exists():
            return

        clean_path = mic_track.output_path.parent / "mic_clean.wav"

        cmd = [
            "ffmpeg",
            "-i", str(mic_track.output_path),
            "-i", str(sys_track.output_path),
            "-filter_complex",
            "[1:a]adelay=0|0,volume=0.8[ref];"
            "[0:a][ref]amix=inputs=2:duration=first:weights=1 -1[out]",
            "-map", "[out]",
            "-y",
            str(clean_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, timeout=120, check=True)
            print(f"Echo cancellation: {clean_path}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"Warning: echo cancellation failed (non-fatal): {e}")

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
            "-y",
            str(output_path),
        ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        time.sleep(0.3)
        if proc.poll() is not None:
            _, stderr = proc.communicate()
            raise RuntimeError(
                f"ffmpeg failed to start recording from {device.name}: "
                f"{stderr.decode(errors='replace')[-500:]}"
            )

        return proc

    def _start_system_capture(self, output_path: Path) -> subprocess.Popen | None:
        binary = SYSTEM_CAPTURE_BINARY
        if not binary.exists():
            print(f"Warning: system audio capture binary not found at {binary}")
            print("System audio will not be recorded.")
            return None

        cmd = [str(binary), str(output_path)]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        time.sleep(0.5)
        if proc.poll() is not None:
            _, stderr = proc.communicate()
            print(
                f"Warning: system audio capture failed to start: "
                f"{stderr.decode(errors='replace')[-500:]}"
            )
            return None

        return proc

    # --- Post-processing ---

    def _mix_tracks(self) -> None:
        if not self.session or len(self.session.tracks) < 2:
            return

        inputs = []
        for track in self.session.tracks:
            if track.output_path.exists() and track.output_path.stat().st_size > 0:
                inputs.extend(["-i", str(track.output_path)])

        if len(inputs) < 4:
            return

        n_inputs = len(inputs) // 2
        mix_path = self.session.output_dir / f"mixed.{self.config.format}"

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
            print(f"Warning: mixing failed: {e}")

    def _run_post_hooks(self) -> None:
        if not self.session or not self.config.post_hooks:
            return

        session_dir = str(self.session.output_dir)
        for hook in self.config.post_hooks:
            cmd = hook.replace("{session_dir}", session_dir)
            cmd = cmd.replace("{session_id}", self.session.session_id)
            try:
                subprocess.run(cmd, shell=True, timeout=300)
            except Exception as e:
                print(f"Warning: post-hook failed: {hook}: {e}")
