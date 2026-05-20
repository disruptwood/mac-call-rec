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
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from .audio import AudioDevice, AudioSetup
from .config import RecordingConfig

SYSTEM_CAPTURE_BINARY = Path(__file__).parent.parent / "scripts" / "capture_system_audio"
MIC_PA_SCRIPT = Path(__file__).parent.parent / "scripts" / "capture_mic_pa.py"


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

        if not self.audio.headphones_connected:
            print("WARN: no headphones detected — mic will pick up bleed from speakers. "
                  "For clean speaker separation, plug in headphones.")

        self._save_and_boost_mic_volume()

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

        self._restore_mic_volume()

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
            # Mic via PortAudio (CoreAudio HAL, bypasses AVFoundation/VPIO).
            # Verified 2026-05-20: PortAudio is cleaner audio and tighter sync
            # with the system track than the previous ffmpeg avfoundation path,
            # which suffered up to 40% drift on long sessions under VPIO.
            # No fallback: if PortAudio can't start (missing dep, permission
            # denied, device busy), we record without mic rather than fall back
            # to a known-broken path.
            pa_seg_path = output_dir / f"_mic_pa_seg{seg:03d}.wav"
            pa_proc = self._start_mic_pa_capture(
                pa_seg_path, device_name=self.audio.microphone.name,
            )
            if pa_proc:
                self.session.tracks.append(Track(
                    name="mic_pa", output_path=pa_seg_path, process=pa_proc,
                    device_name=f"PortAudio: {self.audio.microphone.name}",
                ))
                self.session.segments.setdefault("mic_pa", []).append(pa_seg_path)
            else:
                print("WARN: PortAudio mic capture failed to start — session "
                      "will record system audio only")

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
        """Stop every active capture process. Must NOT raise — a failure on one
        track (e.g. mic ffmpeg died and its stdin is closed) cannot prevent
        the others (mic_pa, system) from being terminated cleanly."""
        for track in self.session.tracks:
            if not track.process or track.process.poll() is not None:
                continue
            try:
                if track.name == "mic":
                    try:
                        track.process.communicate(input=b"q", timeout=5)
                    except (subprocess.TimeoutExpired, BrokenPipeError, OSError):
                        try:
                            track.process.send_signal(signal.SIGINT)
                            track.process.wait(timeout=5)
                        except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                            try:
                                track.process.kill()
                            except (ProcessLookupError, OSError):
                                pass
                else:
                    try:
                        track.process.send_signal(signal.SIGTERM)
                        track.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            track.process.kill()
                        except (ProcessLookupError, OSError):
                            pass
                    except (ProcessLookupError, OSError):
                        pass
            except Exception as e:
                print(f"Warning: error stopping track {track.name}: {e}")

    def _concat_segments(self) -> None:
        """Concatenate pause/resume segments into single track files."""
        output_dir = self.session.output_dir

        for track_name, seg_paths in self.session.segments.items():
            valid = [p for p in seg_paths if p.exists() and p.stat().st_size > 0]
            if not valid:
                continue

            ext = "wav"  # Both mic and system are WAV
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
        """Produce recording.m4a from available source tracks.

        Cases:
          - mic + system → loudnorm mic, then amix with system
          - system only → transcode to m4a (no mix needed)
          - mic only → rename WAV directly to recording.m4a (cheapest path)

        Mic is always PortAudio (`mic_pa` track) since 2026-05-20 — the
        ffmpeg avfoundation path was dropped because it drifted up to 40%
        under VPIO on long sessions.
        """
        recording_path = self.session.output_dir / f"recording.{self.config.format}"
        mic = next((t for t in self.session.tracks
                    if t.name == "mic_pa" and t.output_path.exists()), None)
        sys_t = next((t for t in self.session.tracks
                      if t.name == "system" and t.output_path.exists()), None)

        cmd = self._build_normalize_mix_cmd(mic, sys_t, recording_path)
        if cmd is None:
            if mic:
                mic.output_path.rename(recording_path)
                self.session.recording_path = recording_path
            return

        try:
            subprocess.run(cmd, capture_output=True, timeout=300, check=True)
            self.session.recording_path = recording_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"Warning: normalize+mix failed: {e}")

    def _build_normalize_mix_cmd(
        self, mic: Track | None, sys_t: Track | None, output_path: Path,
    ) -> list[str] | None:
        """Build ffmpeg command for mic+system mix or system-only transcode.

        Returns None when there is no ffmpeg work to do (mic-only is handled by
        the caller via rename, no transcode needed)."""
        if mic and sys_t:
            return [
                "ffmpeg",
                "-i", str(mic.output_path),
                "-i", str(sys_t.output_path),
                "-filter_complex",
                "[0:a]loudnorm=I=-16:TP=-1.5:LRA=11[m];"
                "[m][1:a]amix=inputs=2:duration=longest[out]",
                "-map", "[out]",
                "-ar", str(self.config.sample_rate),
                "-ac", str(self.config.channels),
                "-c:a", self.config.codec, "-b:a", self.config.bitrate,
                "-y", str(output_path),
            ]
        if sys_t:
            return [
                "ffmpeg",
                "-i", str(sys_t.output_path),
                "-ar", str(self.config.sample_rate),
                "-ac", str(self.config.channels),
                "-c:a", self.config.codec, "-b:a", self.config.bitrate,
                "-y", str(output_path),
            ]
        return None

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

    def _save_and_boost_mic_volume(self) -> None:
        """Save current mic input volume and boost to 85% for recording.

        macOS default mic input is often ~50% which records -42 dB. We boost
        to 85% on start and restore on stop (see _restore_mic_volume).
        Mocked in tests to avoid changing real system volume.
        """
        try:
            r = subprocess.run(
                ["osascript", "-e", "input volume of (get volume settings)"],
                capture_output=True, text=True, timeout=5,
            )
            self._original_input_volume = r.stdout.strip()
            subprocess.run(
                ["osascript", "-e", "set volume input volume 85"],
                capture_output=True, timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._original_input_volume = ""

    def _restore_mic_volume(self) -> None:
        """Restore mic input volume to the pre-recording level."""
        original = getattr(self, "_original_input_volume", "")
        if not original:
            return
        try:
            subprocess.run(
                ["osascript", "-e", f"set volume input volume {original}"],
                capture_output=True, timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    def _start_system_capture(self, output_path: Path) -> subprocess.Popen | None:
        """Start ScreenCaptureKit system audio capture, log stderr to file."""
        binary = SYSTEM_CAPTURE_BINARY
        if not binary.exists():
            print(f"Warning: system audio capture binary not found at {binary}")
            return None
        cmd = [str(binary), str(output_path)]
        log_path = output_path.with_suffix(output_path.suffix + ".capture.log")
        log_file = open(log_path, "wb")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log_file, stderr=log_file)
        time.sleep(0.5)
        if proc.poll() is not None:
            log_file.close()
            tail = log_path.read_bytes()[-500:] if log_path.exists() else b""
            print(f"Warning: system audio capture failed: {tail.decode(errors='replace')}")
            return None
        return proc

    def _start_mic_pa_capture(
        self, output_path: Path, device_name: str | None = None,
    ) -> subprocess.Popen | None:
        """Start PortAudio mic capture (parallel A/B against ffmpeg avfoundation).

        Spawns scripts/capture_mic_pa.py using the same interpreter that runs the
        recorder (sys.executable). Logs stderr (PortAudio + soundfile diagnostics)
        to a .pa.log file. Soft-fail: returns None if the script is missing or the
        process dies in the first ~300 ms (e.g. sounddevice not installed in this
        env). The ffmpeg mic track continues unaffected.

        device_name is matched by sounddevice as a substring against all input
        devices — passing the same name that AVFoundation picked ensures both
        tracks read from the same physical device.
        """
        if not MIC_PA_SCRIPT.exists():
            return None
        cmd = [sys.executable, str(MIC_PA_SCRIPT), str(output_path)]
        if device_name:
            cmd.extend(["--device", device_name])
        log_path = output_path.with_suffix(output_path.suffix + ".pa.log")
        log_file = open(log_path, "wb")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log_file, stderr=log_file)
        # PortAudio's first stream open on macOS can take up to ~1.5s when the
        # TCC mic-permission dialog appears or when waking a Bluetooth device.
        # Sleep long enough that poll() reliably catches early-death failures.
        time.sleep(2.0)
        if proc.poll() is not None:
            log_file.close()
            tail = log_path.read_bytes()[-500:] if log_path.exists() else b""
            print(f"Warning: PortAudio mic capture failed (ffmpeg mic continues): "
                  f"{tail.decode(errors='replace')}")
            return None
        return proc
