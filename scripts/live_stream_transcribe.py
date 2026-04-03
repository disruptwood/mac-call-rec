#!/usr/bin/env python3
"""Live streaming transcription using Qwen3-ASR + ScreenCaptureKit.

Captures system audio directly via the compiled Swift binary,
feeds chunks to Qwen3-ASR, prints transcription in real-time.
Runs independently from call-recorder — just start alongside it.

Usage: python3 live_stream_transcribe.py [--language Russian]
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CAPTURE_BINARY = Path(__file__).parent / "capture_system_audio"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="Russian")
    parser.add_argument("--model", default="/Users/ilya/models/qwen3-asr-1.7b")
    parser.add_argument("--chunk-seconds", type=int, default=60)
    parser.add_argument("--output", default=None, help="Save transcript to file")
    args = parser.parse_args()

    if not CAPTURE_BINARY.exists():
        print(f"ERROR: {CAPTURE_BINARY} not found")
        sys.exit(1)

    # Load HF token for pyannote diarization
    if not os.environ.get("HF_TOKEN"):
        token_file = Path.home() / ".huggingface_token"
        if token_file.exists():
            os.environ["HF_TOKEN"] = token_file.read_text().strip()

    print(f"Loading Qwen3-ASR ({args.model})...", flush=True)
    from mlx_qwen3_asr import Session
    session = Session(model=args.model)
    print("Model loaded. Listening...\n", flush=True)

    tmpdir = Path(tempfile.mkdtemp(prefix="live_transcribe_"))
    chunk_idx = 0
    all_text = []

    # Find mic device index
    mic_idx = None
    try:
        r = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stderr.splitlines():
            if "MacBook Air Microphone" in line:
                import re
                m = re.search(r"\[(\d+)\]", line)
                if m:
                    mic_idx = m.group(1)
    except Exception:
        pass

    if mic_idx is None:
        print("WARNING: MacBook mic not found, recording system audio only", flush=True)

    try:
        while True:
            # Record system audio + mic simultaneously, mix into one file
            sys_path = tmpdir / f"sys_{chunk_idx:04d}.wav"
            chunk_path = tmpdir / f"chunk_{chunk_idx:04d}.wav"

            # Start system audio capture
            sys_proc = subprocess.Popen(
                [str(CAPTURE_BINARY), str(sys_path), str(args.chunk_seconds)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )

            # Start mic capture in parallel
            mic_path = tmpdir / f"mic_{chunk_idx:04d}.wav"
            mic_proc = None
            if mic_idx is not None:
                mic_proc = subprocess.Popen(
                    ["ffmpeg", "-f", "avfoundation", "-i", f":{mic_idx}",
                     "-t", str(args.chunk_seconds), "-ar", "16000", "-ac", "1",
                     "-y", str(mic_path)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )

            # Wait for both
            try:
                sys_proc.wait(timeout=args.chunk_seconds + 15)
            except subprocess.TimeoutExpired:
                sys_proc.kill()
                sys_proc.wait()

            if mic_proc:
                try:
                    mic_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    mic_proc.kill()
                    mic_proc.wait()

            # Mix mic + system into one chunk
            if mic_idx and mic_path.exists() and sys_path.exists() and sys_path.stat().st_size > 0:
                subprocess.run(
                    ["ffmpeg", "-i", str(mic_path), "-i", str(sys_path),
                     "-filter_complex",
                     "[0:a]loudnorm=I=-16:TP=-1.5:LRA=11[mic];[mic][1:a]amix=inputs=2:duration=longest[out]",
                     "-map", "[out]", "-ar", "16000", "-ac", "1", "-y", str(chunk_path)],
                    capture_output=True, timeout=30,
                )
                mic_path.unlink(missing_ok=True)
                sys_path.unlink(missing_ok=True)
            elif sys_path.exists() and sys_path.stat().st_size > 0:
                sys_path.rename(chunk_path)

            if not chunk_path.exists() or chunk_path.stat().st_size < 1000:
                chunk_idx += 1
                continue

            # Transcribe with diarization via CLI
            elapsed = chunk_idx * args.chunk_seconds
            json_out = chunk_path.with_suffix(".json")
            cmd = [
                sys.executable, "-m", "mlx_qwen3_asr",
                str(chunk_path),
                "--model", args.model,
                "--language", args.language,
                "--diarize", "--num-speakers", "2",
                "-f", "json",
                "-o", str(tmpdir),
                "--no-progress", "--quiet",
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=120)
            except subprocess.TimeoutExpired:
                pass

            if json_out.exists():
                import json as js
                data = js.loads(json_out.read_text())
                segments = data if isinstance(data, list) else data.get("segments", data.get("chunks", []))
                prev_speaker = all_text[-1].split("\n")[0] if all_text else ""
                for seg in segments:
                    start = seg.get("start", 0)
                    text = seg.get("text", "").strip()
                    speaker = seg.get("speaker", "")
                    if not text:
                        continue
                    m, s = divmod(int(start + elapsed), 60)
                    h, m = divmod(m, 60)
                    ts = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
                    if speaker and speaker != prev_speaker:
                        print(f"\n{speaker}  {ts}", flush=True)
                        all_text.append(f"\n{speaker}  {ts}")
                        prev_speaker = speaker
                    print(f"{text}", flush=True)
                    all_text.append(text)
                json_out.unlink(missing_ok=True)
            else:
                # Fallback: simple transcription
                result = session.transcribe(str(chunk_path), language=args.language)
                text = result.text.strip() if hasattr(result, 'text') else ""
                if text:
                    m, s = divmod(elapsed, 60)
                    h, m = divmod(m, 60)
                    ts = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
                    print(f"[{ts}] {text}", flush=True)
                    all_text.append(f"[{ts}] {text}")

            # Cleanup chunk
            chunk_path.unlink(missing_ok=True)
            chunk_idx += 1

    except KeyboardInterrupt:
        print("\n\nStopped.", flush=True)
        if args.output and all_text:
            Path(args.output).write_text("\n".join(all_text), encoding="utf-8")
            print(f"Saved: {args.output}")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
