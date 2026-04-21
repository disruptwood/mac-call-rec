#!/usr/bin/env python3
"""Diagnostic: run pyannote speaker-diarization-3.1 standalone on an audio file.

Compares three setups to diagnose why mlx_qwen3_asr produced 1 cluster:
  1. pyannote with num_speakers=2 (forced)
  2. pyannote without num_speakers (auto-detect)
  3. pyannote with min_speakers=2, max_speakers=4 (ranged)

For each: prints unique speakers, per-speaker duration, first few segments.
Also runs on _system.wav alone to verify pyannote can detect 1 speaker there.

Usage:
    python3 diag_diarize.py <session_dir>
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Load token
_token_file = Path.home() / ".huggingface_token"
if _token_file.exists() and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = _token_file.read_text().strip()


def load_pipeline(use_mps: bool = True):
    from pyannote.audio import Pipeline
    import torch

    token = os.environ.get("HF_TOKEN", "")
    print(f"Loading pipeline (token={'yes' if token else 'NO'})...", flush=True)
    t = time.time()
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)

    if use_mps and torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
        print(f"  Moved to MPS ({time.time() - t:.1f}s)", flush=True)
    else:
        print(f"  Running on CPU ({time.time() - t:.1f}s)", flush=True)
    return pipeline


def summarize(diarization, label: str, duration: float):
    from collections import defaultdict
    print(f"\n=== {label} ===", flush=True)

    speakers = defaultdict(float)
    segments = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        dur = segment.end - segment.start
        speakers[speaker] += dur
        segments.append((speaker, segment.start, segment.end))

    print(f"  Unique speakers: {len(speakers)}", flush=True)
    total = sum(speakers.values())
    for spk, dur in sorted(speakers.items(), key=lambda x: -x[1]):
        pct = 100 * dur / duration if duration > 0 else 0
        print(f"    {spk}: {dur:.1f}s ({pct:.1f}% of audio)", flush=True)
    print(f"  Total segments: {len(segments)}", flush=True)
    print(f"  First 10 segments:", flush=True)
    for spk, s, e in segments[:10]:
        print(f"    {spk:12s}  {s:7.1f}s -> {e:7.1f}s  ({e-s:.1f}s)", flush=True)
    if len(segments) > 10:
        print(f"    ... and {len(segments) - 10} more", flush=True)


def run_diarize(pipeline, audio_path: str, duration: float, **kwargs):
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    t = time.time()
    with ProgressHook() as hook:
        try:
            diarization = pipeline(audio_path, hook=hook, **kwargs)
            elapsed = time.time() - t
            print(f"\n  Diarization: {elapsed:.1f}s", flush=True)
            return diarization
        except Exception as e:
            print(f"\n  ERROR: {type(e).__name__}: {e}", flush=True)
            return None


def get_audio_duration(path: str) -> float:
    import subprocess
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def main():
    if len(sys.argv) < 2:
        print("Usage: diag_diarize.py <session_dir>")
        sys.exit(1)

    session_dir = Path(sys.argv[1])
    mixed = session_dir / "recording.m4a"
    system_only = session_dir / "_system.wav"

    if not mixed.exists():
        print(f"ERROR: {mixed} not found")
        sys.exit(1)

    duration = get_audio_duration(str(mixed))
    print(f"Audio: {mixed.name} ({duration:.0f}s)")

    pipeline = load_pipeline(use_mps=True)

    # Test 1: forced 2 speakers on mixed — does pyannote even find 2?
    d1 = run_diarize(pipeline, str(mixed), duration, num_speakers=2)
    if d1 is not None:
        summarize(d1, "TEST 1: mixed, num_speakers=2 (forced)", duration)

    # Test 2: sanity — pyannote on clean _system.wav should find 1 speaker
    if system_only.exists():
        sys_dur = get_audio_duration(str(system_only))
        print(f"\n\nSanity: running on _system.wav ({sys_dur:.0f}s, should find 1 speaker)...")
        d2 = run_diarize(pipeline, str(system_only), sys_dur)
        if d2 is not None:
            summarize(d2, "TEST 2: _system.wav alone (sanity)", sys_dur)


if __name__ == "__main__":
    main()
