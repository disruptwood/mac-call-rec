#!/usr/bin/env python3
"""Transcribe a call-recorder session using Qwen3-ASR + Silero VAD.

Pipeline:
  1. Silero VAD on mic and system tracks → speaker timestamps
  2. Qwen3-ASR on mixed recording → text with chunk timestamps
  3. Overlay VAD speaker labels on transcription chunks
  4. Output markdown with Я/Собеседница labels

Usage:
    python3 transcribe.py <session_dir> [--lang Russian] [--note "..."]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

QWEN_MODEL = "/Users/ilya/models/qwen3-asr-1.7b"


def load_pcm(path: str) -> "np.ndarray":
    import numpy as np
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True,
    )
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def run_vad(mic_path: str, sys_path: str) -> tuple[list, list]:
    import numpy as np
    import torch
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    get_stamps = utils[0]

    mic_stamps = []
    sys_stamps = []

    if Path(mic_path).exists() and Path(mic_path).stat().st_size > 1000:
        audio = load_pcm(mic_path)
        raw = get_stamps(torch.tensor(audio), model, sampling_rate=16000)
        mic_stamps = [(float(s["start"]) / 16000, float(s["end"]) / 16000) for s in raw]

    if Path(sys_path).exists() and Path(sys_path).stat().st_size > 1000:
        audio = load_pcm(sys_path)
        raw = get_stamps(torch.tensor(audio), model, sampling_rate=16000)
        sys_stamps = [(float(s["start"]) / 16000, float(s["end"]) / 16000) for s in raw]

    return mic_stamps, sys_stamps


def transcribe_mixed(mixed_path: str, language: str, model: str) -> list[dict]:
    from mlx_qwen3_asr import Session
    session = Session(model=model)
    r = session.transcribe(mixed_path, language=language, return_chunks=True)
    chunks = []
    if hasattr(r, "chunks") and r.chunks:
        for c in r.chunks:
            text = c["text"].strip()
            if text:
                chunks.append({"start": c["start"], "end": c.get("end", c["start"] + 1), "text": text})
    elif hasattr(r, "text") and r.text:
        chunks.append({"start": 0, "end": 0, "text": r.text.strip()})
    return chunks


def assign_speakers(chunks: list[dict], mic_stamps: list, sys_stamps: list) -> list[dict]:
    for chunk in chunks:
        start, end = chunk["start"], chunk["end"]
        mic_ov = sum(max(0, min(e, end) - max(s, start)) for s, e in mic_stamps)
        sys_ov = sum(max(0, min(e, end) - max(s, start)) for s, e in sys_stamps)
        if mic_ov > sys_ov and mic_ov > 0.1:
            chunk["speaker"] = "Я"
        elif sys_ov > 0.1:
            chunk["speaker"] = "Собеседница"
        else:
            chunk["speaker"] = None
    # Fill gaps with previous speaker
    prev = None
    for chunk in chunks:
        if chunk["speaker"] is None:
            chunk["speaker"] = prev or "Я"
        prev = chunk["speaker"]
    return chunks


def format_time(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    if hrs:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def format_markdown(chunks: list[dict], note: str | None, started_at: datetime, duration: float) -> str:
    days = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]
    months = ["", "января", "февраля", "марта", "апреля", "мая", "июня",
              "июля", "августа", "сентября", "октября", "ноября", "декабря"]
    date_str = f"{days[started_at.weekday()]}, {started_at.day} {months[started_at.month]} {started_at.year}, {started_at.strftime('%H:%M')}"
    mins = int(duration / 60)
    dur_str = f"{mins} мин" if mins < 60 else f"{mins // 60} ч {mins % 60} мин"

    lines = [f"# {note}" if note else "# Запись разговора", f"{date_str} — {dur_str}", ""]

    prev_speaker = None
    for chunk in chunks:
        speaker = chunk["speaker"]
        ts = format_time(chunk["start"])
        if speaker != prev_speaker:
            if prev_speaker is not None:
                lines.append("")
            lines.append(f"{speaker}  {ts}")
            prev_speaker = speaker
        lines.append(chunk["text"])

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Transcribe a call-recorder session")
    parser.add_argument("session_dir")
    parser.add_argument("--lang", default="Russian")
    parser.add_argument("--model", default=QWEN_MODEL)
    parser.add_argument("--note", default=None)
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    manifest_path = session_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"ERROR: No manifest.json in {session_dir}")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    print(f"Session: {manifest.get('session_id', '?')}", flush=True)
    print(f"Model: {args.model}", flush=True)

    # Find files
    recording = manifest.get("recording", {})
    mixed_path = session_dir / recording.get("file", "recording.m4a") if recording else None

    # Look for source tracks (for VAD)
    mic_path = None
    sys_path = None
    for f in sorted(session_dir.glob("_mic*")):
        mic_path = f
        break
    for f in sorted(session_dir.glob("_system*")):
        sys_path = f
        break

    if not mixed_path or not mixed_path.exists():
        print("ERROR: No recording.m4a found")
        sys.exit(1)

    # Step 1: VAD
    t = time.time()
    if mic_path and sys_path:
        print("VAD on source tracks...", flush=True)
        mic_stamps, sys_stamps = run_vad(str(mic_path), str(sys_path))
        print(f"  VAD: {time.time() - t:.1f}s (mic: {len(mic_stamps)} segments, system: {len(sys_stamps)} segments)", flush=True)
    else:
        print("No source tracks for VAD — transcribing without speaker labels", flush=True)
        mic_stamps, sys_stamps = [], []

    # Step 2: Transcribe mixed
    print("Transcribing...", flush=True)
    t = time.time()
    chunks = transcribe_mixed(str(mixed_path), args.lang, args.model)
    print(f"  Transcribe: {time.time() - t:.1f}s ({len(chunks)} chunks)", flush=True)

    # Step 3: Assign speakers
    if mic_stamps or sys_stamps:
        chunks = assign_speakers(chunks, mic_stamps, sys_stamps)

    # Step 4: Format and save
    started_at = datetime.fromisoformat(manifest["started_at"])
    duration = manifest.get("duration_seconds", 0)
    note = args.note or manifest.get("note")

    md = format_markdown(chunks, note, started_at, duration)

    output_path = session_dir / "transcript.md"
    output_path.write_text(md, encoding="utf-8")
    print(f"\nTranscript: {output_path}", flush=True)

    # Preview
    for line in md.split("\n")[:20]:
        print(f"  {line}", flush=True)


if __name__ == "__main__":
    main()
