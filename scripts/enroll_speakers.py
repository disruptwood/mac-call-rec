#!/usr/bin/env python3
"""Extract and save speaker voice embeddings for future transcription.

Uses pyannote/wespeaker-voxceleb-resnet34-LM to compute speaker embeddings
from known-clean audio segments.

For therapy sessions without headphones:
  - _system.wav contains clean therapist voice
  - _mic.wav contains user voice + therapist bleed from speaker
  - To get clean user voice: use mic segments where system VAD is silent

Usage:
    # Enroll therapist from system track (clean):
    python3 enroll_speakers.py <session_dir> --name "Therapist" --source system

    # Enroll user from mic track (auto-filters using VAD):
    python3 enroll_speakers.py <session_dir> --name "Client" --source mic-clean

    # List saved speakers:
    python3 enroll_speakers.py --list

Profiles saved to: ~/.call-recorder/speakers/<name>.npz
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SPEAKERS_DIR = Path.home() / ".call-recorder" / "speakers"
EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"

# Load HF token from file if not in env (stored by live_stream_transcribe.py convention)
_token_file = Path.home() / ".huggingface_token"
if _token_file.exists() and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = _token_file.read_text().strip()


def load_pcm(path: str, sr: int = 16000) -> np.ndarray:
    """Load audio as float32 PCM via ffmpeg."""
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True,
    )
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def run_vad(audio: np.ndarray, sr: int = 16000) -> list[tuple[float, float]]:
    """Run Silero VAD, return list of (start_sec, end_sec) speech segments."""
    import torch
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    get_stamps = utils[0]
    raw = get_stamps(torch.tensor(audio), model, sampling_rate=sr)
    return [(float(s["start"]) / sr, float(s["end"]) / sr) for s in raw]


def get_embedding_model():
    from pyannote.audio import Model, Inference
    model = Model.from_pretrained(EMBEDDING_MODEL)
    return Inference(model, window="whole")


def extract_embedding_from_segments(
    inference, audio_path: str, segments: list[tuple[float, float]],
    min_duration: float = 1.0, max_segments: int = 50,
) -> np.ndarray:
    """Extract embeddings from multiple segments and average them."""
    from pyannote.core import Segment

    embeddings = []
    for start, end in segments:
        if (end - start) < min_duration:
            continue
        try:
            emb = inference.crop(audio_path, Segment(start, end))
            embeddings.append(emb)
        except Exception:
            continue
        if len(embeddings) >= max_segments:
            break

    if not embeddings:
        print("ERROR: No valid segments found for embedding extraction")
        sys.exit(1)

    # Average all segment embeddings
    mean_emb = np.mean(embeddings, axis=0)
    # L2 normalize
    mean_emb = mean_emb / np.linalg.norm(mean_emb)
    return mean_emb


def enroll_from_system(session_dir: Path, name: str) -> None:
    """Enroll speaker from _system.wav (clean therapist voice)."""
    sys_path = None
    for f in sorted(session_dir.glob("_system*")):
        sys_path = f
        break

    if not sys_path or not sys_path.exists():
        print(f"ERROR: No _system.wav in {session_dir}")
        sys.exit(1)

    print(f"Loading audio: {sys_path.name} ({sys_path.stat().st_size / 1e6:.0f} MB)")

    # VAD to find speech segments
    t = time.time()
    audio = load_pcm(str(sys_path))
    segments = run_vad(audio)
    print(f"  VAD: {time.time() - t:.1f}s, {len(segments)} speech segments")

    if not segments:
        print("ERROR: No speech found in system track")
        sys.exit(1)

    # Extract embeddings
    t = time.time()
    inference = get_embedding_model()
    embedding = extract_embedding_from_segments(inference, str(sys_path), segments)
    print(f"  Embedding: {time.time() - t:.1f}s")

    save_profile(name, embedding, source="system", session=session_dir.name)


def enroll_from_mic_clean(session_dir: Path, name: str) -> None:
    """Enroll user from _mic.wav, using only segments where system is silent."""
    mic_path = sys_path = None
    for f in sorted(session_dir.glob("_mic*")):
        mic_path = f
        break
    for f in sorted(session_dir.glob("_system*")):
        sys_path = f
        break

    if not mic_path or not mic_path.exists():
        print(f"ERROR: No _mic.wav in {session_dir}")
        sys.exit(1)

    print(f"Loading mic: {mic_path.name} ({mic_path.stat().st_size / 1e6:.0f} MB)")
    print(f"Loading system: {sys_path.name} ({sys_path.stat().st_size / 1e6:.0f} MB)")

    # VAD on both tracks
    t = time.time()
    mic_audio = load_pcm(str(mic_path))
    sys_audio = load_pcm(str(sys_path))
    mic_segments = run_vad(mic_audio)
    sys_segments = run_vad(sys_audio)
    print(f"  VAD: {time.time() - t:.1f}s (mic: {len(mic_segments)}, sys: {len(sys_segments)})")

    # Find mic segments where system is silent (= only user speaking)
    def overlaps_system(start: float, end: float) -> bool:
        for ss, se in sys_segments:
            if ss < end and se > start:
                overlap = min(se, end) - max(ss, start)
                if overlap > 0.3:  # more than 300ms overlap = therapist speaking
                    return True
        return False

    clean_mic = [(s, e) for s, e in mic_segments if not overlaps_system(s, e)]
    print(f"  Clean mic segments (no system overlap): {len(clean_mic)} of {len(mic_segments)}")

    if not clean_mic:
        print("ERROR: No clean mic segments found. Try with headphones.")
        sys.exit(1)

    # Extract embeddings from clean mic segments
    t = time.time()
    inference = get_embedding_model()
    embedding = extract_embedding_from_segments(inference, str(mic_path), clean_mic)
    print(f"  Embedding: {time.time() - t:.1f}s")

    save_profile(name, embedding, source="mic-clean", session=session_dir.name)


def save_profile(name: str, embedding: np.ndarray, source: str, session: str) -> None:
    """Save speaker profile."""
    SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPEAKERS_DIR / f"{name}.npz"
    np.savez(path, embedding=embedding, name=name, source=source, session=session)
    print(f"\nSaved: {path}")
    print(f"  Name: {name}, Source: {source}, Dim: {embedding.shape[0]}")


def list_speakers() -> None:
    """List saved speaker profiles."""
    if not SPEAKERS_DIR.exists():
        print("No speakers enrolled yet.")
        return

    profiles = sorted(SPEAKERS_DIR.glob("*.npz"))
    if not profiles:
        print("No speakers enrolled yet.")
        return

    print(f"Speaker profiles ({SPEAKERS_DIR}):\n")
    for p in profiles:
        data = np.load(p, allow_pickle=True)
        name = str(data.get("name", p.stem))
        source = str(data.get("source", "?"))
        session = str(data.get("session", "?"))
        print(f"  {name:15s}  source={source:10s}  session={session}")


def main():
    parser = argparse.ArgumentParser(description="Enroll speaker voice profiles")
    parser.add_argument("session_dir", nargs="?", help="Session directory")
    parser.add_argument("--name", help="Speaker name")
    parser.add_argument("--source", choices=["system", "mic-clean"],
                        help="system = therapist from _system.wav, mic-clean = user from _mic.wav filtered by VAD")
    parser.add_argument("--list", action="store_true", help="List saved speakers")
    args = parser.parse_args()

    if args.list:
        list_speakers()
        return

    if not args.session_dir or not args.name or not args.source:
        parser.error("Required: session_dir, --name, --source")

    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        print(f"ERROR: {session_dir} not found")
        sys.exit(1)

    print(f"Enrolling '{args.name}' from {session_dir.name} ({args.source})\n")

    if args.source == "system":
        enroll_from_system(session_dir, args.name)
    elif args.source == "mic-clean":
        enroll_from_mic_clean(session_dir, args.name)


if __name__ == "__main__":
    main()
