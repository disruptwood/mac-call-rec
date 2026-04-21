#!/usr/bin/env python3
"""ONE-SHOT: transcribe using _mic.wav only (not recording.m4a).

Created 2026-04-21 to rescue transcripts from two sessions where
recording.m4a has mix bugs (mic/system drift + 96kHz artifact):
  - therapy_20260417_101109 (Мария)
  - therapy_20260421_131739 (Света)

Since _mic.wav has both voices (no headphones — therapist bleeds through
speakers into mic), we can run speaker identification directly on it.
mic timebase != system timebase, but internally consistent.

DELETE THIS SCRIPT after recording pipeline is fixed.

Usage:
    python3 oneshot_transcribe_mic.py <session_dir> --speakers "Илья,Мария"
"""

from __future__ import annotations

import sys
from pathlib import Path

# Reuse everything from transcribe.py
_here = Path(__file__).parent
sys.path.insert(0, str(_here))

# Monkeypatch: override audio file resolution to force _mic.wav
import transcribe  # noqa: E402

_orig_main = transcribe.main


def patched_main():
    """Force _mic.wav as audio source by temporarily renaming the recording."""
    import argparse
    import json
    import os
    import subprocess
    import time

    # Re-parse args (same as transcribe.main but we handle audio ourselves)
    parser = argparse.ArgumentParser(description="ONE-SHOT: transcribe from _mic.wav")
    parser.add_argument("session_dir")
    parser.add_argument("--lang", default="Russian")
    parser.add_argument("--model", default=transcribe.QWEN_MODEL)
    parser.add_argument("--note", default=None)
    parser.add_argument("--speakers-dir", default=str(transcribe.SPEAKERS_DIR))
    parser.add_argument("--speakers", default=None)
    parser.add_argument("--smooth-radius", type=int, default=1)
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    mic_path = session_dir / "_mic.wav"
    if not mic_path.exists():
        print(f"ERROR: {mic_path} not found")
        sys.exit(1)

    manifest_path = session_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: no manifest.json in {session_dir}")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())

    # Get actual _mic.wav duration
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(mic_path)],
        capture_output=True, text=True,
    )
    try:
        audio_duration = float(r.stdout.strip())
    except Exception:
        audio_duration = manifest.get("duration_seconds", 0)

    print(f"ONE-SHOT MODE: using _mic.wav ({audio_duration:.0f}s)", flush=True)
    print(f"Session: {manifest.get('session_id', '?')}", flush=True)

    # Load profiles
    speakers_dir = Path(args.speakers_dir)
    filter_names = [n.strip() for n in args.speakers.split(",")] if args.speakers else None
    print(f"\nLoading profiles from {speakers_dir} (filter: {filter_names})...", flush=True)
    profiles = transcribe.load_speaker_profiles(speakers_dir, filter_names=filter_names)
    if not profiles:
        print("ERROR: no profiles loaded", flush=True)
        sys.exit(1)

    t_total = time.time()

    # Step 1: ASR
    print(f"\n1/5 Transcribing _mic.wav with Qwen3-ASR...", flush=True)
    t = time.time()
    asr = transcribe.transcribe_asr(str(mic_path), args.lang, args.model)
    print(f"    Done: {time.time() - t:.1f}s, {len(asr['words'])} words", flush=True)

    # Step 2: VAD
    print(f"\n2/5 Silero VAD on _mic.wav...", flush=True)
    t = time.time()
    regions = transcribe.run_vad(str(mic_path))
    total_speech = sum(e - s for s, e in regions)
    print(f"    Done: {time.time() - t:.1f}s, {len(regions)} regions, "
          f"{total_speech:.0f}s speech ({100*total_speech/audio_duration:.1f}%)", flush=True)

    # Step 3: Windows + embeddings
    print(f"\n3/5 Windows + wespeaker embeddings...", flush=True)
    windows = transcribe.generate_windows(regions)
    print(f"    {len(windows)} windows", flush=True)
    embeddings = transcribe.compute_window_embeddings(str(mic_path), windows)

    # Step 4: Identify
    print(f"\n4/5 Identifying speakers...", flush=True)
    t = time.time()
    raw_speakers, scores = transcribe.identify_speakers(embeddings, profiles)
    smoothed = transcribe.smooth_speakers(raw_speakers, radius=args.smooth_radius)
    from collections import Counter
    distribution = Counter(smoothed)
    avg_conf = sum(scores) / len(scores) if scores else 0
    print(f"    Done: {time.time() - t:.1f}s", flush=True)
    print(f"    Distribution: {dict(distribution)}", flush=True)
    print(f"    Mean similarity: {avg_conf:.3f}", flush=True)
    if avg_conf < 0.35:
        print(f"    WARNING: low similarity — enrolled profiles may not match mic-domain voices", flush=True)

    # Step 5: Map words → speakers → utterances
    print(f"\n5/5 Assigning speakers to words...", flush=True)
    labeled_words = transcribe.assign_word_speakers(asr["words"], windows, smoothed)
    utterances = transcribe.group_into_utterances(labeled_words)

    coverage: dict[str, float] = {}
    for u in utterances:
        sp = u["speaker"]
        if sp:
            coverage[sp] = coverage.get(sp, 0) + (u["end"] - u["start"])

    # Write outputs with -mic suffix to not clobber existing transcript.md
    from datetime import datetime
    started_at_str = manifest.get("started_at")
    started_at = datetime.fromisoformat(started_at_str) if started_at_str else datetime.now()
    note = args.note or manifest.get("note")
    speakers_used = sorted(coverage.keys(), key=lambda n: -coverage[n])

    md = transcribe.format_markdown(utterances, note, started_at, audio_duration, speakers_used, coverage)
    out_md = session_dir / "transcript_mic.md"
    out_md.write_text(md, encoding="utf-8")

    out_json = session_dir / "transcript_mic.json"
    raw = {
        "note": "ONE-SHOT from _mic.wav; mic timebase != real time (see project memory)",
        "audio_file": str(mic_path.name),
        "audio_duration": audio_duration,
        "text": asr["text"],
        "language": asr["language"],
        "speakers_used": speakers_used,
        "coverage_seconds": coverage,
        "mean_similarity": avg_conf,
        "utterances": utterances,
        "windows": [
            {"start": s, "end": e, "speaker": sp, "score": sc}
            for (s, e), sp, sc in zip(windows, smoothed, scores)
        ],
    }
    out_json.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    total = time.time() - t_total
    print(f"\n✓ Transcript: {out_md}", flush=True)
    print(f"✓ Raw: {out_json}", flush=True)
    print(f"Total: {total:.0f}s for {audio_duration:.0f}s audio", flush=True)

    print("\n--- Preview ---")
    for line in md.split("\n")[:25]:
        print(line)


if __name__ == "__main__":
    patched_main()
