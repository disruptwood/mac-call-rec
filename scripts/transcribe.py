#!/usr/bin/env python3
"""Transcribe a call-recorder session with direct speaker identification.

Pipeline (no pyannote diarization — uses enrolled embeddings directly):
  1. mlx-whisper on recording.m4a with word timestamps → word-level text
  2. Silero VAD on recording.m4a → speech regions
  3. Sliding windows (2s with 1s hop) inside speech regions → wespeaker embeddings
  4. Each window: cosine similarity vs enrolled profiles → assign speaker
  5. Smooth speaker track (majority vote over 3 consecutive windows)
  6. Each ASR word → find overlapping window → inherit speaker
  7. Group consecutive same-speaker words → markdown

Why no pyannote diarization:
  - Diarization is unsupervised clustering — recomputes embeddings every run
  - We HAVE reference embeddings (enrolled profiles) — just use them directly
  - This is called "speaker identification", not diarization
  - Much faster (~10x) and uses our profiles continuously

Usage:
    python3 transcribe.py <session_dir> --speakers "Client,Therapist" --note "..."
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

WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"  # downloaded on first use to HF cache
EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
SPEAKERS_DIR = Path.home() / ".call-recorder" / "speakers"

# Embedding window parameters
WIN_SEC = 2.0    # seconds per embedding window
HOP_SEC = 1.0    # seconds between window starts (overlap = WIN_SEC - HOP_SEC)
MIN_REGION_SEC = 0.4  # skip VAD regions shorter than this

# Load HF token from file if not in env
_token_file = Path.home() / ".huggingface_token"
if _token_file.exists() and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = _token_file.read_text().strip()
if os.environ.get("HF_TOKEN") and not os.environ.get("PYANNOTE_AUTH_TOKEN"):
    os.environ["PYANNOTE_AUTH_TOKEN"] = os.environ["HF_TOKEN"]


def load_speaker_profiles(
    speakers_dir: Path, filter_names: list[str] | None = None,
) -> dict[str, "np.ndarray"]:
    """Load enrolled speaker embeddings from .npz files."""
    import numpy as np
    profiles = {}
    if not speakers_dir.exists():
        return profiles

    filter_set = {n.strip() for n in filter_names} if filter_names else None

    for f in sorted(speakers_dir.glob("*.npz")):
        data = np.load(f, allow_pickle=True)
        name = str(data["name"])
        if filter_set is not None and name not in filter_set:
            continue
        emb = data["embedding"]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        profiles[name] = emb
        print(f"  Loaded: {name}", flush=True)

    if filter_set:
        missing = filter_set - set(profiles.keys())
        if missing:
            print(f"  WARNING: requested profiles not found: {missing}", flush=True)
    return profiles


def run_vad(audio_path: str) -> list[tuple[float, float]]:
    """Silero VAD → list of (start_sec, end_sec) speech regions."""
    import numpy as np
    import torch

    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", audio_path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True,
    )
    audio = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0

    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    get_stamps = utils[0]
    stamps = get_stamps(torch.tensor(audio), model, sampling_rate=16000)
    return [(float(s["start"]) / 16000, float(s["end"]) / 16000) for s in stamps]


def generate_windows(regions: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Generate overlapping WIN_SEC windows inside each VAD region.

    For short regions (< WIN_SEC), yield a single window spanning the whole region.
    For longer regions, slide with HOP_SEC step.
    """
    windows = []
    for rs, re in regions:
        dur = re - rs
        if dur < MIN_REGION_SEC:
            continue
        if dur <= WIN_SEC:
            windows.append((rs, re))
            continue
        t = rs
        while t + WIN_SEC <= re:
            windows.append((t, t + WIN_SEC))
            t += HOP_SEC
        # Catch the trailing window if significant tail remains
        if re - (windows[-1][1] if windows else rs) > 0.5:
            windows.append((re - WIN_SEC, re))
    return windows


def compute_window_embeddings(
    audio_path: str, windows: list[tuple[float, float]],
) -> "np.ndarray":
    """Compute wespeaker embedding for each window. Returns (N, 256) array."""
    import numpy as np
    from pyannote.audio import Model, Inference
    from pyannote.core import Segment

    print(f"  Loading embedding model ({EMBEDDING_MODEL})...", flush=True)
    model = Model.from_pretrained(EMBEDDING_MODEL)
    try:
        import torch
        if torch.backends.mps.is_available():
            model = model.to(torch.device("mps"))
            print("  Embedding model on MPS", flush=True)
    except Exception:
        pass
    inference = Inference(model, window="whole")

    t0 = time.time()
    last_print = t0
    embeddings = np.zeros((len(windows), 256), dtype=np.float32)

    for i, (s, e) in enumerate(windows):
        try:
            emb = inference.crop(audio_path, Segment(s, e))
            norm = np.linalg.norm(emb)
            if norm > 0:
                embeddings[i] = emb / norm
        except Exception:
            pass  # zero vector — will match nothing strongly
        if time.time() - last_print > 3.0:
            pct = 100 * (i + 1) / len(windows)
            rate = (i + 1) / (time.time() - t0)
            eta = (len(windows) - i - 1) / rate if rate > 0 else 0
            print(f"    [{pct:4.1f}%] {i + 1}/{len(windows)} windows, {rate:.0f}/s, ETA {eta:.0f}s",
                  flush=True)
            last_print = time.time()

    print(f"  Embeddings: {time.time() - t0:.1f}s for {len(windows)} windows", flush=True)
    return embeddings


def identify_speakers(
    embeddings: "np.ndarray", profiles: dict[str, "np.ndarray"],
) -> tuple[list[str], list[float]]:
    """For each embedding, find best-matching profile. Returns (names, scores)."""
    import numpy as np

    names = list(profiles.keys())
    profile_mat = np.stack([profiles[n] for n in names])  # (P, 256)

    # Compute cosine similarity (embeddings and profiles already normalized)
    sim = embeddings @ profile_mat.T  # (N, P)

    # Zero rows (failed embedding) → score 0, assign most common as fallback
    has_emb = np.linalg.norm(embeddings, axis=1) > 0.1

    best_idx = np.argmax(sim, axis=1)
    best_scores = np.max(sim, axis=1)

    assigned_names = [names[i] for i in best_idx]
    # Mark failed ones
    for i in range(len(assigned_names)):
        if not has_emb[i]:
            assigned_names[i] = None
            best_scores[i] = 0.0
    return assigned_names, best_scores.tolist()


def smooth_speakers(speakers: list[str], radius: int = 1) -> list[str]:
    """Majority vote over (2*radius+1) consecutive windows. Fills None via neighbors."""
    from collections import Counter

    smoothed = list(speakers)
    n = len(speakers)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        votes = [s for s in speakers[lo:hi] if s is not None]
        if votes:
            smoothed[i] = Counter(votes).most_common(1)[0][0]
    return smoothed


def assign_word_speakers(
    words: list[dict], windows: list[tuple[float, float]], speakers: list[str],
) -> list[dict]:
    """For each word, find containing/overlapping window, inherit its speaker.

    Uses majority vote if word spans multiple windows.
    """
    from collections import Counter

    if not words or not windows:
        return [{"start": w.get("start", 0), "end": w.get("end", 0),
                 "text": w.get("text", ""), "speaker": None} for w in words]

    # Precompute window starts for quick bisect
    import bisect
    starts = [w[0] for w in windows]

    out = []
    for word in words:
        ws = float(word.get("start", 0))
        we = float(word.get("end", ws))
        text = str(word.get("text", "")).strip()

        # Find windows that overlap with [ws, we]
        lo = bisect.bisect_right(starts, ws) - 1
        if lo < 0:
            lo = 0
        candidates = []
        for j in range(lo, len(windows)):
            s, e = windows[j]
            if s > we:
                break
            overlap = max(0.0, min(e, we) - max(s, ws))
            if overlap > 0 and speakers[j] is not None:
                candidates.append((overlap, speakers[j]))

        if candidates:
            # Weight by overlap duration
            votes = Counter()
            for dur, spk in candidates:
                votes[spk] += dur
            speaker = votes.most_common(1)[0][0]
        else:
            speaker = None

        out.append({"start": ws, "end": we, "text": text, "speaker": speaker})

    # Fill gaps: inherit previous speaker for words with None
    prev = None
    for w in out:
        if w["speaker"] is None:
            w["speaker"] = prev
        else:
            prev = w["speaker"]
    # Back-fill leading Nones
    for w in out:
        if w["speaker"] is None:
            # Find first non-None
            for w2 in out:
                if w2["speaker"] is not None:
                    w["speaker"] = w2["speaker"]
                    break
    return out


def transcribe_asr(
    audio_path: str, language: str, model: str,
    condition_on_previous_text: bool = False,
    initial_prompt: str | None = None,
) -> dict:
    """Run mlx-whisper (large-v3) with word-level timestamps.

    mlx-whisper downloads the model to HF cache on first use.
    Returns dict with 'text', 'words' (word-level), 'language'.

    condition_on_previous_text: if True, each 30-sec window uses previous
        output as context. Better coherence but can cause repetition loops
        on silence. Default False (safer).
    initial_prompt: optional seed text for the first window. NOT included
        in output text (Whisper strips prompt tokens before returning).
        Useful for biasing toward proper names and domain vocabulary.
    """
    import mlx_whisper

    # mlx-whisper uses ISO 639-1 codes: "ru" not "Russian"
    lang_map = {"Russian": "ru", "English": "en"}
    lang_code = lang_map.get(language, language.lower()[:2])

    print(f"  Loading model {model} (first use downloads ~3 GB)...", flush=True)
    print(f"  condition_on_previous_text={condition_on_previous_text}, "
          f"initial_prompt={'yes (' + str(len(initial_prompt)) + ' chars)' if initial_prompt else 'no'}",
          flush=True)
    t0 = time.time()

    kwargs = dict(
        path_or_hf_repo=model,
        language=lang_code,
        word_timestamps=True,
        condition_on_previous_text=condition_on_previous_text,
        verbose=False,
    )
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt

    result = mlx_whisper.transcribe(audio_path, **kwargs)

    elapsed = time.time() - t0
    print(f"  ASR done: {elapsed:.1f}s", flush=True)

    # Extract words with timestamps. mlx-whisper segments have optional 'words' list.
    words = []
    segments = result.get("segments", []) if isinstance(result, dict) else []
    for seg in segments:
        seg_words = seg.get("words", [])
        if seg_words:
            for w in seg_words:
                text = str(w.get("word", "")).strip()
                if text:
                    words.append({
                        "text": text,
                        "start": float(w.get("start", 0)),
                        "end": float(w.get("end", w.get("start", 0))),
                    })
        else:
            # Fallback: segment-level only
            text = str(seg.get("text", "")).strip()
            if text:
                words.append({
                    "text": text,
                    "start": float(seg.get("start", 0)),
                    "end": float(seg.get("end", seg.get("start", 0))),
                })

    # Last resort: use overall text
    if not words:
        full_text = result.get("text", "").strip() if isinstance(result, dict) else ""
        if full_text:
            print("  WARNING: no segments, using full text as single span", flush=True)
            words.append({"text": full_text, "start": 0.0, "end": 0.0})

    return {
        "text": result.get("text", "") if isinstance(result, dict) else "",
        "language": result.get("language", language) if isinstance(result, dict) else language,
        "words": words,
    }


def format_time(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    if hrs:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def group_into_utterances(
    labeled_words: list[dict], max_gap_sec: float = 1.0,
) -> list[dict]:
    """Group consecutive same-speaker words into utterances.

    New utterance when speaker changes OR gap > max_gap_sec.
    """
    out = []
    for w in labeled_words:
        if not w["text"]:
            continue
        if not out:
            out.append({"speaker": w["speaker"], "start": w["start"],
                        "end": w["end"], "text": w["text"]})
            continue
        last = out[-1]
        gap = w["start"] - last["end"]
        if w["speaker"] == last["speaker"] and gap <= max_gap_sec:
            last["end"] = w["end"]
            last["text"] += " " + w["text"]
        else:
            out.append({"speaker": w["speaker"], "start": w["start"],
                        "end": w["end"], "text": w["text"]})
    return out


def format_markdown(
    utterances: list[dict], note: str | None, started_at: datetime, duration: float,
    speakers_used: list[str], coverage_stats: dict,
) -> str:
    days = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]
    months = ["", "января", "февраля", "марта", "апреля", "мая", "июня",
              "июля", "августа", "сентября", "октября", "ноября", "декабря"]
    date_str = f"{days[started_at.weekday()]}, {started_at.day} {months[started_at.month]} {started_at.year}, {started_at.strftime('%H:%M')}"
    mins = int(duration / 60)
    dur_str = f"{mins} мин" if mins < 60 else f"{mins // 60} ч {mins % 60} мин"

    lines = [
        f"# {note}" if note else "# Запись разговора",
        f"{date_str} — {dur_str}",
        "",
        "## Спикеры",
    ]
    for name in speakers_used:
        dur_speaker = coverage_stats.get(name, 0)
        pct = 100 * dur_speaker / duration if duration > 0 else 0
        lines.append(f"- **{name}**: {dur_speaker:.0f}s ({pct:.1f}%)")
    lines.append("")
    lines.append("## Расшифровка")
    lines.append("")

    prev_speaker = None
    for u in utterances:
        speaker = u["speaker"] or "?"
        ts = format_time(u["start"])
        if speaker != prev_speaker:
            if prev_speaker is not None:
                lines.append("")
            lines.append(f"**{speaker}**  `{ts}`")
            prev_speaker = speaker
        lines.append(u["text"])

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Transcribe session using enrolled-embedding speaker ID")
    parser.add_argument("session_dir")
    parser.add_argument("--lang", default="Russian")
    parser.add_argument("--model", default=WHISPER_MODEL)
    parser.add_argument("--note", default=None)
    parser.add_argument("--speakers-dir", default=str(SPEAKERS_DIR))
    parser.add_argument("--speakers", default=None,
                        help="Comma-separated profile names (e.g. 'Client,Therapist')")
    parser.add_argument("--smooth-radius", type=int, default=1,
                        help="Majority-vote radius for smoothing (default: 1 = 3-window window)")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: no manifest.json in {session_dir}")
        sys.exit(1)

    t_total = time.time()
    manifest = json.loads(manifest_path.read_text())
    audio_duration = manifest.get("duration_seconds", 0)
    print(f"Session: {manifest.get('session_id', '?')} ({audio_duration:.0f}s audio)", flush=True)

    recording = manifest.get("recording", {})
    mixed_path = session_dir / recording.get("file", "recording.m4a") if recording else None
    if not mixed_path or not mixed_path.exists():
        mixed_path = session_dir / "recording.m4a"
    if not mixed_path.exists():
        print(f"ERROR: no recording.m4a in {session_dir}")
        sys.exit(1)

    # Load enrolled profiles
    speakers_dir = Path(args.speakers_dir)
    filter_names = [n.strip() for n in args.speakers.split(",")] if args.speakers else None
    print(f"\nLoading profiles from {speakers_dir} (filter: {filter_names})...", flush=True)
    profiles = load_speaker_profiles(speakers_dir, filter_names=filter_names)
    if not profiles:
        print("ERROR: no profiles loaded", flush=True)
        sys.exit(1)
    if len(profiles) < 2:
        print(f"WARNING: only {len(profiles)} profile(s) loaded — all speech will map to one", flush=True)

    # Step 1: ASR with word timestamps
    print(f"\n1/5 Transcribing with mlx-whisper (word timestamps)...", flush=True)
    t = time.time()
    asr = transcribe_asr(str(mixed_path), args.lang, args.model)
    print(f"    Done: {time.time() - t:.1f}s, {len(asr['words'])} words", flush=True)

    # Step 2: VAD
    print(f"\n2/5 Running Silero VAD on {mixed_path.name}...", flush=True)
    t = time.time()
    regions = run_vad(str(mixed_path))
    total_speech = sum(e - s for s, e in regions)
    print(f"    Done: {time.time() - t:.1f}s, {len(regions)} regions, "
          f"{total_speech:.0f}s of speech ({100*total_speech/audio_duration:.1f}%)", flush=True)

    # Step 3: Windows + embeddings
    print(f"\n3/5 Generating windows and computing embeddings...", flush=True)
    windows = generate_windows(regions)
    print(f"    {len(windows)} windows ({WIN_SEC}s with {HOP_SEC}s hop)", flush=True)
    embeddings = compute_window_embeddings(str(mixed_path), windows)

    # Step 4: Identify speaker per window, smooth
    print(f"\n4/5 Identifying speakers (cosine similarity vs {len(profiles)} profiles)...", flush=True)
    t = time.time()
    raw_speakers, scores = identify_speakers(embeddings, profiles)
    smoothed_speakers = smooth_speakers(raw_speakers, radius=args.smooth_radius)

    # Stats
    from collections import Counter
    distribution = Counter(smoothed_speakers)
    avg_conf = sum(scores) / len(scores) if scores else 0
    print(f"    Done: {time.time() - t:.1f}s", flush=True)
    print(f"    Window distribution: {dict(distribution)}", flush=True)
    print(f"    Mean similarity score: {avg_conf:.3f}", flush=True)

    # Step 5: Map ASR words to speakers via windows
    print(f"\n5/5 Assigning speakers to ASR words...", flush=True)
    labeled_words = assign_word_speakers(asr["words"], windows, smoothed_speakers)
    utterances = group_into_utterances(labeled_words)

    # Speaker coverage
    coverage: dict[str, float] = {}
    for u in utterances:
        sp = u["speaker"]
        if sp:
            coverage[sp] = coverage.get(sp, 0) + (u["end"] - u["start"])

    # Write markdown
    started_at_str = manifest.get("started_at")
    started_at = datetime.fromisoformat(started_at_str) if started_at_str else datetime.now()
    note = args.note or manifest.get("note")
    speakers_used = sorted(coverage.keys(), key=lambda n: -coverage[n])

    md = format_markdown(utterances, note, started_at, audio_duration, speakers_used, coverage)
    out_md = session_dir / "transcript.md"
    out_md.write_text(md, encoding="utf-8")

    # Raw data
    out_json = session_dir / "transcript.json"
    raw = {
        "text": asr["text"],
        "language": asr["language"],
        "speakers_used": speakers_used,
        "coverage_seconds": coverage,
        "mean_similarity": avg_conf,
        "utterances": utterances,
        "windows": [
            {"start": s, "end": e, "speaker": sp, "score": sc}
            for (s, e), sp, sc in zip(windows, smoothed_speakers, scores)
        ],
    }
    out_json.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    total = time.time() - t_total
    print(f"\n✓ Transcript: {out_md}", flush=True)
    print(f"✓ Raw data: {out_json}", flush=True)
    print(f"Total: {total:.0f}s for {audio_duration:.0f}s audio ({audio_duration/total:.2f}x real-time)",
          flush=True)

    # Preview
    print("\n--- Preview ---")
    for line in md.split("\n")[:25]:
        print(line)


if __name__ == "__main__":
    main()
