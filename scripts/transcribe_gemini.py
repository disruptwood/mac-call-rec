# /// script
# requires-python = ">=3.11"
# dependencies = ["google-genai>=0.8.0", "python-dotenv>=1.0.0"]
# ///
"""Transcribe an audio file with Gemini, including speaker diarization.

This is the primary post-recording transcription path. The local
mlx-whisper / pyannote / Qwen3 pipeline (scripts/transcribe.py and friends)
is kept as a fallback but is not invoked by default anymore.

Usage:
    # 1. Put your Gemini API key into project-root .env (gitignored).
    # 2. Either:
    uv run --no-project scripts/transcribe_gemini.py <path/to/audio.wav>
    # ...or with the project env directly:
    python3 scripts/transcribe_gemini.py <path/to/audio.wav>

Outputs (next to the input file):
    transcript_<stem>_gemini_<model>.md   — the transcript itself
    transcript_<stem>_gemini_<model>.json — full response + usage/safety meta
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

# Load .env from project root (two levels up from scripts/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


MODEL_ID = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

PROMPT = """Ты — профессиональный транскрибатор русскоязычной речи.

На входе — аудиозапись психотерапевтической сессии. В записи могут быть один или несколько говорящих (обычно терапевт и клиент).

Задача:
1. Полностью транскрибируй речь на русском языке.
2. Раздели транскрипт по репликам. Каждая реплика начинается с новой строки в формате:
   [MM:SS] Speaker N: <текст реплики>
   где MM:SS — таймкод начала реплики, N — номер говорящего (1, 2, ...).
3. Если говорящий один — используй Speaker 1 для всей записи.
4. Не додумывай содержание. Если фрагмент неразборчив — пиши [неразборчиво].
5. Сохраняй разговорный стиль: паузы, оговорки, частицы «ну», «э-э», «вот» — только если они реально произнесены.
6. Не добавляй никаких комментариев, заголовков, заключений — только сам транскрипт.
"""


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: temp_gemini_transcribe.py <audio.wav>", file=sys.stderr)
        sys.exit(2)

    audio_path = Path(sys.argv[1]).expanduser().resolve()
    if not audio_path.exists():
        print(f"not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    client = genai.Client()

    print(f"[1/3] uploading {audio_path.name} ({audio_path.stat().st_size / 1e6:.1f} MB)...", file=sys.stderr)
    t0 = time.time()
    uploaded = client.files.upload(file=str(audio_path))
    print(f"      uploaded in {time.time() - t0:.1f}s, uri={uploaded.uri}", file=sys.stderr)

    print("[2/3] waiting for file to become ACTIVE...", file=sys.stderr)
    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name != "ACTIVE":
        print(f"file failed to process: state={uploaded.state.name}", file=sys.stderr)
        sys.exit(1)

    # Therapy content routinely hits safety filters on sexuality / self-harm /
    # substance discussions — disable blocking so we get full transcripts.
    safety_settings = [
        genai_types.SafetySetting(category=cat, threshold="BLOCK_NONE")
        for cat in (
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_CIVIC_INTEGRITY",
        )
    ]

    print(f"[3/3] calling {MODEL_ID}...", file=sys.stderr)
    t0 = time.time()
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[PROMPT, uploaded],
        config=genai_types.GenerateContentConfig(
            safety_settings=safety_settings,
            max_output_tokens=65536,
        ),
    )
    elapsed = time.time() - t0
    print(f"      got response in {elapsed:.1f}s", file=sys.stderr)

    finish_reason = None
    safety_ratings = None
    if response.candidates:
        cand = response.candidates[0]
        finish_reason = getattr(cand.finish_reason, "name", str(cand.finish_reason))
        safety_ratings = [
            {
                "category": getattr(r.category, "name", str(r.category)),
                "probability": getattr(r.probability, "name", str(r.probability)),
                "blocked": getattr(r, "blocked", None),
            }
            for r in (cand.safety_ratings or [])
        ]
    print(f"      finish_reason={finish_reason}", file=sys.stderr)
    if finish_reason and finish_reason not in ("STOP", "MAX_TOKENS"):
        print(f"      WARNING: transcript may be truncated; safety_ratings={safety_ratings}", file=sys.stderr)

    out_dir = audio_path.parent
    stem = audio_path.stem
    suffix = MODEL_ID.replace(".", "_").replace("-", "_")
    md_path = out_dir / f"transcript_{stem}_gemini_{suffix}.md"
    json_path = out_dir / f"transcript_{stem}_gemini_{suffix}.json"

    md_path.write_text(response.text or "", encoding="utf-8")

    usage = getattr(response, "usage_metadata", None)
    meta = {
        "model": MODEL_ID,
        "audio": str(audio_path),
        "elapsed_sec": elapsed,
        "file_uri": uploaded.uri,
        "finish_reason": finish_reason,
        "safety_ratings": safety_ratings,
        "usage": {
            "prompt_token_count": getattr(usage, "prompt_token_count", None),
            "candidates_token_count": getattr(usage, "candidates_token_count", None),
            "total_token_count": getattr(usage, "total_token_count", None),
        } if usage else None,
        "text": response.text,
    }
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nwrote: {md_path}")
    print(f"wrote: {json_path}")


if __name__ == "__main__":
    main()
