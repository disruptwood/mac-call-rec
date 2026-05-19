# call-recorder

Запись обеих сторон голосовых звонков на macOS + транскрипция через Gemini.

## Quickstart

```bash
# Однократная установка
brew install ffmpeg                                       # для mic-трека
python3 -m venv ~/qwen-asr-env && source ~/qwen-asr-env/bin/activate
pip install -e ".[gemini]"                                # recording + Gemini
xcrun -sdk macosx swiftc scripts/capture_system_audio.swift -o scripts/capture_system_audio
cp .env.example .env && $EDITOR .env                      # вписать GEMINI_API_KEY

# Permissions (System Settings → Privacy & Security):
#   - Screen Recording: разрешить Terminal/iTerm/IDE (для ScreenCaptureKit)
#   - Microphone: разрешить Terminal/iTerm/IDE и Python (для PortAudio)
```

## Запись

```bash
source ~/qwen-asr-env/bin/activate
python3 -m recorder start -l "therapy"
# наушники включены → лучше разделение голосов; без них работает, но с bleed
# [space] пауза/resume, [q] стоп
```

Появится папка `~/.call-recorder/recordings/therapy_YYYYMMDD_HHMMSS/`
с тремя WAV-источниками и одним m4a-миксом — см. [ARCHITECTURE.md](ARCHITECTURE.md#файлы-сессии).

## Диагностика после записи

```bash
python3 scripts/diag_postmortem.py ~/.call-recorder/recordings/therapy_*
```

Покажет drift каждого mic-трека против system. Ожидаемое здоровое
состояние: drift < 2%. Drift > 5% обычно означает что WebRTC в браузере
посадил mic в VPIO режим — см. [ARCHITECTURE.md](ARCHITECTURE.md#почему-два-mic-трека-параллельно).

## Транскрипция

```bash
python3 scripts/transcribe_gemini.py ~/.call-recorder/recordings/therapy_*/recording.m4a
```

Положит `transcript_recording_gemini_*.md` рядом с входным файлом.
Промпт настроен на русскоязычную терапевтическую сессию с диаризацией
по таймкодам.

## Тесты

```bash
pytest tests/                  # 63 теста, все subprocess замоканы, ~50ms
```

Тесты НЕ должны спавнить реальные процессы (ffmpeg, Swift binary, Python
скрипты, ML модели). См. шапку `tests/test_recording.py` для списка
обязательных моков.

## Где что лежит

- [ARCHITECTURE.md](ARCHITECTURE.md) — что и почему так устроено,
  включая объяснение VPIO drift и почему два mic-трека параллельно
- [ROADMAP.md](ROADMAP.md) — что дальше и в каком порядке
