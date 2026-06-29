# Архитектура call-recorder

## Что это

CLI для записи обеих сторон голосовых звонков на macOS. Запись и диагностика
делаются локально. Транскрипция не запускается по умолчанию; пользователь явно
выбирает local или Gemini backend.

## Запись

Параллельно стартуют **два** capture-процесса при `python3 -m recorder start`:

```
mic (PortAudio sounddevice)      → _mic_pa.wav     [sounddevice, WAV PCM s16le mono 48kHz]
system (Swift ScreenCaptureKit)  → _system.wav     [WAV PCM s16le stereo 48kHz]
```

При остановке (`q`):
- Микс mic + system через ffmpeg loudnorm + amix → `recording.m4a`
- Source-треки (`_mic_pa.wav`, `_system.wav`) НЕ удаляются
- `manifest.json` пишет длительность, время старта, состояние наушников

### Почему PortAudio для mic, а не ffmpeg avfoundation

Под активным WebRTC-звонком в браузере CoreAudio переводит mic-устройство
в VPIO-режим (Voice Processing I/O), который иногда тихо не доставляет frames.
ffmpeg avfoundation ходит через AudioUnit и ловит эту VPIO-просадку →
mic-трек отставал от wall-clock на 15-40% (наблюдали в Safari, Google Meet,
ВКонтакте Web).

PortAudio (через Python `sounddevice`) ходит в CoreAudio HAL **напрямую**
(`AudioDeviceCreateIOProcID`), минуя все AudioUnit'ы включая VPIO. Это
та же библиотека, что использует Audacity, REAPER, sox — 20+ лет mature
кода. Verified 2026-05-20 на реальной сессии: PortAudio дал чистый,
синхронный со ScreenCaptureKit аудио-трек.

Параллельный ffmpeg avfoundation путь (`_mic.wav`) был удалён 2026-05-20.
Если PortAudio не стартует (нет permission, dep, device busy), сессия
пишет только system-track, без mic-fallback.

### Наушники — observability, не feature

`detect_headphones()` находит наушники один раз на старте, записывает
в manifest, и печатает WARN если их нет. **Логика записи на состояние
наушников не ветвится**, mid-session отключение наушников игнорируется.

Рекомендация: наушники включены → mic ловит только тебя, system ловит
собеседника, два чистых раздельных трека. Без наушников → mic ловит
обоих (bleed из колонок), хуже разделение.

## Транскрипция

**Local backend:**

`scripts/transcribe.py` + `scripts/enroll_speakers.py` — mlx-whisper +
pyannote + enrolled speaker embeddings. Это offline-путь для локальной
транскрипции после установки extra `local-asr`.

CLI может запустить его после записи:

```
call-recorder start --transcribe local
```

**Gemini backend (cloud, optional):**

```
recording.m4a → scripts/transcribe_gemini.py → transcript_<stem>_gemini_<model>.md
```

Скрипт делает upload в Files API, ждёт ACTIVE, шлёт промпт с просьбой
сделать диаризацию по таймкодам. Все safety-filters установлены в
BLOCK_NONE (терапия попадает в фильтры по умолчанию).

Требует `GEMINI_API_KEY` в `.env` (gitignored). Не запускается без явного
`--transcribe gemini` или локальной настройки `transcription_backend=gemini`.

`scripts/diag_diarize.py`, `scripts/oneshot_transcribe_mic.py`,
`scripts/live_stream_transcribe.py` — экспериментальные / legacy, не
поддерживаются, не запускаются автоматически.

## Диагностика

После каждой сессии:
```
python3 scripts/diag_postmortem.py ~/.call-recorder/recordings/<session>
```
Покажет:
- Длительность каждого трека vs wall-clock
- Drift `_mic_pa.wav` vs `_system.wav` (должен быть < 1% на здоровой сессии)
- Status events PortAudio (input_overflow, queue_full_drops и т.п.)
- Логи ScreenCaptureKit (mid-session halts через SCStreamDelegate)

Поддерживает legacy сессии с обоими `_mic.wav` (ffmpeg) и `_mic_pa.wav`
файлами — выведет A/B сравнение для них.

`diag_preflight.py` — быстрая проверка mic/system синхронизации ДО
сессии (~10 сек запись).

## Файлы сессии

```
~/.call-recorder/recordings/<label>_<YYYYMMDD_HHMMSS>/
  _mic_pa.wav             ← PortAudio mic (CoreAudio HAL, VPIO-bypass)
  _mic_pa.wav.pa.log      ← PortAudio diagnostic
  _system.wav             ← ScreenCaptureKit system audio
  _system.wav.capture.log ← Swift binary stderr
  recording.m4a           ← loudnorm + amix микс для прослушивания
  manifest.json           ← session_id, длительность, наушники, recording size
  transcript_*.md         ← (опционально) после запуска transcribe_gemini.py
```

## Структура кода

```
recorder/
  __main__.py        ← python3 -m recorder (без аргументов = start с picker)
  cli.py             ← argparse subcommands: start, stop, status, devices, setup,
                       init, therapist, sessions, config. Содержит TUI picker,
                       local session presets и post-recording transcribe hook.
  recording.py       ← RecordingEngine: lifecycle, segment management, PA/Swift launch
  audio.py           ← AudioDevice/AudioSetup detection, headphone detection
  config.py          ← RecordingConfig, AudioProfile, SessionType
  setup_helper.py    ← ffmpeg presence check, setup instructions print

scripts/
  capture_system_audio.swift  ← компилируется в capture_system_audio binary
  capture_mic_pa.py           ← PortAudio mic recorder (standalone subprocess)
  bootstrap_macos.sh          ← fresh-machine setup for macOS
  transcribe.py               ← mlx-whisper transcription (local backend)
  enroll_speakers.py          ← enrollment для local pipeline
  transcribe_gemini.py        ← optional Gemini transcription
  diag_postmortem.py          ← post-session drift analysis
  diag_preflight.py           ← pre-session quick sync check
  diag_diarize.py             ← (legacy)
  oneshot_transcribe_mic.py   ← (legacy)
  live_stream_transcribe.py   ← (legacy)

tests/                ← pytest, 104 теста, все процессы мокаются
  test_recording.py   ← lifecycle, PA capture, system capture, volume mgmt, mixing
  test_audio.py       ← device detection
  test_config.py      ← config + profiles + session types
  test_cli.py         ← TUI session picker, post-recording transcribe prompt
  test_setup_helper.py
  test_diag_postmortem.py
```

## Правила разработки

1. **Дорожки не разъезжаются** — никакой ротации/сегментации в активной
   записи кроме pause/resume. Сегменты склеиваются на остановке.
2. **Source-треки не удаляются автоматически** — `_mic_pa.wav` и
   `_system.wav` нужны для диагностики и re-транскрипции.
3. **Никаких реальных subprocess в тестах.** Volume mgmt через osascript,
   Swift binary, PortAudio Python — ВСЕ мокаются. См. шапку
   `tests/test_recording.py` со списком обязательных моков.
4. **Длинные процессы — командой пользователю**, не запускать в фоне из
   Claude. Запись/транскрипция идут минутами/часами.
5. **E2e smoke test перед каждой реальной сессией** — 60 сек запись с
   активным WebRTC в браузере + `diag_postmortem.py`.
