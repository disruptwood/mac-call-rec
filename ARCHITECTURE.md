# Архитектура call-recorder

## Что это

CLI для записи обеих сторон голосовых звонков на macOS. Запись + диагностика
делается локально; транскрипция отправляется в Gemini (или локально, если есть
желание — см. ниже).

## Запись

Параллельно стартуют **три** capture-процесса при `python3 -m recorder start`:

```
mic (ffmpeg avfoundation)        → _mic.wav        [ffmpeg, WAV PCM s16le mono 48kHz]
mic (PortAudio sounddevice)      → _mic_pa.wav     [sounddevice, WAV PCM s16le mono 48kHz]
system (Swift ScreenCaptureKit)  → _system.wav     [WAV PCM s16le stereo 48kHz]
```

При остановке (`q`):
- Микс mic + system через ffmpeg loudnorm + amix → `recording.m4a`
- Source-треки (`_mic.wav`, `_mic_pa.wav`, `_system.wav`) НЕ удаляются
- `manifest.json` пишет длительность, время старта, состояние наушников

### Почему два mic-трека параллельно

Под активным WebRTC-звонком в браузере CoreAudio переводит mic-устройство
в VPIO-режим (Voice Processing I/O), который иногда тихо не доставляет frames.
ffmpeg avfoundation ходит через AudioUnit и ловит эту VPIO-просадку →
`_mic.wav` отстаёт от wall-clock на 15-18%.

PortAudio (через `sounddevice`) ходит в CoreAudio HAL **напрямую**
(`AudioDeviceCreateIOProcID`), минуя все AudioUnit'ы включая VPIO.
`_mic_pa.wav` должен оставаться синхронным с wall-clock независимо
от состояния браузерного WebRTC.

Пока что в `recording.m4a` идёт старый `_mic.wav` — после подтверждения
что PortAudio чистый на реальной сессии, в `_build_normalize_mix_cmd`
переключим на `_mic_pa.wav` и удалим ffmpeg-mic путь.

### Наушники — observability, не feature

`detect_headphones()` находит наушники один раз на старте, записывает
в manifest, и печатает WARN если их нет. **Логика записи на состояние
наушников не ветвится**, mid-session отключение наушников игнорируется.

Рекомендация: наушники включены → mic ловит только тебя, system ловит
собеседника, два чистых раздельных трека. Без наушников → mic ловит
обоих (bleed из колонок), хуже разделение.

## Транскрипция

**Основной путь — Gemini 3 Flash (cloud):**

```
recording.m4a → scripts/transcribe_gemini.py → transcript_<stem>_gemini_<model>.md
```

Скрипт делает upload в Files API, ждёт ACTIVE, шлёт промпт с просьбой
сделать диаризацию по таймкодам. Все safety-filters установлены в
BLOCK_NONE (терапия попадает в фильтры по умолчанию).

Требует `GEMINI_API_KEY` в `.env` (gitignored).

**Локальный путь (fallback, не основной):**

`scripts/transcribe.py` + `scripts/enroll_speakers.py` — mlx-whisper +
pyannote + enrolled speaker embeddings. Сохранён в репо на случай если
понадобится оффлайн или если Gemini будет не подходить. Запускается
вручную; в основной recording-pipeline не интегрирован.

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
- A/B drift verdict (`_mic.wav` vs `_mic_pa.wav` vs `_system.wav`)
- Что ffmpeg видел на входе (формат, частота)
- Status events PortAudio (overflows и т.п.)

`diag_preflight.py` — быстрая проверка mic/system синхронизации ДО
сессии (~10 сек запись).

## Файлы сессии

```
~/.call-recorder/recordings/<label>_<YYYYMMDD_HHMMSS>/
  _mic.wav                ← ffmpeg avfoundation mic (VPIO-affected)
  _mic.wav.ffmpeg.log     ← ffmpeg stderr (диагностика)
  _mic_pa.wav             ← PortAudio mic (HAL bypass, проверяем)
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
  __main__.py        ← python3 -m recorder
  cli.py             ← argparse subcommands: start, stop, status, devices, setup, config
  recording.py       ← RecordingEngine: lifecycle, segment management, ffmpeg/PA/Swift launch
  audio.py           ← AudioDevice/AudioSetup detection, headphone detection
  config.py          ← RecordingConfig, AudioProfile
  setup_helper.py    ← ffmpeg presence check, setup instructions print

scripts/
  capture_system_audio.swift  ← компилируется в capture_system_audio binary
  capture_mic_pa.py           ← PortAudio mic recorder (standalone subprocess)
  transcribe_gemini.py        ← Gemini transcription (основной)
  transcribe.py               ← mlx-whisper transcription (fallback, не интегрировано)
  enroll_speakers.py          ← enrollment для local pipeline (fallback)
  diag_postmortem.py          ← post-session drift analysis
  diag_preflight.py           ← pre-session quick sync check
  diag_diarize.py             ← (legacy)
  oneshot_transcribe_mic.py   ← (legacy)
  live_stream_transcribe.py   ← (legacy)

tests/                ← pytest, 63 теста, все процессы мокаются
  test_recording.py   ← lifecycle, ffmpeg cmd, PA capture, volume mgmt, mixing
  test_audio.py       ← device detection
  test_config.py      ← config + profiles
  test_setup_helper.py
  test_diag_postmortem.py
```

## Правила разработки

1. **Дорожки не разъезжаются** — никакой ротации/сегментации в активной
   записи кроме pause/resume. Сегменты склеиваются на остановке.
2. **Source-треки не удаляются автоматически** — `_mic.wav`, `_mic_pa.wav`,
   `_system.wav` нужны для диагностики и re-транскрипции.
3. **Никаких реальных subprocess в тестах.** Volume mgmt через osascript,
   ffmpeg, Swift binary, PortAudio Python — ВСЕ мокаются. См. шапку
   `tests/test_recording.py` со списком обязательных моков.
4. **Длинные процессы — командой пользователю**, не запускать в фоне из
   Claude. Запись/транскрипция идут минутами/часами.
5. **E2e smoke test перед каждой реальной сессией** — 60 сек запись с
   активным WebRTC в браузере + `diag_postmortem.py`.
