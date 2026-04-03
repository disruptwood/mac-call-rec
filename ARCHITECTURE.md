## Архитектура call-recorder

### Что это

CLI для записи обеих сторон голосовых звонков на macOS с автоматической транскрипцией и разделением реплик.

### Запись

```
python3 -m recorder start -l "therapy"
  ├─ ffmpeg (AVFoundation) → _mic.wav      [твой голос, PCM int16 mono 48kHz]
  └─ capture_system_audio  → _system.wav   [голос собеседника, ScreenCaptureKit]

При остановке (q):
  ├─ mic громкость нормализуется (loudnorm -16 LUFS)
  ├─ mic + system → recording.m4a          [микс для прослушивания]
  ├─ _mic.wav и _system.wav сохраняются    [для VAD-разметки спикеров]
  └─ manifest.json                          [метаданные сессии]
```

**Критические инварианты записи:**
- Mic и system записываются непрерывно, без ротации/сегментации
- Ротация сегментов ЗАПРЕЩЕНА — вызывает дрифт между дорожками (см. dec-20260403-001)
- Исходные файлы (_mic, _system) НИКОГДА не удаляются автоматически
- При паузе/resume: сегменты финализируются и склеиваются

**Громкость микрофона:**
- При старте записи: macOS input volume → 85% (был 50%, давал -42 dB)
- При остановке: восстанавливается оригинальное значение
- В mixed: mic нормализуется через loudnorm, system остаётся как есть

### Транскрипция (batch, после записи)

```
scripts/transcribe.py <session_dir>
  1. Silero VAD на _mic.wav → таймкоды "Я"           [~0.6 сек]
  2. Silero VAD на _system.wav → таймкоды "Собеседни:ца"  [~0.6 сек]
  3. Qwen3-ASR 1.7B на recording.m4a → текст + чанки  [1.6x real-time]
  4. Наложение VAD-таймкодов → speaker labels
  5. → transcript.md
```

**Почему так:**
- Один прогон Qwen3-ASR по mixed (не два по раздельным трекам) — быстрее
- VAD мгновенный (~0.3 сек на 60 сек аудио) — даёт 100% точные speaker labels по источнику
- pyannote не используется — слишком медленный (3 мин на 30 сек аудио)
- VAD не работает на музыке — это ожидаемо, только на речи

### Транскрипция (streaming, в разработке)

Текущий статус: **не работает**. Попытка читать растущие WAV файлы провалилась (галлюцинации на тишине).

Правильная архитектура (из ресёрча RealtimeSTT, WhisperLive):
```
mic (sounddevice)        → PCM 16kHz → Silero VAD gate → feed_audio → "Я: ..."
system (capture --pipe)  → PCM 16kHz → Silero VAD gate → feed_audio → "Собеседни:ца: ..."
```
Ключевое: **VAD как gate перед моделью** — тишина не подаётся в Qwen3-ASR, предотвращает галлюцинации.

### Стек технологий

| Компонент | Технология | Где |
|-----------|-----------|-----|
| Запись mic | ffmpeg + AVFoundation | recorder/recording.py |
| Запись system | Swift + ScreenCaptureKit | scripts/capture_system_audio.swift |
| Транскрипция | Qwen3-ASR 1.7B (MLX, GPU) | scripts/transcribe.py |
| VAD | Silero VAD (torch) | scripts/transcribe.py |
| Speaker labels | По источнику (mic/system) через VAD-таймкоды | scripts/transcribe.py |
| CLI | Python argparse + raw terminal | recorder/cli.py |

### Модели и окружения

| Что | Где | Размер |
|-----|-----|--------|
| Qwen3-ASR 1.7B | ~/models/qwen3-asr-1.7b | ~3.4 GB |
| Whisper large-v3 (legacy) | ~/models/whisper-large-v3 | ~3 GB |
| Whisper Hebrew (legacy) | ~/models/whisper-he | ~1.5 GB |
| Silero VAD | ~/.cache/torch/hub/ | ~2 MB |
| qwen-asr-env | ~/qwen-asr-env | Python 3.14 |
| whisper-env (legacy) | ~/whisper-env | Python 3.12 |

### Файлы сессии

```
~/.call-recorder/recordings/<label>_<YYYYMMDD_HHMMSS>/
  _mic.wav            ← твой голос (PCM, НЕ УДАЛЯТЬ)
  _system.wav         ← голос собеседника (PCM, НЕ УДАЛЯТЬ)
  recording.m4a       ← нормализованный микс для прослушивания
  manifest.json       ← метаданные, длительность, устройства
  transcript.md       ← транскрипт с Я/Собеседни:ца метками
```

### Известные проблемы

1. **VAD misattribution**: крупные чанки Qwen3-ASR (12-34 сек) могут содержать речь обоих — VAD назначает спикера по преобладающему overlap, граница нечёткая
2. **Mic тихий при WebRTC**: браузер может дополнительно снижать gain сверх наших 85%
3. **Формат аудио**: system.wav = ~700 MB/час (несжатый), можно оптимизировать
4. **ScreenCaptureKit exclusive access**: только один процесс может захватывать
5. **Streaming не реализован**: batch-режим 1.6x real-time, streaming требует VAD-gated pipeline

### Правила разработки

- **E2e тест перед реальной сессией** — запись 1 мин, проверить audio + transcript + labels
- **Никогда rm** — только mv в ~/.Trash/
- **Тесты не должны спавнить тяжёлые процессы** — mock _start_ffmpeg_wav, _start_system_capture
- **После pytest: проверить ps aux** на утечки
- **Длинные процессы** — давать команду пользователю, не запускать в фоне
- **Тестировать от малого к большому** — сначала токены/permissions, потом компоненты, потом pipeline
