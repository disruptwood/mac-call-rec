# Roadmap

## Текущий статус (v0.4)

- Запись 2 параллельных треков: PortAudio mic + Swift system audio
- Post-recording mix → `recording.m4a` (loudnorm mic + amix с system)
- Pause/resume через space, stop через `q`
- `diag_postmortem.py` показывает drift mic vs system + PortAudio counters
- **TUI выбора типа сессии при `python3 -m recorder`** (стрелки/цифры)
- **Локальные session presets** через `call-recorder init`,
  `call-recorder therapist add`, `call-recorder sessions`
- Транскрипция opt-in: `--transcribe local` или `--transcribe gemini`
- Преднастроенные сессии в `config.session_types`, дефолты нейтральные
- 104 unit-теста, все subprocess замоканы

## Сделано — фикс drift в записи (2026-05-20)

Ffmpeg avfoundation mic-путь удалён. PortAudio (sounddevice → CoreAudio HAL)
теперь единственный mic-источник. Подтверждено на реальной 75-минутной
сессии: чистое аудио, синхронизация со ScreenCaptureKit-system.

## Сделано — короткий запуск + безопасные session presets (2026-05-22)

- `python3 -m recorder` / `call-recorder` без аргументов открывает TUI меню.
  Поддержка стрелок и цифр; non-TTY fallback на numeric prompt.
- После `q` транскрипция не запускается сама, если backend не выбран явно.
- Сессии редактируются командами CLI и хранятся в
  `~/.call-recorder/config.json` (`session_types`).
- `scripts/bootstrap_macos.sh` готовит fresh clone: venv, package install,
  Swift helper, optional local ASR dependencies.
- Тесты: новые TestSessionTypes, TestChooseSessionLabelNonTTY,
  TestPromptCustomLabel, TestMaybeRunTranscription, TestDefaultSubcommand.
  Все subprocess (включая post-recording transcription hook) замоканы.

## Следующее

### Этап 2 — Заметка к сессии

После остановки — ввести описание встречи. Сохраняется в `manifest.json`,
попадает в заголовок `transcript.md`.

### Этап 3 — Prompt override per session type

Сейчас `SessionType` хранит только `name` + `label`. Добавить опциональное
поле prompt/context override — чтобы для интервью использовать prompt без
«психотерапевтической» формулировки, а для встреч — структурированный
протокол с задачами.

## Backlog

- Streaming транскрипция (Whisper local или Gemini streaming)
- Анализ сессий: суммаризация, темы, прогресс между встречами
- Очистка source-треков после подтверждения transcript
- Улучшить UX локальной транскрипции: проверка speaker profiles до запуска,
  понятный wizard для enrollment
- Архивация старых сессий

## Критические правила (не нарушать)

1. **Дорожки не должны разъезжаться** — никакой ротации/сегментации
2. **Source-треки не удалять автоматически**
3. **Никаких реальных процессов в тестах** — особенно ASR моделей в RAM
4. **E2e smoke перед реальной сессией** при любых изменениях в `recording.py`
