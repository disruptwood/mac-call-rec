# Roadmap

## Текущий статус (v0.2)

- Запись 3 параллельных треков: ffmpeg mic + PortAudio mic (A/B test) + Swift system audio
- Post-recording mix → `recording.m4a` (пока через ffmpeg mic)
- Pause/resume через space, stop через `q`
- `diag_postmortem.py` показывает A/B drift verdict
- Транскрипция: ручной запуск `scripts/transcribe_gemini.py` после записи
- 63 unit-теста, все subprocess замоканы

## Сейчас в работе — фикс drift в записи

**Блокер всех остальных пунктов:** убедиться что PortAudio mic
(`_mic_pa.wav`) чистый при активном WebRTC браузерном звонке.

Шаги:
1. 60-сек smoke test: открыть Google Meet "Instant meeting" в Chrome →
   `python3 -m recorder start -l "vpio-test"` → `q` → `diag_postmortem.py`.
2. Реальная сессия с психологом — записать как обычно, потом
   `diag_postmortem.py` покажет drift обоих треков.

**Решение по результату:**
- PortAudio drift < 2% → переключить `_build_normalize_mix_cmd` на
  `_mic_pa.wav`, удалить ffmpeg-mic путь, обновить тесты
- PortAudio drift ≥ 5% → escalate к прямому CoreAudio HAL в Swift
  (`AudioDeviceCreateIOProcID`, ~200 строк, гарантированно мимо VPIO)

## После фикса записи

### Этап 1 — Интеграция Gemini в lifecycle

Сейчас `scripts/transcribe_gemini.py` запускается вручную. Можно
автоматизировать:
- После `q` спросить "Запустить транскрипцию? [Y/n]"
- Прогресс в том же терминале
- `--no-transcribe` флаг для записи без транскрипции

### Этап 2 — Заметка к сессии

После остановки — ввести описание встречи. Сохраняется в `manifest.json`,
попадает в заголовок `transcript.md`.

### Этап 3 — TUI для выбора типа сессии

```
$ rec
   ┌─ Выбери тип сессии ─┐
   │ ▶ Терапия с Марией │
   │   Терапия со Светой │
   │   Созвон            │
   │   Интервью          │
   └─────────────────────┘
```
Типы сессий в конфиге: имя, label, контекстный prompt override для Gemini.

## Backlog (не делать пока не подтверждён weekly use Gemini-пути)

- Streaming транскрипция (Whisper local или Gemini streaming)
- Анализ сессий: суммаризация, темы, прогресс между встречами
- Очистка source-треков после подтверждения transcript
- Поддержка локальной транскрипции в основном lifecycle
  (сейчас `scripts/transcribe.py` запускается только вручную)
- Архивация старых сессий

## Критические правила (не нарушать)

1. **Дорожки не должны разъезжаться** — никакой ротации/сегментации
2. **Source-треки не удалять автоматически**
3. **Никаких реальных процессов в тестах** — особенно ASR моделей в RAM
4. **E2e smoke перед реальной сессией** при любых изменениях в `recording.py`
