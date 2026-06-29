# call-recorder

CLI для записи обеих сторон голосового звонка на macOS.

По умолчанию проект ничего не отправляет в облако: запись сохраняется локально
в `~/.call-recorder/recordings`, а транскрипция запускается только если явно
выбран backend.

## Установка на новом Mac

```bash
git clone <repo-url> call-recorder
cd call-recorder
./scripts/bootstrap_macos.sh
```

Что делает bootstrap:
- ставит системные зависимости через Homebrew, если он доступен: `ffmpeg`, `portaudio`
- создает `.venv` в проекте
- ставит Python-пакет в editable mode
- собирает `scripts/capture_system_audio` из Swift source
- добавляет symlink `~/.local/bin/call-recorder`

Для локальной транскрипции сразу ставь тяжелые ML-зависимости:

```bash
./scripts/bootstrap_macos.sh --local-asr
```

Если `~/.local/bin` не в `PATH`, добавь в `~/.zshrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Первый запуск

Один раз сохрани имя психолога для меню:

```bash
call-recorder init --therapist "Name"
```

Это пишет только локальный файл `~/.call-recorder/config.json`. Он не хранится
в репозитории. После этого запуск:

```bash
call-recorder
```

Откроется меню: стрелки/цифры выбирают тип сессии, Enter стартует запись.

Во время записи:
- `[space]` — пауза/resume
- `[q]` или Ctrl+C — стоп

Посмотреть меню:

```bash
call-recorder sessions
```

Добавить еще одного психолога:

```bash
call-recorder therapist add "Another Name"
```

## Permissions macOS

В `System Settings -> Privacy & Security` разреши терминалу, из которого
запускается CLI:
- `Screen Recording` — для системного звука через ScreenCaptureKit
- `Microphone` — для микрофона через PortAudio

Проверка окружения:

```bash
call-recorder setup
```

## Запись без меню

```bash
call-recorder start -l therapy
call-recorder start -l therapy --no-transcribe
```

Папка сессии:

```text
~/.call-recorder/recordings/<label>_YYYYMMDD_HHMMSS/
```

Внутри остаются source-треки (`_mic_pa.wav`, `_system.wav`), микс
`recording.m4a`, логи и `manifest.json`.

## Локальная транскрипция

Установи extra:

```bash
./scripts/bootstrap_macos.sh --local-asr
```

Запуск после записи:

```bash
call-recorder start --transcribe local
```

Или вручную на готовой сессии:

```bash
python3 scripts/transcribe.py ~/.call-recorder/recordings/<session>
```

Первый запуск локальных моделей может долго скачивать веса в Hugging Face cache.
Для speaker identification нужны заранее сохраненные voice profiles:

```bash
python3 scripts/enroll_speakers.py <session_dir> --name "Therapist" --source system
python3 scripts/enroll_speakers.py <session_dir> --name "Client" --source mic-clean
```

## Опционально: Gemini

Cloud-транскрипция оставлена как опция, но не включена по умолчанию.

```bash
cp .env.example .env
$EDITOR .env
call-recorder start --transcribe gemini
```

`.env` находится в `.gitignore`; не коммить реальные ключи.

## Тесты

```bash
pytest tests/
```

Тесты не должны запускать реальные процессы записи, Swift helper, ffmpeg,
Python subprocess транскрипции или ML-модели.

## Безопасность Git

В репозитории не должны храниться:
- `.env`, API keys, Hugging Face tokens
- `~/.call-recorder/config.json`
- аудиофайлы и transcripts
- `scripts/capture_system_audio` compiled binary
- `.venv`, `.pytest_cache`, локальные IDE/agent настройки

Проверка перед push:

```bash
git status --short
git grep -nE '/Users/|qwen-asr-env|therapy-svetlana|therapy-maria' -- ':!README.md'
git grep -nE 'GEMINI_API_KEY=.+|HF_TOKEN=.+' -- ':!.env.example' ':!README.md'
```
