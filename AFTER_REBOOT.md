## После перезагрузки

> Когда все шаги выполнены и всё работает — агент может удалить этот файл.

### 1. Проверить что BlackHole загрузился

```bash
cd /Users/ilya/Downloads/Therapy/call-recorder
python3 -m recorder devices
```

BlackHole 2ch должен появиться в списке.

### 2. Создать Multi-Output Device

```bash
swift scripts/create_multi_output.swift
```

Это создаст виртуальное устройство "Record Output", которое отправляет звук одновременно в колонки/наушники И в BlackHole (для записи).

### 3. Переключить системный выход звука

System Settings → Sound → Output → выбрать **"Record Output"**

Или через терминал:
```bash
brew install switchaudio-osx
SwitchAudioSource -s "Record Output"
```

### 4. Запустить запись

```bash
cd /Users/ilya/Downloads/Therapy/call-recorder

# С профилем по умолчанию (наушники + мик мака):
python3 -m recorder start

# С другим профилем:
python3 -m recorder start -p speaker
python3 -m recorder start -p headphones

# С меткой:
python3 -m recorder start -l "звонок-с-Петей"
```

Ctrl+C чтобы остановить.

### 5. Где записи

```
~/.call-recorder/recordings/<session_id>/
  mic.m4a      — твой голос
  system.m4a   — голос собеседника
  mixed.m4a    — оба вместе
```

### Полезные команды

```bash
python3 -m recorder status      # статус текущей записи
python3 -m recorder profiles    # список профилей
python3 -m recorder setup       # диагностика системы
python3 -m recorder config      # показать конфиг
```
