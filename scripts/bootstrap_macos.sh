#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_LOCAL_ASR=0

usage() {
  cat <<'EOF'
Usage:
  scripts/bootstrap_macos.sh [--local-asr]

Options:
  --local-asr   Install the heavy local transcription dependencies too.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local-asr)
      INSTALL_LOCAL_ASR=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "Project: $ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.11+ first." >&2
  exit 1
fi

if command -v brew >/dev/null 2>&1; then
  for pkg in ffmpeg portaudio; do
    if ! brew list "$pkg" >/dev/null 2>&1; then
      echo "Installing Homebrew package: $pkg"
      brew install "$pkg"
    fi
  done
else
  echo "WARN: Homebrew not found. Install ffmpeg and portaudio manually." >&2
fi

python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip setuptools wheel

INSTALL_TARGET="$ROOT"
if [[ "$INSTALL_LOCAL_ASR" -eq 1 ]]; then
  INSTALL_TARGET="$ROOT[local-asr]"
fi
"$ROOT/.venv/bin/python" -m pip install -e "$INSTALL_TARGET"

if ! command -v xcrun >/dev/null 2>&1; then
  echo "ERROR: xcrun not found. Install Xcode Command Line Tools: xcode-select --install" >&2
  exit 1
fi

xcrun -sdk macosx swiftc \
  "$ROOT/scripts/capture_system_audio.swift" \
  -o "$ROOT/scripts/capture_system_audio"

mkdir -p "$HOME/.local/bin"
ln -sf "$ROOT/.venv/bin/call-recorder" "$HOME/.local/bin/call-recorder"

cat <<EOF

Done.

Next:
  export PATH="\$HOME/.local/bin:\$PATH"
  call-recorder init --therapist "Therapist Name"
  call-recorder

If you installed local ASR:
  call-recorder start --transcribe local
EOF
