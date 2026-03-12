#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_VERSION="3.12"
SYNC_ARGS=(--dev)
REQUEST_ALL=0

print_usage() {
  cat <<'EOF'
Usage: ./scripts/bootstrap.sh [--all] [--extra NAME ...]

Sets up the local development environment with uv:
- installs uv if missing
- installs required system packages on macOS or apt-based Linux
- installs Python 3.12 via uv
- runs uv sync
- creates .env from .env.example if needed

Options:
  --all           Install all optional backend extras
  --extra NAME    Install a specific optional extra (repeatable)
  -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      REQUEST_ALL=1
      shift
      ;;
    --extra)
      [[ $# -ge 2 ]] || { echo "Missing value for --extra" >&2; exit 1; }
      SYNC_ARGS+=(--extra "$2")
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_usage >&2
      exit 1
      ;;
  esac
done

if [[ "$REQUEST_ALL" -eq 1 ]]; then
  SYNC_ARGS+=(--extra all)
fi

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

install_uv() {
  if have_cmd uv; then
    return
  fi

  case "$(uname -s)" in
    Darwin)
      if have_cmd brew; then
        brew install uv
      else
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
      fi
      ;;
    Linux)
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$PATH"
      ;;
    *)
      echo "Unsupported OS for automatic uv install. Install uv first: https://docs.astral.sh/uv/" >&2
      exit 1
      ;;
  esac
}

install_system_packages() {
  case "$(uname -s)" in
    Darwin)
      if ! have_cmd brew; then
        echo "Homebrew is required for automatic macOS system package install." >&2
        echo "Install Homebrew or install ffmpeg and espeak-ng manually, then rerun this script." >&2
        exit 1
      fi
      brew install ffmpeg espeak-ng
      ;;
    Linux)
      if have_cmd apt-get; then
        sudo apt-get update
        sudo apt-get install -y ffmpeg espeak-ng
      else
        echo "Automatic Linux system package install currently supports apt-based distros only." >&2
        echo "Install ffmpeg and espeak-ng manually, then rerun this script." >&2
        exit 1
      fi
      ;;
    *)
      echo "Unsupported OS for automatic system package install." >&2
      exit 1
      ;;
  esac
}

echo "==> Ensuring uv is installed"
install_uv

echo "==> Installing system packages (ffmpeg, espeak-ng)"
install_system_packages

echo "==> Installing Python ${PYTHON_VERSION} via uv"
uv python install "$PYTHON_VERSION"

echo "==> Syncing project dependencies"
if [[ -f uv.lock ]]; then
  uv sync --frozen "${SYNC_ARGS[@]}"
else
  uv sync "${SYNC_ARGS[@]}"
fi

if [[ ! -f .env ]]; then
  echo "==> Creating .env from .env.example"
  cp .env.example .env
fi

cat <<'EOF'

Setup complete.

Run the app with:
  uv run python app.py
EOF
