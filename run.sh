#!/usr/bin/env bash
# run.sh — NeuTTS local UI launcher
#
# Sets up a uv-managed virtual environment, compiles llama-cpp-python with the
# right hardware backend for your platform, then starts the Gradio UI.
#
# Usage:
#   ./run.sh                      # start on http://127.0.0.1:7860
#   ./run.sh --port 8080          # custom port
#   ./run.sh --share              # create a public Gradio link
#   ./run.sh --host 0.0.0.0       # listen on all interfaces (LAN access)
#
# Environment variable overrides:
#   NEUTTS_HOST   bind address   (default: 127.0.0.1)
#   NEUTTS_PORT   port           (default: 7860)
#
# To force a fresh Metal recompile of llama-cpp-python:
#   rm .venv/.llama_metal && ./run.sh

set -euo pipefail
IFS=$'\n\t'

# ─── Colour helpers ───────────────────────────────────────────────────────────

RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
BLU='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${GRN}[neutts]${NC} $*"; }
warn()  { echo -e "${YLW}[neutts]${NC} $*"; }
error() { echo -e "${RED}[neutts]${NC} $*" >&2; }
die()   { error "$*"; exit 1; }

# ─── Platform detection ───────────────────────────────────────────────────────

OS="$(uname -s)"
ARCH="$(uname -m)"
IS_MACOS_ARM=false
[[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]] && IS_MACOS_ARM=true

# ─── Argument parsing ─────────────────────────────────────────────────────────

HOST="${NEUTTS_HOST:-127.0.0.1}"
PORT="${NEUTTS_PORT:-7860}"
APP_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --share)           APP_ARGS+=("--share") ;;
        --host=*)          HOST="${arg#*=}"; APP_ARGS+=("--host" "${arg#*=}") ;;
        --host)            : ;;   # handled below with next arg
        --port=*)          PORT="${arg#*=}"; APP_ARGS+=("--port" "${arg#*=}") ;;
        --port)            : ;;
        *)                 APP_ARGS+=("$arg") ;;
    esac
done

# Simple two-argument handling for --host VALUE and --port VALUE
prev=""
for arg in "$@"; do
    if [[ "$prev" == "--host" ]]; then HOST="$arg"; fi
    if [[ "$prev" == "--port" ]]; then PORT="$arg"; fi
    prev="$arg"
done

# ─── Require: uv ─────────────────────────────────────────────────────────────

require_uv() {
    if command -v uv &>/dev/null; then
        info "uv $(uv --version | awk '{print $2}') found."
        return
    fi

    warn "uv not found — installing via the official installer..."
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        die "Failed to install uv automatically.
  Install it manually and re-run:
    curl -LsSf https://astral.sh/uv/install.sh | sh
  or visit: https://docs.astral.sh/uv/getting-started/installation/"
    fi

    # Bring uv into PATH for this session
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv &>/dev/null; then
        die "uv was installed but is not in PATH.
  Open a new shell, then run this script again."
    fi
    info "uv $(uv --version | awk '{print $2}') installed."
}

# ─── Require: Python 3.10–3.13 ───────────────────────────────────────────────

require_python() {
    if uv python find ">=3.10,<3.14" &>/dev/null 2>&1; then
        return
    fi
    info "Python 3.10-3.13 not found — installing 3.11 via uv..."
    uv python install 3.11 \
        || die "Could not install Python 3.11.
  Install it manually: https://python.org/downloads/"
}

# ─── Require: espeak-ng ───────────────────────────────────────────────────────

require_espeak() {
    # Apple Silicon Homebrew installs to /opt/homebrew
    if [[ "$OS" == "Darwin" ]]; then
        for dir in /opt/homebrew/bin /usr/local/bin; do
            [[ -x "$dir/espeak-ng" ]] && { export PATH="$dir:$PATH"; return; }
        done
    fi

    if command -v espeak-ng &>/dev/null; then
        return
    fi

    warn "espeak-ng not found."

    if [[ "$OS" == "Darwin" ]]; then
        if command -v brew &>/dev/null; then
            info "Installing espeak-ng via Homebrew..."
            brew install espeak-ng \
                || die "Homebrew install failed.  Run manually: brew install espeak-ng"
        else
            die "espeak-ng is required and Homebrew is not installed.
  Install Homebrew first:  https://brew.sh
  Then run:  brew install espeak-ng"
        fi

    elif [[ "$OS" == "Linux" ]]; then
        info "Attempting to install espeak-ng..."
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y espeak-ng
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y espeak-ng
        elif command -v pacman &>/dev/null; then
            sudo pacman -S --noconfirm espeak-ng
        else
            die "Cannot auto-install espeak-ng.  Install it with your package manager or from:
  https://github.com/espeak-ng/espeak-ng/releases"
        fi

    else
        die "espeak-ng not found.  Install from: https://github.com/espeak-ng/espeak-ng/releases"
    fi
}

# ─── Create virtual environment ───────────────────────────────────────────────

create_venv() {
    if [[ -d ".venv" ]]; then
        info ".venv already exists — skipping creation."
        return
    fi
    info "Creating virtual environment..."
    uv venv --python ">=3.10,<3.14"
}

# ─── Install Python dependencies ──────────────────────────────────────────────

# Stamp file — prevents recompiling llama-cpp-python on every run.
# Delete it to force a fresh Metal recompile: rm .venv/.llama_metal
METAL_STAMP=".venv/.llama_metal"

install_deps() {
    info "Installing NeuTTS + ONNX + UI dependencies..."
    uv pip install -e ".[onnx,ui]" \
        || die "Dependency install failed.  Check the error above."

    if $IS_MACOS_ARM; then
        # ── Apple Silicon: compile llama-cpp-python with Metal ───────────────
        if [[ -f "$METAL_STAMP" ]]; then
            info "llama-cpp-python (Metal) already compiled — skipping."
        else
            warn "Compiling llama-cpp-python with Apple Metal support."
            warn "This is a one-time step and typically takes 5–15 minutes."
            warn "Ensure Xcode Command Line Tools are installed:  xcode-select --install"
            echo ""

            CMAKE_ARGS="-DGGML_METAL=ON" \
            uv pip install \
                --no-binary llama-cpp-python \
                --reinstall-package llama-cpp-python \
                llama-cpp-python \
            && touch "$METAL_STAMP" \
            || die "Metal compilation failed.
  Common fixes:
    • Install Xcode tools:  xcode-select --install
    • Ensure CMake is installed:  brew install cmake
    • Check that Xcode is up to date in the App Store"
            info "Metal build complete."
        fi

    else
        # ── Other platforms ─────────────────────────────────────────────────
        if command -v nvcc &>/dev/null; then
            info "CUDA detected — compiling llama-cpp-python with CUDA support..."
            CMAKE_ARGS="-DGGML_CUDA=ON" \
            uv pip install --no-binary llama-cpp-python llama-cpp-python \
            || {
                warn "CUDA compilation failed — falling back to CPU-only build."
                uv pip install llama-cpp-python
            }
        else
            info "Installing llama-cpp-python (CPU build)..."
            uv pip install llama-cpp-python \
                || die "llama-cpp-python install failed.  Check the error above."
        fi
    fi
}

# ─── Port availability check ─────────────────────────────────────────────────

check_port() {
    if command -v lsof &>/dev/null; then
        if lsof -iTCP:"$PORT" -sTCP:LISTEN &>/dev/null 2>&1; then
            warn "Port $PORT is already in use.  Gradio will pick the next available port."
        fi
    fi
}

# ─── Launch UI ────────────────────────────────────────────────────────────────

launch() {
    info "Starting NeuTTS UI..."
    info "Open in your browser: http://${HOST}:${PORT}"
    echo ""

    .venv/bin/python app.py --host "$HOST" --port "$PORT" "${APP_ARGS[@]+"${APP_ARGS[@]}"}"
}

# ─── Preflight checks ────────────────────────────────────────────────────────

preflight() {
    # Must run from the repo root (where app.py lives)
    if [[ ! -f "app.py" || ! -f "pyproject.toml" ]]; then
        die "Run this script from the neutts repo root directory."
    fi
}

# ─── Main ────────────────────────────────────────────────────────────────────

main() {
    echo ""
    echo -e "${BLU}NeuTTS — Local Voice Synthesis${NC}"
    echo    "────────────────────────────────"
    echo    "Platform: $OS / $ARCH"
    $IS_MACOS_ARM && echo "Hardware: Apple Silicon (Metal available)"
    echo ""

    preflight
    require_uv
    require_python
    require_espeak
    create_venv
    install_deps
    check_port
    launch
}

main "$@"
