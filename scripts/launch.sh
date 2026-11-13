#!/usr/bin/env bash
# ==============================================================================
# mlx-gen Studio Launcher
# ==============================================================================
# Starts the standalone Director Studio for mlx-gen on Apple Silicon.
# Usage:
#   ./scripts/launch.sh              # Start on default port 8080 and open browser
#   ./scripts/launch.sh --port 8188  # Use custom port
#   ./scripts/launch.sh --no-open    # Start without opening browser
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PORT=8080
OPEN_BROWSER=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --no-open)
      OPEN_BROWSER=false
      shift
      ;;
    -h|--help)
      echo "Usage: ./scripts/launch.sh [--port <port>] [--no-open]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo ""
echo "✦ mlx-gen Director Studio Launcher"
echo "-------------------------------------------------------"

# 1. Architecture check
ARCH="$(uname -m)"
OS="$(uname -s)"
if [[ "$OS" != "Darwin" || "$ARCH" != "arm64" ]]; then
  echo "⚠️  Warning: mlx-gen is designed for Apple Silicon (macOS arm64). Detected $OS $ARCH."
fi

# 2. Check Metal Toolchain
if ! xcrun metal --version &>/dev/null; then
  echo "ℹ️  Note: Xcode Metal Toolchain is not active."
  echo "   If you rebuild MLX kernels from source, run: xcodebuild -downloadComponent MetalToolchain"
fi

# 3. Check pmetal metallib cache (sc-7889)
if [[ -f "${PROJECT_ROOT}/tools/refresh_pmetal_metallib.sh" ]]; then
  echo "🔍 Checking local Metal kernel cache..."
  if ! bash "${PROJECT_ROOT}/tools/refresh_pmetal_metallib.sh" --check &>/dev/null; then
    echo "⚙️  Refreshing pmetal metallib cache..."
    bash "${PROJECT_ROOT}/tools/refresh_pmetal_metallib.sh" || true
  else
    echo "✓ Metal kernel cache verified."
  fi
fi

# 4. Open browser in background if requested
if [[ "$OPEN_BROWSER" == "true" ]]; then
  (
    sleep 1.2
    echo "🚀 Opening http://localhost:${PORT} in your browser..."
    open "http://localhost:${PORT}" 2>/dev/null || true
  ) &
fi

# 5. Start the Studio Server
echo "⚡ Starting Studio Server on port ${PORT}..."
cd "${PROJECT_ROOT}"
exec python3 "${PROJECT_ROOT}/scripts/studio_server.py" "${PORT}"
