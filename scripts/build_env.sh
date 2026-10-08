#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
unset VIRTUAL_ENV
export UV_PROJECT_ENVIRONMENT=naturalRAG

if command -v uv >/dev/null 2>&1; then
    printf '%-41s  %-12s  %s\n' "[naturalRAG uv environment exists]" "" "$(uv --version)"
else
    echo "[naturalRAG] uv missing; installing..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    else
        echo "[naturalRAG] uv install needs curl or wget" >&2
        exit 1
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "[naturalRAG] uv installation failed" >&2
    exit 1
fi

[[ -x "$ROOT/naturalRAG/bin/python" ]] || exec uv sync --locked
