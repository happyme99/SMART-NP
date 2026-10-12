#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
unset VIRTUAL_ENV
export UV_PROJECT_ENVIRONMENT=SMART-NP
"$ROOT/scripts/build_env.sh"
export PATH="$HOME/.local/bin:$PATH"
exec uv run naturalrag build-model "$@"
