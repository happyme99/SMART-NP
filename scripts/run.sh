#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
unset VIRTUAL_ENV
export UV_PROJECT_ENVIRONMENT=SMART-NP
cd "$ROOT"
"$ROOT/scripts/build_env.sh"
exec uv run naturalrag run "$@"
