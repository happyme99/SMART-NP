#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
unset VIRTUAL_ENV
export UV_PROJECT_ENVIRONMENT=SMART-NP

exec uv sync --locked
