#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="$ROOT/data/dataset"
ARCHIVE=smart-np-dataset-v1.tar.gz

if ! command -v gh >/dev/null 2>&1; then
    echo "Install GitHub CLI first: https://cli.github.com/" >&2
    exit 1
fi
if find "$DATASET" -mindepth 1 -maxdepth 1 ! -name .gitkeep -print -quit | grep -q .; then
    echo "Data already exists in $DATASET; refusing to overwrite it" >&2
    exit 1
fi

gh auth status --hostname github.com >/dev/null 2>&1 || gh auth login --hostname github.com --web
DOWNLOAD_DIR=$(mktemp -d "$DATASET/.download.XXXXXX")
trap 'rm -rf -- "$DOWNLOAD_DIR"' EXIT
gh release download data-v1 --repo happyme99/SMART-NP --pattern "${ARCHIVE}*" --dir "$DOWNLOAD_DIR"
(cd "$DOWNLOAD_DIR" && sha256sum -c "${ARCHIVE}.sha256")
tar -xzf "$DOWNLOAD_DIR/$ARCHIVE" -C "$ROOT/data"
