#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

: "${CKPT:?Set CKPT to the trained Lightning checkpoint path}"
: "${INPUT_DIR:?Set INPUT_DIR to a directory containing WAV files}"
: "${OUTPUT_DIR:=$ROOT_DIR/recon_wavs}"
: "${OUTPUT_CHANNEL:=1}"

: "${PYTHON:=python3}"
: "${CONFIG_NAME:=moss_16khz}"

"$PYTHON" inference.py --config-name="$CONFIG_NAME" \
  "++ckpt=\"$CKPT\"" \
  "++input_dir=\"$INPUT_DIR\"" \
  "++output_dir=\"$OUTPUT_DIR\"" \
  "++output_channel=$OUTPUT_CHANNEL" \
  "hydra.output_subdir=null" \
  "hydra.job.chdir=False" \
  "$@"


