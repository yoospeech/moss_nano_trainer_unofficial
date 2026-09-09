#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
: "${PYTHON:=python3}"
: "${CONFIG_NAME:=moss_16khz}"

# YAML is authoritative. Environment variables override only when supplied.
ARGS=()
add_override() {
  local variable="$1" key="$2"
  if [[ -n "${!variable:-}" ]]; then
    ARGS+=("$key=${!variable}")
  fi
}
add_override LIBRISPEECH_ROOT preprocess.datasets.LibriSpeech.root
add_override TRAIN_FILELIST dataset.train.filelist
add_override VAL_FILELIST dataset.val.filelist
add_override TEST_FILELIST dataset.test.filelist
add_override BATCH_SIZE dataset.train.batch_size
add_override DEVICES train.trainer.devices
add_override ACCELERATOR train.trainer.accelerator
add_override PRECISION train.trainer.precision
add_override MAX_STEPS train.trainer.max_steps
add_override LOG_DIR log_dir
add_override RESUME_CKPT ++resume_ckpt
exec "$PYTHON" train.py --config-name="$CONFIG_NAME" "${ARGS[@]}" "$@"
