#!/usr/bin/env bash
# Run the portable CURP pipeline sequentially for multiple tasks.

set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GPU_ID=${1:-0}
TASK_LIST=${TASK_LIST:-"lamp1 lamp2 lamp3 lamp4 lamp5 lamp7"}

for TASK in $TASK_LIST; do
    bash "$ROOT_DIR/run_task.sh" "$TASK" "$GPU_ID"
done
