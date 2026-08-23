#!/usr/bin/env bash
# Portable CURP pipeline for one task.
# Usage: LLM_PATH=... ENCODER_PATH=... bash run_task.sh <task> [gpu_id]

set -euo pipefail

TASK=${1:-}
GPU_ID=${2:-0}

if [[ -z "$TASK" ]]; then
    echo "Usage: LLM_PATH=... ENCODER_PATH=... bash run_task.sh <task> [gpu_id]"
    exit 1
fi

: "${LLM_PATH:?Set LLM_PATH to a local model directory or model identifier}"
: "${ENCODER_PATH:?Set ENCODER_PATH to a local encoder directory or model identifier}"

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
DATA_DIR=${DATA_DIR:-"$ROOT_DIR/data"}
STAGE1_DIR=${STAGE1_DIR:-"$ROOT_DIR/checkpoints/stage1"}
STAGE2_DIR=${STAGE2_DIR:-"$ROOT_DIR/checkpoints/stage2"}
RESULTS_DIR=${RESULTS_DIR:-"$ROOT_DIR/results"}
QUANTIZER=${QUANTIZER:-pq}

case "$QUANTIZER" in
    pq)
        CODEBOOK_DIR="$STAGE1_DIR/$TASK"
        ;;
    rq|opq)
        CODEBOOK_DIR="$STAGE1_DIR/${TASK}_${QUANTIZER}"
        ;;
    *)
        echo "QUANTIZER must be one of: pq, rq, opq"
        exit 1
        ;;
esac

CODEBOOK_PATH="$CODEBOOK_DIR/stage1_final/codebook_model.pt"
STAGE2_TASK_DIR="$STAGE2_DIR/${TASK}_${QUANTIZER}"
PREDICTIONS_PATH="$RESULTS_DIR/${TASK}_${QUANTIZER}_val_predictions.jsonl"
METRICS_PATH="$RESULTS_DIR/${TASK}_${QUANTIZER}_val_metrics.json"

mkdir -p "$CODEBOOK_DIR/stage1_final" "$STAGE2_TASK_DIR" "$RESULTS_DIR"
cd "$ROOT_DIR"

if [[ "${SKIP_STAGE1:-0}" != "1" ]]; then
    if [[ "$QUANTIZER" == "pq" ]]; then
        CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" stage1.py \
            --embeddings_path "$DATA_DIR/${TASK}_train_embeddings.pt" \
            --output_dir "$CODEBOOK_DIR" \
            --codebook_size "${CODEBOOK_SIZE:-1000}" \
            --num_subspaces "${NUM_SUBSPACES:-4}" \
            --use_balanced_kmeans_init \
            --num_epochs "${STAGE1_EPOCHS:-10}" \
            --batch_size "${STAGE1_BATCH_SIZE:-32}" \
            --batch_size_per_sample "${STAGE1_SAMPLE_BATCH_SIZE:-8}"
    else
        CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" build_altquant_codebook.py \
            --embeddings_path "$DATA_DIR/${TASK}_train_embeddings.pt" \
            --method "$QUANTIZER" \
            --output_path "$CODEBOOK_PATH" \
            --codebook_size "${CODEBOOK_SIZE:-1000}" \
            --num_subspaces "${NUM_SUBSPACES:-4}" \
            --num_stages "${NUM_STAGES:-4}" \
            --sample "${CODEBOOK_SAMPLE_SIZE:-200000}" \
            --iters "${CODEBOOK_ITERS:-25}"
    fi
fi

PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} \
CUDA_VISIBLE_DEVICES="$GPU_ID" "$ACCELERATE_BIN" launch --num_processes 1 stage2.py \
    --data_path "$DATA_DIR/${TASK}_train.jsonl" \
    --embeddings_path "$DATA_DIR/${TASK}_train_embeddings.pt" \
    --llm_path "$LLM_PATH" \
    --codebook_path "$CODEBOOK_PATH" \
    --encoder_path "$ENCODER_PATH" \
    --output_dir "$STAGE2_TASK_DIR" \
    --num_query_tokens "${NUM_QUERY_TOKENS:-4}" \
    --qformer_layers "${QFORMER_LAYERS:-2}" \
    --num_heads "${NUM_HEADS:-4}" \
    --num_epochs "${STAGE2_EPOCHS:-3}" \
    --batch_size "${STAGE2_BATCH_SIZE:-1}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-32}" \
    --max_length "${MAX_LENGTH:-1024}" \
    --learning_rate "${LEARNING_RATE:-1e-4}" \
    --save_steps "${SAVE_STEPS:-500}" \
    --task "$TASK" \
    --ablation "${ABLATION:-both}"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" infer.py \
    --data_path "$DATA_DIR/${TASK}_val.jsonl" \
    --embeddings_path "$DATA_DIR/${TASK}_val_embeddings.pt" \
    --llm_path "$LLM_PATH" \
    --codebook_path "$CODEBOOK_PATH" \
    --encoder_path "$ENCODER_PATH" \
    --checkpoint_path "$STAGE2_TASK_DIR/stage2_final/model.pt" \
    --output_path "$PREDICTIONS_PATH" \
    --batch_size "${INFER_BATCH_SIZE:-4}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-128}" \
    --task "$TASK" \
    --ablation "${ABLATION:-both}"

EVAL_ARGS=(
    --predictions_path "$PREDICTIONS_PATH"
    --output_path "$METRICS_PATH"
    --device "${EVAL_DEVICE:-cuda}"
    --task "$TASK"
)
if [[ -n "${BERT_MODEL_PATH:-}" ]]; then
    EVAL_ARGS+=(--bert_model_path "$BERT_MODEL_PATH")
fi
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" evaluate.py "${EVAL_ARGS[@]}"

echo "Metrics written to $METRICS_PATH"
