#!/usr/bin/env bash
# ============================================================
# Tuning to Survive - Unified Run Script
# Usage: bash run_pipeline.sh [config_file_path]
# Default config file: scripts/config.yaml
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG_FILE="${1:-$SCRIPT_DIR/config.yaml}"

# ---------- Dependency Check ----------
if ! command -v yq &>/dev/null; then
    echo "Error: yq is required to parse YAML configuration file"
    echo "  pip install yq   or   https://github.com/mikefarah/yq"
    exit 1
fi

# ---------- Helper Functions to Read YAML Config ----------
cfg()    { yq ".$1" "$CONFIG_FILE"; }
cfg_str(){ cfg "$1" | sed 's/^"\(.*\)"$/\1/'; }   # Remove quotes from yq output
cfg_int(){ printf "%.0f" "$(cfg "$1")"; }
cfg_bool(){
    local v
    v=$(cfg_str "$1")
    [[ "$v" == "true" ]] && return 0 || return 1
}
cfg_list(){
    # Return all elements of a YAML list, one per line
    local len
    len=$(cfg "$1 | length")
    len=${len:-0}
    for ((i=0; i<len; i++)); do
        cfg_str "$1[$i]"
    done
}

# ---------- Read General Configuration ----------
DEVICE=$(cfg_str device)
DATA_PATH=$(cfg_str data_path)
LOG_DIR=$(cfg_str log_dir)
# Convert LOG_DIR to absolute path (relative to project root)
if [[ "$LOG_DIR" != /* ]]; then
    LOG_DIR="$PROJECT_ROOT/$LOG_DIR"
fi
IDX=$(cfg_int idx)
DATASET=$(cfg_str dataset)
MODEL=$(cfg_str model)

# ---------- Read Watermark Labels ----------
SOURCE_LABEL1=$(cfg source_label1)
# SOURCE_LABEL2=$(cfg source_label2)
TARGET_LABEL=$(cfg target_label)

echo "============================================================"
echo "  Tuning to Survive Pipeline"
echo "============================================================"
echo "  Config File : $CONFIG_FILE"
echo "  Device      : $DEVICE"
echo "  Dataset     : $DATASET"
echo "  Model       : $MODEL"
echo "  Index       : $IDX"
echo "  Watermark   : src1=$SOURCE_LABEL1, tgt=$TARGET_LABEL"
echo "============================================================"

# ============================================================
# Defense Phase
# ============================================================

# ---------- 1. Train Clean ----------
echo ""
echo ">>> [Defense] Train Clean Model"
TC_EPOCHS=$(cfg_int defense.train_clean.epochs)
TC_LR=$(cfg defense.train_clean.lr)
TC_BS=$(cfg_int defense.train_clean.batch_size)
TC_IMG=$(cfg_int defense.train_clean.image_size)

python "$SCRIPT_DIR/defense/train_clean.py" \
    --idx "$IDX" --dataset "$DATASET" --model "$MODEL" \
    --epochs "$TC_EPOCHS" --lr "$TC_LR" --batch_size "$TC_BS" \
    --image_size "$TC_IMG" --data_path "$DATA_PATH" --device "$DEVICE" \
    --log_dir "$LOG_DIR"

# ---------- 2. Get Trigger ----------
echo ""
echo ">>> [Defense] Generate Trigger"
GT_EPOCHS=$(cfg_int defense.get_trigger.epochs)
GT_LINF=$(cfg defense.get_trigger.l_inf_r)
GT_IMG=$(cfg_int defense.get_trigger.image_size)

python "$SCRIPT_DIR/defense/get_trigger.py" \
    --idx "$IDX" --dataset "$DATASET" --model "$MODEL" \
    --epochs "$GT_EPOCHS" --source_label "$SOURCE_LABEL1" \
    --image_size "$GT_IMG" --data_path "$DATA_PATH" --device "$DEVICE" \
    --log_dir "$LOG_DIR"

# ---------- 3. Watermarking ----------
echo ""
echo ">>> [Defense] Embed Watermark"
WM_MODE=$(cfg_str defense.watermarking.mode)
WM_EPOCHS=$(cfg_int defense.watermarking.epochs)
WM_LR=$(cfg defense.watermarking.lr)
WM_BS=$(cfg_int defense.watermarking.batch_size)
WM_NUM=$(cfg_int defense.watermarking.num)
WM_IMG=$(cfg_int defense.watermarking.image_size)

python "$SCRIPT_DIR/defense/watermarking.py" \
    --idx "$IDX" --dataset "$DATASET" --model "$MODEL" \
    --mode "$WM_MODE" --epochs "$WM_EPOCHS" --lr "$WM_LR" --batch_size "$WM_BS" \
    --source_label1 "$SOURCE_LABEL1" --target_label "$TARGET_LABEL" \
    --image_size "$WM_IMG" --data_path "$DATA_PATH" --device "$DEVICE" \
    --log_dir "$LOG_DIR"

# ---------- 4. T2S (Tuning to Survive) ----------
echo ""
echo ">>> [Defense] Tuning to Survive"
T2S_MODE=$(cfg_str defense.t2s.mode)
T2S_ALPHA=$(cfg_int defense.t2s.alpha)
T2S_LR_OUTER=$(cfg defense.t2s.lr_outer)
T2S_LR_INNER=$(cfg defense.t2s.lr_inner)
T2S_EPOCHS=$(cfg_int defense.t2s.epochs)
T2S_INNER_BS=$(cfg_int defense.t2s.inner_batch_size)
T2S_INNER_DISTILL_BS=$(cfg_int defense.t2s.inner_distill_batch_size)
T2S_OUTER_BS=$(cfg_int defense.t2s.outer_batch_size)
T2S_IMG=$(cfg_int defense.t2s.image_size)

python "$SCRIPT_DIR/defense/tuning.py" \
    --idx "$IDX" --dataset "$DATASET" --model "$MODEL" \
    --mode "$T2S_MODE" --alpha "$T2S_ALPHA" \
    --lr_outer "$T2S_LR_OUTER" --lr_inner "$T2S_LR_INNER" \
    --epochs "$T2S_EPOCHS" \
    --inner_batch_size "$T2S_INNER_BS" \
    --inner_distill_batch_size "$T2S_INNER_DISTILL_BS" \
    --outer_batch_size "$T2S_OUTER_BS" \
    --source_label1 "$SOURCE_LABEL1" --target_label "$TARGET_LABEL" \
    --image_size "$T2S_IMG" --data_path "$DATA_PATH" --device "$DEVICE" \
    --log_dir "$LOG_DIR"

# ============================================================
# Attack Phase
# ============================================================

EX_MODE=$(cfg_str attack.extraction.mode)
EX_IMG=$(cfg_int attack.extraction.image_size)
EX_BS=$(cfg_int attack.extraction.batch_size)
EX_EPOCHS=$(cfg_int attack.extraction.epochs)
EX_LR=$(cfg attack.extraction.lr)

# ---------- 5. Extraction - Soft Label ----------
if cfg_bool attack.extraction.soft_label; then
    echo ""
    echo ">>> [Attack] Model Extraction - Soft Label"
    python "$SCRIPT_DIR/attack/extraction.py" \
        --idx "$IDX" --target_dataset "$DATASET" --target_model "$MODEL" \
        --stolen_model "$MODEL" --sur_dataset "$DATASET" \
        --mode "$EX_MODE" --epochs "$EX_EPOCHS" --lr "$EX_LR" \
        --batch_size "$EX_BS" --image_size "$EX_IMG" \
        --source_label1 "$SOURCE_LABEL1" --target_label "$TARGET_LABEL" \
        --data_path "$DATA_PATH" --device "$DEVICE" --log_dir "$LOG_DIR"
fi

# ---------- 6. Extraction - Hard Label ----------
if cfg_bool attack.extraction.hard_label; then
    echo ""
    echo ">>> [Attack] Model Extraction - Hard Label"
    python "$SCRIPT_DIR/attack/extraction.py" \
        --idx "$IDX" --target_dataset "$DATASET" --target_model "$MODEL" \
        --stolen_model "$MODEL" --sur_dataset "$DATASET" \
        --mode "$EX_MODE" --hard_label \
        --epochs "$EX_EPOCHS" --lr "$EX_LR" \
        --batch_size "$EX_BS" --image_size "$EX_IMG" \
        --source_label1 "$SOURCE_LABEL1" --target_label "$TARGET_LABEL" \
        --data_path "$DATA_PATH" --device "$DEVICE" --log_dir "$LOG_DIR"
fi

# ---------- 7. Extraction - Different Surrogate Datasets ----------
SUR_DATASETS=$(cfg_list attack.extraction.sur_datasets)
if [[ -n "$SUR_DATASETS" ]]; then
    echo ""
    echo ">>> [Attack] Model Extraction - Different Surrogate Datasets"
    for sur_ds in $SUR_DATASETS; do
        echo "  --- sur_dataset=$sur_ds ---"
        python "$SCRIPT_DIR/attack/extraction.py" \
            --idx "$IDX" --target_dataset "$DATASET" --target_model "$MODEL" \
            --stolen_model "$MODEL" --sur_dataset "$sur_ds" \
            --mode "$EX_MODE" --epochs "$EX_EPOCHS" --lr "$EX_LR" \
            --batch_size "$EX_BS" --image_size "$EX_IMG" \
            --source_label1 "$SOURCE_LABEL1" --target_label "$TARGET_LABEL" \
            --data_path "$DATA_PATH" --device "$DEVICE" --log_dir "$LOG_DIR"
    done
fi

# ---------- 8. Extraction - Different Stolen Models ----------
STOLEN_MODELS=$(cfg_list attack.extraction.stolen_models)
if [[ -n "$STOLEN_MODELS" ]]; then
    echo ""
    echo ">>> [Attack] Model Extraction - Different Stolen Models"
    for stolen in $STOLEN_MODELS; do
        echo "  --- stolen_model=$stolen ---"
        python "$SCRIPT_DIR/attack/extraction.py" \
            --idx "$IDX" --target_dataset "$DATASET" --target_model "$MODEL" \
            --stolen_model "$stolen" --sur_dataset "$DATASET" \
            --mode "$EX_MODE" --epochs "$EX_EPOCHS" --lr "$EX_LR" \
            --batch_size "$EX_BS" --image_size "$EX_IMG" \
            --source_label1 "$SOURCE_LABEL1" --target_label "$TARGET_LABEL" \
            --data_path "$DATA_PATH" --device "$DEVICE" --log_dir "$LOG_DIR"
    done
fi

# ---------- 9. Pruning ----------
SP_START=$(cfg_int attack.pruning.sparsity_start)
SP_END=$(cfg_int attack.pruning.sparsity_end)
SP_STEP=$(cfg_int attack.pruning.sparsity_step)

echo ""
echo ">>> [Attack] Pruning Attack"
for ((j=SP_START; j<SP_END; j+=SP_STEP)); do
    sparsity=$(echo "scale=1; $j/10" | bc)
    echo "  --- sparsity=$sparsity ---"
    python "$SCRIPT_DIR/attack/pruning.py" \
        --idx "$IDX" --dataset "$DATASET" --model "$MODEL" \
        --sparsity "$sparsity" --device "$DEVICE" --log_dir "$LOG_DIR"
done

# ---------- 10. Quantization ----------
BITS_LIST=$(cfg_list attack.quantization.bits_list)

echo ""
echo ">>> [Attack] Quantization Attack"
for bits in $BITS_LIST; do
    echo "  --- bits=$bits ---"
    python "$SCRIPT_DIR/attack/quantization.py" \
        --idx "$IDX" --dataset "$DATASET" --model "$MODEL" \
        --bits "$bits" --device "$DEVICE" --log_dir "$LOG_DIR"
done

echo ""
echo "============================================================"
echo "  Pipeline Completed!"
echo "============================================================"
