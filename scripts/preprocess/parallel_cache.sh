#!/bin/bash

# Parallel JVS latent caching using 4 GPUs
# This script splits 100 JVS speakers across 4 GPUs for parallel processing
# Expected speedup: ~4x faster (4-6 hours -> 1-1.5 hours)

set -e  # Exit on error

echo "================================================================================"
echo "PARALLEL JVS LATENT CACHING (4 GPUs)"
echo "================================================================================"
echo "GPU configuration: Tesla T4 x 4"
echo "Batch size: 64 per GPU"
echo "Speakers per GPU: 25"
echo "Expected time: 1-1.5 hours"
echo "================================================================================"

# Configuration
BATCH_SIZE=64
CACHE_SCRIPT="scripts/preprocess/cache_jvs_latents.py"
LOG_DIR="logs/cache"

# Create log directory
mkdir -p "$LOG_DIR"

# Speaker groups (25 speakers per GPU)
SPEAKERS_GPU0="jvs001 jvs002 jvs003 jvs004 jvs005 jvs006 jvs007 jvs008 jvs009 jvs010 jvs011 jvs012 jvs013 jvs014 jvs015 jvs016 jvs017 jvs018 jvs019 jvs020 jvs021 jvs022 jvs023 jvs024 jvs025"
SPEAKERS_GPU1="jvs026 jvs027 jvs028 jvs029 jvs030 jvs031 jvs032 jvs033 jvs034 jvs035 jvs036 jvs037 jvs038 jvs039 jvs040 jvs041 jvs042 jvs043 jvs044 jvs045 jvs046 jvs047 jvs048 jvs049 jvs050"
SPEAKERS_GPU2="jvs051 jvs052 jvs053 jvs054 jvs055 jvs056 jvs057 jvs058 jvs059 jvs060 jvs061 jvs062 jvs063 jvs064 jvs065 jvs066 jvs067 jvs068 jvs069 jvs070 jvs071 jvs072 jvs073 jvs074 jvs075"
SPEAKERS_GPU3="jvs076 jvs077 jvs078 jvs079 jvs080 jvs081 jvs082 jvs083 jvs084 jvs085 jvs086 jvs087 jvs088 jvs089 jvs090 jvs091 jvs092 jvs093 jvs094 jvs095 jvs096 jvs097 jvs098 jvs099 jvs100"

echo ""
echo "[1/5] Starting GPU 0 process (jvs001-jvs025)..."
CUDA_VISIBLE_DEVICES=0 uv run python "$CACHE_SCRIPT" \
    --speakers $SPEAKERS_GPU0 \
    --batch-size $BATCH_SIZE \
    > "$LOG_DIR/gpu0.log" 2>&1 &
PID_GPU0=$!
echo "  GPU 0: PID $PID_GPU0, Log: $LOG_DIR/gpu0.log"

echo ""
echo "[2/5] Starting GPU 1 process (jvs026-jvs050)..."
CUDA_VISIBLE_DEVICES=1 uv run python "$CACHE_SCRIPT" \
    --speakers $SPEAKERS_GPU1 \
    --batch-size $BATCH_SIZE \
    > "$LOG_DIR/gpu1.log" 2>&1 &
PID_GPU1=$!
echo "  GPU 1: PID $PID_GPU1, Log: $LOG_DIR/gpu1.log"

echo ""
echo "[3/5] Starting GPU 2 process (jvs051-jvs075)..."
CUDA_VISIBLE_DEVICES=2 uv run python "$CACHE_SCRIPT" \
    --speakers $SPEAKERS_GPU2 \
    --batch-size $BATCH_SIZE \
    > "$LOG_DIR/gpu2.log" 2>&1 &
PID_GPU2=$!
echo "  GPU 2: PID $PID_GPU2, Log: $LOG_DIR/gpu2.log"

echo ""
echo "[4/5] Starting GPU 3 process (jvs076-jvs100)..."
CUDA_VISIBLE_DEVICES=3 uv run python "$CACHE_SCRIPT" \
    --speakers $SPEAKERS_GPU3 \
    --batch-size $BATCH_SIZE \
    > "$LOG_DIR/gpu3.log" 2>&1 &
PID_GPU3=$!
echo "  GPU 3: PID $PID_GPU3, Log: $LOG_DIR/gpu3.log"

echo ""
echo "================================================================================"
echo "[5/5] All 4 processes started successfully!"
echo "================================================================================"
echo "Process IDs:"
echo "  GPU 0: $PID_GPU0"
echo "  GPU 1: $PID_GPU1"
echo "  GPU 2: $PID_GPU2"
echo "  GPU 3: $PID_GPU3"
echo ""
echo "Monitor progress with:"
echo "  tail -f $LOG_DIR/gpu0.log  # GPU 0 progress"
echo "  tail -f $LOG_DIR/gpu1.log  # GPU 1 progress"
echo "  tail -f $LOG_DIR/gpu2.log  # GPU 2 progress"
echo "  tail -f $LOG_DIR/gpu3.log  # GPU 3 progress"
echo ""
echo "Or use: watch -n 5 'ps aux | grep cache_jvs_latents'"
echo ""
echo "Waiting for all processes to complete..."
echo "================================================================================"

# Wait for all background processes
wait $PID_GPU0
EXIT_CODE_GPU0=$?
echo "[DONE] GPU 0 process completed (exit code: $EXIT_CODE_GPU0)"

wait $PID_GPU1
EXIT_CODE_GPU1=$?
echo "[DONE] GPU 1 process completed (exit code: $EXIT_CODE_GPU1)"

wait $PID_GPU2
EXIT_CODE_GPU2=$?
echo "[DONE] GPU 2 process completed (exit code: $EXIT_CODE_GPU2)"

wait $PID_GPU3
EXIT_CODE_GPU3=$?
echo "[DONE] GPU 3 process completed (exit code: $EXIT_CODE_GPU3)"

echo ""
echo "================================================================================"
echo "PARALLEL CACHING COMPLETE"
echo "================================================================================"

# Check exit codes
if [ $EXIT_CODE_GPU0 -eq 0 ] && [ $EXIT_CODE_GPU1 -eq 0 ] && [ $EXIT_CODE_GPU2 -eq 0 ] && [ $EXIT_CODE_GPU3 -eq 0 ]; then
    echo "✓ All 4 GPUs completed successfully!"
    echo ""
    echo "Cache directory: data/jvs_ver1_latents/"
    echo "Total cached files: $(find data/jvs_ver1_latents -name '*.pt' | wc -l)"
    echo ""
    echo "Next step: Start training with"
    echo "  uv run --no-sync accelerate launch --multi-gpu scripts/train/teacher_japanese.py"
    exit 0
else
    echo "✗ Some processes failed:"
    [ $EXIT_CODE_GPU0 -ne 0 ] && echo "  GPU 0: Failed (exit code $EXIT_CODE_GPU0)"
    [ $EXIT_CODE_GPU1 -ne 0 ] && echo "  GPU 1: Failed (exit code $EXIT_CODE_GPU1)"
    [ $EXIT_CODE_GPU2 -ne 0 ] && echo "  GPU 2: Failed (exit code $EXIT_CODE_GPU2)"
    [ $EXIT_CODE_GPU3 -ne 0 ] && echo "  GPU 3: Failed (exit code $EXIT_CODE_GPU3)"
    echo ""
    echo "Check logs in $LOG_DIR/ for details"
    exit 1
fi
