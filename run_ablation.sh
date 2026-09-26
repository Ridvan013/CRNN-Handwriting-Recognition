#!/usr/bin/env bash
# Augmentation ablation trainings - local run
#
# Usage:
#   bash run_ablation.sh              # the two endpoint configurations (narrow, full)
#   bash run_ablation.sh main         # the same
#   bash run_ablation.sh components   # photo, elastic, morph
#   bash run_ablation.sh all          # all six, including the zero-augmentation anchor
#   bash run_ablation.sh full         # a single mode
#
# One mode took 0.9-1.5 hours on the RTX 4070 Laptop GPU of the paper.
# Logs are written under logs/. Train ALL configurations on the SAME machine
# (environment consistency).
set -u

# In Git Bash "python" may not be on PATH; try the candidates in turn.
if [ -z "${PY:-}" ]; then
  for c in python python3 py     "/c/Users/$USERNAME/AppData/Local/Programs/Python/Python313/python.exe"     "/c/Users/$USERNAME/AppData/Local/Programs/Python/Python312/python.exe"; do
    command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }
    [ -x "$c" ] && { PY="$c"; break; }
  done
fi
PY="${PY:-python}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-128}"
LR="${LR:-7e-4}"
PATIENCE="${PATIENCE:-15}"
GPU_AUG="${GPU_AUG:-1}"
ELASTIC_LEGACY="${ELASTIC_LEGACY:-0}"     # 0 (paper): alpha = RMS px   1: earlier no-op amplitude
ELASTIC_ALPHA="${ELASTIC_ALPHA:-1 3}"     # paper setting; the earlier code used "2 5" with ELASTIC_LEGACY=1

case "${1:-main}" in
  main)        MODES="narrow full" ;;
  components)  MODES="photo elastic morph" ;;
  all)         MODES="none narrow full photo elastic morph" ;;
  *)           MODES="$1" ;;
esac

# Python buffers its output when writing to a pipe; progress would not show through tee.
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

mkdir -p logs

echo "=============================================================="
echo " Split verification"
echo "=============================================================="
"$PY" verify_aachen_splits.py || { echo "VERIFICATION FAILED - training not started"; exit 1; }

echo
echo "Modes to run: $MODES"
echo "epochs=$EPOCHS batch=$BATCH lr=$LR patience=$PATIENCE"
echo

START_ALL=$(date +%s)
for m in $MODES; do
  DIR="Model_abl_$m"
  LOG="logs/abl_$m.log"
  echo "=============================================================="
  echo " --aug-mode $m   ->  $DIR"
  echo " log: $LOG"
  echo " start: $(date '+%H:%M:%S')"
  echo "=============================================================="
  T0=$(date +%s)
  "$PY" cloud/v3_augmented_train.py \
      --aug-mode "$m" \
      --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" --patience "$PATIENCE" \
      --gpu-aug "$GPU_AUG" --elastic-legacy-amplitude "$ELASTIC_LEGACY" --elastic-alpha $ELASTIC_ALPHA       --model-dir "$DIR" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  T1=$(date +%s)
  if [ "$RC" -ne 0 ]; then
    echo ">>> $m FAILED (exit $RC), moving on to the next mode"
  else
    echo ">>> $m finished - $(( (T1-T0)/60 )) minutes"
  fi
  echo
done

echo "=============================================================="
echo " Lexicon / trigram ablation"
echo "=============================================================="
if [ -f "Model_abl_full/best_model_wa.pth" ]; then
  "$PY" cloud/ablation_lexicon.py \
      --model Model_abl_full/best_model_wa.pth \
      --out results/ablation_lexicon.json 2>&1 | tee logs/ablation_lexicon.log
else
  echo "Model_abl_full/best_model_wa.pth missing - skipped (train the 'full' mode first)"
fi

echo
echo "=============================================================="
echo " SUMMARY   (total $(( ($(date +%s)-START_ALL)/60 )) minutes)"
echo "=============================================================="
"$PY" cloud/summarize_ablation.py
