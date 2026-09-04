#!/usr/bin/env bash
# Ablation egitimleri - yerel calistirma
#
# Kullanim:
#   bash run_ablation.sh              # ana iki konfigurasyon (narrow, full)
#   bash run_ablation.sh main         # ayni sey
#   bash run_ablation.sh components   # photo, elastic, morph
#   bash run_ablation.sh all          # besi birden
#   bash run_ablation.sh full         # tek bir mod
#
# Her mod ~3-4 saat surer. Loglar logs/ altina yazilir.
# BUTUN konfigurasyonlar AYNI makinede calistirilmali (ortam tutarliligi).
set -u

# Git Bash'te "python" PATH'te olmayabilir; sirayla dene.
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

case "${1:-main}" in
  main)        MODES="narrow full" ;;
  components)  MODES="photo elastic morph" ;;
  all)         MODES="narrow full photo elastic morph" ;;
  *)           MODES="$1" ;;
esac

mkdir -p logs

echo "=============================================================="
echo " Split dogrulamasi"
echo "=============================================================="
"$PY" verify_aachen_splits.py || { echo "DOGRULAMA BASARISIZ - egitim baslatilmadi"; exit 1; }

echo
echo "Calistirilacak modlar: $MODES"
echo "epochs=$EPOCHS batch=$BATCH lr=$LR patience=$PATIENCE"
echo

START_ALL=$(date +%s)
for m in $MODES; do
  DIR="Model_abl_$m"
  LOG="logs/abl_$m.log"
  echo "=============================================================="
  echo " --aug-mode $m   ->  $DIR"
  echo " log: $LOG"
  echo " baslangic: $(date '+%H:%M:%S')"
  echo "=============================================================="
  T0=$(date +%s)
  "$PY" cloud/v3_augmented_train.py \
      --aug-mode "$m" \
      --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" --patience "$PATIENCE" \
      --model-dir "$DIR" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  T1=$(date +%s)
  if [ "$RC" -ne 0 ]; then
    echo ">>> $m BASARISIZ (cikis $RC), sonraki moda geciliyor"
  else
    echo ">>> $m tamamlandi - $(( (T1-T0)/60 )) dakika"
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
  echo "Model_abl_full/best_model_wa.pth yok - atlandi (once 'full' modunu egit)"
fi

echo
echo "=============================================================="
echo " OZET   (toplam $(( ($(date +%s)-START_ALL)/60 )) dakika)"
echo "=============================================================="
"$PY" cloud/summarize_ablation.py
