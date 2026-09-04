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
"$PY" - <<'PYEND'
import csv, math, os
LBL = {"narrow": "CRNN-L (baseline)", "photo": "+ wide photometric",
       "elastic": "+ elastic", "morph": "+ morphological",
       "full": "AugCRNN-T (proposed)"}
print(f"{'Configuration':<24}{'WA (%)':>9}{'95% CI':>20}{'k/n':>16}")
print("-" * 70)
for m in ("narrow", "photo", "elastic", "morph", "full"):
    p = f"Model_abl_{m}/test_results_analysis.csv"
    if not os.path.exists(p):
        print(f"{LBL[m]:<24}{'-':>9}{'(kosulmadi)':>20}")
        continue
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    key = "correct" if "correct" in rows[0] else "Is_Correct"
    k = sum(1 for r in rows if str(r[key]).strip().lower() in ("1", "true"))
    n = len(rows); ph = k / n; z = 1.96
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    e = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    print(f"{LBL[m]:<24}{ph*100:>9.2f}{f'[{(c-e)*100:.2f}, {(c+e)*100:.2f}]':>20}{f'{k}/{n}':>16}")
PYEND
