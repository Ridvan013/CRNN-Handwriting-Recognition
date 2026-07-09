#!/usr/bin/env bash
# H2 Cloud Pipeline — Phase 1 → 2 → 3
#
# Kullanım (repo root'undan, screen/tmux içinde):
#   screen -S h2
#   bash cloud/run_pipeline.sh
#   # Ctrl+A D ile detach; screen -r h2 ile geri dön
#
# Her faz kendi log dosyasına yazar:
#   logs/phase1.log
#   logs/phase2.log
#   logs/phase3.log
#
# Faz başarısız olursa pipeline durur (set -e).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "========================================"
echo " H2 Pipeline başlatıldı: $(timestamp)"
echo " Repo: $REPO_ROOT"
echo "========================================"

# ── Phase 1: Synthetic Data Generation ───────────────────────────────────────
echo ""
echo "[PHASE 1] Synthetic Data Generation — $(timestamp)"
echo "  Log: $LOG_DIR/phase1.log"
python cloud/phase1_synthetic_gen.py \
    --count 300000 \
    --output synthetic_data \
    --font-dir cloud/fonts \
    2>&1 | tee "$LOG_DIR/phase1.log"

echo ""
echo "[PHASE 1] TAMAMLANDI — $(timestamp)"

# ── Phase 2: Synthetic Pretrain ───────────────────────────────────────────────
echo ""
echo "[PHASE 2] Synthetic Pretraining — $(timestamp)"
echo "  Log: $LOG_DIR/phase2.log"
python cloud/phase2_pretrain.py \
    --epochs 8 \
    --batch 128 \
    --lr 1e-3 \
    --synthetic-dir synthetic_data \
    --ckpt-dir checkpoints \
    2>&1 | tee "$LOG_DIR/phase2.log"

echo ""
echo "[PHASE 2] TAMAMLANDI — $(timestamp)"

# Checkpoint var mı kontrol et
if [ ! -f "$REPO_ROOT/checkpoints/pretrain_best.pth" ]; then
    echo "ERROR: checkpoints/pretrain_best.pth bulunamadı. Phase 2 başarısız mı?"
    exit 1
fi

# ── Phase 3: IAM Aachen Fine-tuning ──────────────────────────────────────────
echo ""
echo "[PHASE 3] IAM Aachen Fine-tuning — $(timestamp)"
echo "  Log: $LOG_DIR/phase3.log"
python cloud/phase3_finetune.py \
    --epochs 50 \
    --batch 128 \
    --lr 1e-4 \
    --cnn-freeze 5 \
    --patience 15 \
    --ckpt-dir checkpoints \
    --model-dir Model_aachen_v3_pretrained \
    2>&1 | tee "$LOG_DIR/phase3.log"

echo ""
echo "[PHASE 3] TAMAMLANDI — $(timestamp)"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " TÜM FAZLAR TAMAMLANDI — $(timestamp)"
echo "========================================"

if [ -f "$REPO_ROOT/results/phase3_results.json" ]; then
    echo ""
    echo "--- SONUÇLAR ---"
    python -c "
import json, sys
with open('results/phase3_results.json') as f:
    r = json.load(f)
print(f'Test WA          : {r[\"test_wa_pct\"]:.2f}%')
ci = r['wilson_95ci_pct']
print(f'Wilson 95% CI    : [{ci[0]:.2f}%, {ci[1]:.2f}%]')
mn = r.get('mcnemar_vs_v3_base', {})
if mn:
    print(f'vs V3-base WA    : {mn.get(\"baseline_wa_pct\",\"?\"):.2f}%')
    print(f'Delta pp         : {mn.get(\"delta_pp\",\"?\"):+.2f}pp')
    print(f'McNemar p        : {mn.get(\"mcnemar_p\",\"?\"):.2e}')
"
    echo ""
    echo "Detaylı sonuçlar: results/phase3_results.json"
    echo "Checkpoint      : Model_aachen_v3_pretrained/best_model_wa.pth"
fi

echo ""
echo "Loglar: logs/phase1.log  logs/phase2.log  logs/phase3.log"
echo "========================================"
