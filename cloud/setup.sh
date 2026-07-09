#!/usr/bin/env bash
# H2 Cloud Setup Script — RunPod / Kaggle
# Çalıştırma: bash cloud/setup.sh [--gdrive-id <DRIVE_FILE_ID>]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "========================================"
echo " H2 Setup — CRNN Handwriting Recognition"
echo " Repo: $REPO_ROOT"
echo "========================================"

# ── 1. Python dependencies ──────────────────────────────────────────────────
echo ""
echo "[1/6] Installing Python packages..."
pip install -q -r cloud/requirements.txt
echo "  OK"

# ── 2. NLTK words corpus ─────────────────────────────────────────────────────
echo ""
echo "[2/6] Downloading NLTK 'words' corpus..."
python -c "import nltk; nltk.download('words', quiet=True)"
echo "  OK"

# ── 3. IAM dataset ───────────────────────────────────────────────────────────
echo ""
echo "[3/6] IAM dataset setup..."

IAM_DIR="$REPO_ROOT/HTR_Using_CRNN/IAM/processed/archive/iam_words"
WORDS_TXT="$IAM_DIR/words.txt"
WORDS_IMG="$IAM_DIR/words"

# Google Drive ID (arkadaşın paylaşacağı link)
# Kullanım: bash cloud/setup.sh --gdrive-id 1AbCdEf...
GDRIVE_ID="${GDRIVE_ID:-}"

for arg in "$@"; do
  case $arg in
    --gdrive-id)
      shift
      GDRIVE_ID="$1"
      shift
      ;;
    --gdrive-id=*)
      GDRIVE_ID="${arg#*=}"
      ;;
  esac
done

if [ -f "$WORDS_TXT" ] && [ -d "$WORDS_IMG" ]; then
  WORD_COUNT=$(find "$WORDS_IMG" -name "*.png" | wc -l)
  echo "  IAM dataset already present ($WORD_COUNT images). Skipping download."
elif [ -n "$GDRIVE_ID" ]; then
  echo "  Downloading IAM dataset from Google Drive (ID: $GDRIVE_ID)..."
  mkdir -p "$IAM_DIR"
  # gdown ile zip indir, çıkar
  gdown "https://drive.google.com/uc?id=${GDRIVE_ID}" -O /tmp/iam_dataset.zip
  echo "  Extracting..."
  unzip -q /tmp/iam_dataset.zip -d "$IAM_DIR"
  rm /tmp/iam_dataset.zip
  echo "  IAM dataset extracted to $IAM_DIR"
else
  echo "  ⚠️  IAM dataset bulunamadı."
  echo "     Arkadaşından Google Drive linkini al ve şu komutla tekrar çalıştır:"
  echo "     GDRIVE_ID=<FILE_ID> bash cloud/setup.sh"
  echo "     Veya: bash cloud/setup.sh --gdrive-id <FILE_ID>"
  echo "     (Phase 3 çalışması için gerekli — Phase 1-2 için gerekli değil)"
fi

# ── 4. Aachen split dosyaları kontrol ────────────────────────────────────────
echo ""
echo "[4/6] Checking Aachen split files..."
AACHEN_DIR="$REPO_ROOT/aachen_splits/splits"
if [ -d "$AACHEN_DIR" ]; then
  TRAIN_FORMS=$(wc -l < "$AACHEN_DIR/train.uttlist" 2>/dev/null || echo 0)
  echo "  Aachen splits OK (train forms: $TRAIN_FORMS)"
else
  echo "  ⚠️  Aachen splits not found at $AACHEN_DIR"
  echo "  Repo'nun feature/aachen-v3-extended-trigram branch'inden çektiğinden emin ol:"
  echo "  git checkout feature/aachen-v3-extended-trigram"
fi

# ── 5. Handwriting fontları indir ─────────────────────────────────────────────
echo ""
echo "[5/6] Downloading handwriting fonts..."
FONT_DIR="$REPO_ROOT/cloud/fonts"
mkdir -p "$FONT_DIR"

declare -A FONTS=(
  ["Caveat"]="https://github.com/google/fonts/raw/main/ofl/caveat/Caveat-Regular.ttf"
  ["IndieFlower"]="https://github.com/google/fonts/raw/main/ofl/indieflower/IndieFlower-Regular.ttf"
  ["Kalam"]="https://github.com/google/fonts/raw/main/ofl/kalam/Kalam-Regular.ttf"
  ["PatrickHand"]="https://github.com/google/fonts/raw/main/ofl/patrickhand/PatrickHand-Regular.ttf"
  ["ShadowsIntoLight"]="https://github.com/google/fonts/raw/main/ofl/shadowsintolight/ShadowsIntoLight.ttf"
  ["ArchitectsDaughter"]="https://github.com/google/fonts/raw/main/ofl/architectsdaughter/ArchitectsDaughter.ttf"
  ["DancingScript"]="https://github.com/google/fonts/raw/main/ofl/dancingscript/DancingScript-Regular.ttf"
  ["Pacifico"]="https://github.com/google/fonts/raw/main/ofl/pacifico/Pacifico-Regular.ttf"
  ["Satisfy"]="https://github.com/google/fonts/raw/main/ofl/satisfy/Satisfy-Regular.ttf"
  ["GloriaHallelujah"]="https://github.com/google/fonts/raw/main/ofl/gloriahallelujah/GloriaHallelujah.ttf"
  ["Handlee"]="https://github.com/google/fonts/raw/main/ofl/handlee/Handlee-Regular.ttf"
  ["JustAnotherHand"]="https://github.com/google/fonts/raw/main/ofl/justanotherhand/JustAnotherHand-Regular.ttf"
  ["Itim"]="https://github.com/google/fonts/raw/main/ofl/itim/Itim-Regular.ttf"
  ["PermanentMarker"]="https://github.com/google/fonts/raw/main/ofl/permanentmarker/PermanentMarker-Regular.ttf"
  ["Sacramento"]="https://github.com/google/fonts/raw/main/ofl/sacramento/Sacramento-Regular.ttf"
)

DOWNLOADED=0
for NAME in "${!FONTS[@]}"; do
  DEST="$FONT_DIR/${NAME}.ttf"
  if [ -f "$DEST" ]; then
    DOWNLOADED=$((DOWNLOADED + 1))
    continue
  fi
  if wget -q "${FONTS[$NAME]}" -O "$DEST" 2>/dev/null; then
    DOWNLOADED=$((DOWNLOADED + 1))
  else
    echo "  ⚠️  Font indirilmedi: $NAME (devam ediliyor)"
    rm -f "$DEST"
  fi
done
echo "  $DOWNLOADED / ${#FONTS[@]} font hazır → $FONT_DIR"

# ── 6. Dizin yapısı ───────────────────────────────────────────────────────────
echo ""
echo "[6/6] Creating output directories..."
mkdir -p "$REPO_ROOT/checkpoints"
mkdir -p "$REPO_ROOT/synthetic_data/words"
mkdir -p "$REPO_ROOT/results"
mkdir -p "$REPO_ROOT/logs"
echo "  checkpoints/, synthetic_data/, results/, logs/ OK"

echo ""
echo "========================================"
echo " Setup tamamlandı!"
echo " Devam etmek için:"
echo "   screen -S h2 && bash cloud/run_pipeline.sh"
echo "========================================"
