#!/usr/bin/env python3
"""
Phase 1 — Synthetic Data Generation

300 K word image üretir (trdg kullanarak).
Vocab: NLTK English words + IAM Aachen train transcriptions = ~240K unique word.
Output:
  synthetic_data/words/<NNNNNN>.png
  synthetic_data/labels.txt          (her satır: "<filename> <word>")

Çalıştırma (repo root'undan):
  python cloud/phase1_synthetic_gen.py [--count 300000] [--output synthetic_data]
"""

import argparse
import os
import re
import sys
import subprocess
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────── argparse ────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=300_000,
                   help="Üretilecek toplam sample sayısı (default: 300000)")
    p.add_argument("--output", type=str, default="synthetic_data",
                   help="Çıktı dizini (default: synthetic_data)")
    p.add_argument("--font-dir", type=str, default="cloud/fonts",
                   help="Font dizini (default: cloud/fonts)")
    p.add_argument("--batch-size", type=int, default=10_000,
                   help="trdg batch boyutu (default: 10000)")
    return p.parse_args()


# ─────────────────────── Vocab ───────────────────────────────────────────────

CHAR_LIST = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def _valid_word(w: str) -> bool:
    """trdg ile render edilebilir mi?"""
    if not w or len(w) > 25 or len(w) < 2:
        return False
    return all(c in CHAR_LIST for c in w)


def build_vocab(output_dir: Path) -> Path:
    vocab_path = output_dir / "synthetic_vocab.txt"
    if vocab_path.exists():
        n = sum(1 for _ in open(vocab_path))
        print(f"  Vocab dosyası mevcut: {n:,} kelime ({vocab_path})")
        return vocab_path

    words = set()

    # 1. NLTK English words
    try:
        import nltk
        try:
            from nltk.corpus import words as nltk_words
            words.update(nltk_words.words())
        except LookupError:
            nltk.download("words", quiet=True)
            from nltk.corpus import words as nltk_words
            words.update(nltk_words.words())
        print(f"  NLTK words: {len(words):,}")
    except Exception as e:
        print(f"  ⚠️  NLTK yüklenemedi: {e}")

    # 2. IAM Aachen train transcriptions
    aachen_train_words = REPO_ROOT / "aachen_splits" / "train_words.txt"
    if aachen_train_words.exists():
        with open(aachen_train_words, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # format: "word_id ... word" — son token word
                    parts = line.split()
                    if len(parts) >= 9:
                        words.add(parts[-1])
                    elif len(parts) == 1:
                        words.add(parts[0])
        print(f"  After IAM Aachen train: {len(words):,}")
    else:
        # Fallback: words.txt'den de çekebiliriz
        iam_words_txt = (REPO_ROOT / "HTR_Using_CRNN" / "IAM" / "processed" /
                         "archive" / "iam_words" / "words.txt")
        if iam_words_txt.exists():
            with open(iam_words_txt, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 9 and parts[1] == "ok":
                        words.add("".join(parts[8:]))
            print(f"  After IAM words.txt: {len(words):,}")

    # Filter
    valid = sorted(w for w in words if _valid_word(w))
    print(f"  Filtered vocab: {len(valid):,}")

    with open(vocab_path, "w", encoding="utf-8") as f:
        for w in valid:
            f.write(w + "\n")
    print(f"  Vocab kaydedildi: {vocab_path}")
    return vocab_path


# ─────────────────────── trdg runner ─────────────────────────────────────────

def run_trdg_batch(vocab_file: Path, font_dir: Path, output_dir: Path,
                   count: int, offset: int) -> int:
    """
    trdg'yi subprocess olarak çalıştırır.
    Returns: number of files actually generated.
    """
    cmd = [
        sys.executable, "-m", "trdg.run",
        "-i", str(vocab_file),
        "-c", str(count),
        "--output_dir", str(output_dir),
        "-fd", str(font_dir),
        "-k", "5",         # max skew (±5°)
        "-rk",             # random skew
        "-bl", "1",        # blur radius
        "-rbl",            # random blur
        "-do", "1",        # distortion
        "-b", "0",         # white background
        "-na", "2",        # name format: word_count
        "-f", "32",        # font size ~32 → image height ~32px
        "-wd", "128",      # width 128px
        "-tc", "#000000",  # text color black
        "--margins", "2,2,2,2",
    ]

    if font_dir.exists() and any(font_dir.glob("*.ttf")):
        pass  # font_dir zaten set edildi
    else:
        print(f"  ⚠️  Font dizini boş: {font_dir} — trdg default fontları kullanacak")
        cmd = [c for c in cmd if c not in ["-fd", str(font_dir)]]

    print(f"  trdg çalışıyor: count={count}, offset_id={offset}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            print(f"  ⚠️  trdg hata kodu {result.returncode}")
            print(result.stderr[:500])
    except subprocess.TimeoutExpired:
        print("  ⚠️  trdg timeout (2 saat)")
    except Exception as e:
        print(f"  ⚠️  trdg başlatılamadı: {e}")
        return 0

    return count


# ─────────────────────── Labels dosyası ──────────────────────────────────────

def build_labels(img_dir: Path, labels_path: Path) -> int:
    """
    img_dir içindeki PNG dosyalarından labels.txt oluşturur.
    trdg dosya ismi formatı: word_N.png → label = word.
    """
    pattern = re.compile(r"^(.+?)_\d+\.png$", re.IGNORECASE)
    lines = []
    for f in sorted(img_dir.glob("*.png")):
        m = pattern.match(f.name)
        if m:
            word = m.group(1)
            if _valid_word(word):
                lines.append(f"{f.name} {word}\n")

    with open(labels_path, "w", encoding="utf-8") as out:
        out.writelines(lines)
    return len(lines)


# ─────────────────────── Preview ─────────────────────────────────────────────

def preview_samples(img_dir: Path, labels_path: Path, n: int = 30):
    """30 rastgele sample'ı preview_grid.png olarak kaydeder."""
    try:
        import random
        import cv2
        import numpy as np

        samples = []
        with open(labels_path) as f:
            lines = f.readlines()
        random.shuffle(lines)
        for line in lines[:n]:
            fname, word = line.strip().split(" ", 1)
            img = cv2.imread(str(img_dir / fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                samples.append((img, word))
            if len(samples) >= n:
                break

        if not samples:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(5, 6, figsize=(18, 8))
        for i, ax in enumerate(axes.flat):
            if i < len(samples):
                ax.imshow(samples[i][0], cmap="gray")
                ax.set_title(samples[i][1], fontsize=8)
            ax.axis("off")
        plt.tight_layout()
        out_path = img_dir.parent / "preview_grid.png"
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f"  Preview kaydedildi: {out_path}")
    except Exception as e:
        print(f"  Preview oluşturulamadı: {e}")


# ─────────────────────── Main ────────────────────────────────────────────────

def main():
    args = parse_args()

    output_dir = REPO_ROOT / args.output
    img_dir = output_dir / "words"
    labels_path = output_dir / "labels.txt"
    font_dir = REPO_ROOT / args.font_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f" Phase 1 — Synthetic Data Generation")
    print(f" Target: {args.count:,} images → {img_dir}")
    print("=" * 60)

    # ── Vocab ─────────────────────────────────────────────────────────────────
    print("\n[1/4] Building vocabulary...")
    vocab_file = build_vocab(output_dir)

    # ── trdg ──────────────────────────────────────────────────────────────────
    existing = list(img_dir.glob("*.png"))
    existing_count = len(existing)
    print(f"\n[2/4] Generating images with trdg...")
    print(f"  Already generated: {existing_count:,}")

    remaining = args.count - existing_count
    if remaining <= 0:
        print(f"  Target already met ({existing_count:,} >= {args.count:,}). Skipping.")
    else:
        print(f"  Need {remaining:,} more images.")
        # trdg batch'ler halinde çalıştır
        batch_sz = args.batch_size
        offset = existing_count
        while remaining > 0:
            n = min(batch_sz, remaining)
            run_trdg_batch(vocab_file, font_dir, img_dir, n, offset)
            offset += n
            remaining -= n
            current = len(list(img_dir.glob("*.png")))
            print(f"  Progress: {current:,} / {args.count:,}")

    # ── Labels ────────────────────────────────────────────────────────────────
    print(f"\n[3/4] Building labels.txt...")
    n_labels = build_labels(img_dir, labels_path)
    print(f"  {n_labels:,} samples labeled → {labels_path}")

    # ── Verification ──────────────────────────────────────────────────────────
    print(f"\n[4/4] Verification...")
    total_imgs = len(list(img_dir.glob("*.png")))
    print(f"  Total PNG images : {total_imgs:,}")
    print(f"  Labels file rows : {n_labels:,}")

    if total_imgs < args.count * 0.95:
        print(f"  ⚠️  WARNING: {total_imgs:,} < {args.count:,} * 0.95 — eksik görüntü var")
    else:
        print(f"  ✓ Image count OK")

    preview_samples(img_dir, labels_path, n=30)

    print(f"\n{'='*60}")
    print(f" Phase 1 TAMAMLANDI")
    print(f" {total_imgs:,} synthetic image → {img_dir}")
    print(f" Labels → {labels_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
