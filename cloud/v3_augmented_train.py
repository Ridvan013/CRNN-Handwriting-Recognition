#!/usr/bin/env python3
"""
V3 Augmented — IAM Aachen Training from Scratch

V3 mimarisini sıfırdan IAM üzerinde eğitir, güçlendirilmiş augmentation ile:
  + Elastic deformation  (doğal kalem titremesi simülasyonu)
  + Morphological ops    (kalem kalınlığı varyasyonu)
  + Daha geniş brightness/contrast aralığı
  + Tüm orijinal V3 augmentasyonları korunur

Hyperparameters (V3 baseline greedy_aachen_v3.py ile aynı):
  optimizer  : AdamW(lr=7e-4, weight_decay=1e-5)
  scheduler  : cosine warmup 5 epoch + cosine decay
  epochs     : 100 (early stopping patience=15)
  batch_size : 128
  CNN freeze : yok (sıfırdan eğitim)
  AMP        : enabled

Output:
  Model_aachen_v3_augmented/best_model_wa.pth
  results/v3_augmented_results.json
"""

import argparse
import os
import sys
import json
import random
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "cloud"))
sys.path.insert(0, str(REPO_ROOT))

from model_v3 import (
    DEVICE, CRNNModel, CHAR_LIST, BLANK_TOKEN, PAD_TOKEN,
    IAMDataset, CRNNTrainer, CTCLoss, greedy_decode,
    custom_collate_fn, process_image_cpu_minimal, encode_to_labels,
    decode_labels, calculate_metrics, wilson_ci, mcnemar_test,
    _gpu_preprocess,
)
import torch
import numpy as np
import cv2
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch",        type=int,   default=128)
    p.add_argument("--lr",           type=float, default=7e-4)
    p.add_argument("--patience",     type=int,   default=15)
    p.add_argument("--model-dir",    type=str,   default="Model_aachen_v3_augmented")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--baseline-csv", type=str,   default="")
    p.add_argument("--iam-words",    type=str,   default="")
    p.add_argument("--iam-root",     type=str,   default="")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader isci sayisi. VARSAYILAN 0 (guvenli): her "
                        "isci veri kopyasi tasidigi icin Windows'ta bellek "
                        "yetmezse kosu sessizce olur. Bol RAM varsa 2 deneyin.")
    p.add_argument("--aug-mode",     type=str,   default="full",
                   choices=["full", "elastic", "morph", "photo", "narrow"],
                   help="Augmentation ablation mode; 'full' = proposed AugCRNN-T.")
    return p.parse_args()


# ─────────────────────── Enhanced augmentation ───────────────────────────────

def _elastic_deform(img_np: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    """Elastic deformation via Gaussian-smoothed random displacement field."""
    h, w = img_np.shape
    sigma_px = sigma * max(h, w)
    dx = cv2.GaussianBlur(
        (np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigmaX=sigma_px
    ) * alpha
    dy = cv2.GaussianBlur(
        (np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigmaX=sigma_px
    ) * alpha
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(x + dx, 0, w - 1)
    map_y = np.clip(y + dy, 0, h - 1)
    return cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR, borderValue=1.0)


# ── Ablation switch ──────────────────────────────────────────────────────────
# Controls which of the two proposed transforms are active. Set from the CLI
# via --aug-mode; "full" reproduces the published AugCRNN-T configuration.
#   full    : wide photometric + elastic + morphological   (proposed)
#   elastic : wide photometric + elastic
#   morph   : wide photometric + morphological
#   photo   : wide photometric only
#   narrow  : baseline photometric only (CRNN-L configuration)
AUG_MODE = "full"


def _aug_flags(mode: str):
    """Return (use_elastic, use_morph, use_wide_photometric) for a mode."""
    return {
        "full":    (True,  True,  True),
        "elastic": (True,  False, True),
        "morph":   (False, True,  True),
        "photo":   (False, False, True),
        "narrow":  (False, False, False),
    }[mode]


def _augment_v4(img: torch.Tensor) -> torch.Tensor:
    """
    Enhanced augmentation. img: float [1, H, W], bg=1.0 (white), text~0 (dark).
    The two proposed transforms can be switched off individually through the
    module-level AUG_MODE flag, which is what the ablation study in the paper
    varies; every other transform is identical across all modes.
    """
    use_elastic, use_morph, use_wide = _aug_flags(AUG_MODE)
    # Geometric (same as V3 baseline)
    if random.random() < 0.6:
        angle = random.uniform(-7, 7)
        img = TF.rotate(img.unsqueeze(0), angle,
                        interpolation=TF.InterpolationMode.BILINEAR, fill=1.0).squeeze(0)
    if random.random() < 0.6:
        h, w = img.shape[-2:]
        dx = random.uniform(-0.05, 0.05) * w
        dy = random.uniform(-0.05, 0.05) * h
        img = TF.affine(img.unsqueeze(0), angle=0, translate=[dx, dy],
                        scale=1.0, shear=[0.0, 0.0],
                        interpolation=TF.InterpolationMode.BILINEAR, fill=1.0).squeeze(0)
    if random.random() < 0.3:
        scale = random.uniform(0.9, 1.1)
        img = TF.affine(img.unsqueeze(0), angle=0, translate=[0, 0],
                        scale=scale, shear=[0.0, 0.0],
                        interpolation=TF.InterpolationMode.BILINEAR, fill=1.0).squeeze(0)
    if random.random() < 0.3:
        shear_x = random.uniform(-5, 5)
        img = TF.affine(img.unsqueeze(0), angle=0, translate=[0, 0],
                        scale=1.0, shear=[shear_x, 0.0],
                        interpolation=TF.InterpolationMode.BILINEAR, fill=1.0).squeeze(0)

    # NEW: elastic deformation (pen jitter / tremor)
    if use_elastic and random.random() < 0.5:
        img_np = img.squeeze(0).cpu().numpy()
        img_np = _elastic_deform(img_np, alpha=random.uniform(2.0, 5.0), sigma=0.08)
        img = torch.from_numpy(img_np).unsqueeze(0).to(img.device)

    # NEW: morphological ops (pen thickness variation)
    # bg=1 (white), text=0: erode expands text (thicker), dilate thins it
    if use_morph and random.random() < 0.3:
        img_np = img.squeeze(0).cpu().numpy()
        k = random.randint(1, 2)
        kernel = np.ones((k, k), np.uint8)
        if random.random() < 0.5:
            img_np = cv2.erode(img_np, kernel)
        else:
            img_np = cv2.dilate(img_np, kernel)
        img = torch.from_numpy(np.clip(img_np, 0.0, 1.0)).unsqueeze(0).to(img.device)

    # Photometric range: wide (0.70–1.35) for the proposed schedule,
    # narrow (0.85–1.15) for the CRNN-L baseline schedule.
    b_lo, b_hi = (0.70, 1.35) if use_wide else (0.85, 1.15)
    g_lo, g_hi = (0.70, 1.30) if use_wide else (0.80, 1.20)
    noise_sd = 0.05 if use_wide else 0.03

    if random.random() < 0.6:
        img = TF.adjust_brightness(img, random.uniform(b_lo, b_hi))
        img = TF.adjust_contrast(img, random.uniform(b_lo, b_hi))

    if random.random() < 0.4:
        img = torch.clamp(img + torch.randn_like(img) * noise_sd, 0.0, 1.0)

    if random.random() < 0.4:
        img = TF.adjust_gamma(img, random.uniform(g_lo, g_hi))

    # Random erasing (same as V3)
    if random.random() < 0.3:
        h, w = img.shape[-2:]
        ph = random.randint(max(1, h // 16), max(2, h // 6))
        pw = random.randint(max(1, w // 16), max(2, w // 6))
        y0 = random.randint(0, max(0, h - ph))
        x0 = random.randint(0, max(0, w - pw))
        img[..., y0:y0 + ph, x0:x0 + pw] = 1.0

    return img


class IAMDatasetV4(IAMDataset):
    """IAMDataset with enhanced _augment_v4 pipeline."""

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if img.ndim == 2:
            img = np.expand_dims(img, 0)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.transpose(img, (2, 0, 1))
        else:
            img = np.expand_dims(img, 0)

        t = torch.from_numpy(img.copy()).float().to(self.device, non_blocking=True)
        if t.max() > 1.0:
            t = t / 255.0

        if self.is_training:
            t = _augment_v4(t)
        t = _gpu_preprocess(t)

        return t, torch.tensor(label, dtype=torch.long)


# ─────────────────────── IAM data loading (phase3_finetune.py'den) ───────────

def _find_path(candidates: List[str]) -> str:
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_iam_aachen(repo_root: Path, iam_words_override: str = "", iam_root_override: str = ""):
    words_file = iam_words_override or _find_path([
        str(repo_root / "HTR_Using_CRNN/IAM/processed/archive/iam_words/words.txt"),
        str(repo_root / "words.txt"),
    ])
    img_root = iam_root_override or _find_path([
        str(repo_root / "HTR_Using_CRNN/IAM/processed/archive/iam_words/words"),
    ])

    if not img_root:
        raise FileNotFoundError("words/ image dizini bulunamadı.")

    aachen_dir = repo_root / "aachen_splits" / "splits"
    if not aachen_dir.exists():
        raise FileNotFoundError(f"Aachen split dizini bulunamadı: {aachen_dir}")

    def _load_forms(name):
        p = aachen_dir / f"{name}.uttlist"
        with open(p) as f:
            return set(line.strip() for line in f if line.strip())

    aachen_train = _load_forms("train")
    aachen_val   = _load_forms("validation")
    aachen_test  = _load_forms("test")
    print(f"  Aachen forms — train:{len(aachen_train)} val:{len(aachen_val)} test:{len(aachen_test)}")

    # ------------------------------------------------------------------
    # LABEL SOURCE
    # Preferred: the pre-built split files shipped in the repo
    #   aachen_splits/{train,validation,test}_words.txt
    # They were generated from the COMPLETE IAM words.txt (115,320 records)
    # by build_aachen_word_splits.py in STRICT mode (official 747/116/336
    # forms only). Reading them makes the run independent of whatever
    # words.txt happens to sit next to the images (Kaggle mirrors ship a
    # truncated 44,859-record file, which silently halves the dataset).
    # Fallback: scan words.txt directly (legacy behaviour).
    # ------------------------------------------------------------------
    split_dir = repo_root / "aachen_splits"
    split_files = {
        "train": split_dir / "train_words.txt",
        "val":   split_dir / "validation_words.txt",
        "test":  split_dir / "test_words.txt",
    }
    tagged = []   # (bucket, raw_line)
    if all(f.exists() for f in split_files.values()):
        for bucket, f in split_files.items():
            with open(f, encoding="utf-8") as fh:
                for l in fh:
                    l = l.strip()
                    if l and not l.startswith("#"):
                        tagged.append((bucket, l))
        print(f"  Labels    : aachen_splits/*_words.txt ({len(tagged):,} lines, strict Aachen)")
    else:
        print(f"  Labels    : {words_file} (fallback scan)")
        with open(words_file, encoding="utf-8") as f:
            for l in f:
                l = l.strip()
                if not l or l.startswith("#"):
                    continue
                form_id = "-".join(l.split()[0].split("-")[:2])
                if form_id in aachen_train:
                    tagged.append(("train", l))
                elif form_id in aachen_val:
                    tagged.append(("val", l))
                elif form_id in aachen_test:
                    tagged.append(("test", l))
                # forms outside the official Aachen partitions are SKIPPED:
                # their writers may overlap val/test writers.

    train_imgs, train_labs = [], []
    val_imgs,   val_labs   = [], []
    test_imgs,  test_labs  = [], []
    skipped = 0

    for bucket, line in tagged:
        parts = line.split()
        if len(parts) < 9:
            skipped += 1
            continue
        if parts[1] != "ok":
            skipped += 1
            continue

        word_id = parts[0]
        word    = "".join(parts[8:])

        a, b = word_id.split("-")[0], word_id.split("-")[1]
        img_path = os.path.join(img_root, a, f"{a}-{b}", f"{word_id}.png")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue

        try:
            img = process_image_cpu_minimal(img)
            lab = encode_to_labels(word)
        except Exception:
            skipped += 1
            continue

        if bucket == "train":
            train_imgs.append(img); train_labs.append(lab)
        elif bucket == "val":
            val_imgs.append(img);   val_labs.append(lab)
        else:
            test_imgs.append(img);  test_labs.append(lab)

    print(f"  Loaded — train:{len(train_imgs):,} val:{len(val_imgs):,} "
          f"test:{len(test_imgs):,}  skipped:{skipped:,}")
    return train_imgs, train_labs, val_imgs, val_labs, test_imgs, test_labs


# ─────────────────────── Test evaluation ─────────────────────────────────────

def evaluate_test_set(model, test_loader, trigram_lm, model_dir: Path) -> dict:
    model.eval()
    all_preds, all_targets = [], []
    ctc = CTCLoss()
    cached_T = None

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            B = images.size(0)
            ctx = torch.amp.autocast("cuda") if torch.cuda.is_available() else torch.no_grad()
            with ctx:
                lp = model(images)
                if cached_T is None:
                    cached_T = lp.size(0)
                input_lengths = torch.full((B,), cached_T, dtype=torch.long, device=DEVICE)

            preds = greedy_decode(lp, input_lengths)

            if trigram_lm:
                corrected = []
                for p in preds:
                    txt = decode_labels(p)
                    txt_c = trigram_lm.correct_word(txt)
                    corrected.append([CHAR_LIST.index(c) for c in txt_c if c in CHAR_LIST])
                all_preds.extend(corrected)
            else:
                all_preds.extend(preds)
            all_targets.extend(labels)

    cer, wa, wer = calculate_metrics(all_preds, all_targets)

    pred_texts = [decode_labels(p) for p in all_preds]
    true_texts = []
    for t in all_targets:
        if isinstance(t, torch.Tensor):
            true_texts.append("".join(CHAR_LIST[i] for i in t.tolist() if i != PAD_TOKEN))
        else:
            true_texts.append(decode_labels(t))
    correct_flags = [p.strip() == g.strip() for p, g in zip(pred_texts, true_texts)]

    n = len(correct_flags)
    ci_lo, ci_hi = wilson_ci(sum(correct_flags), n)

    import csv
    csv_path = model_dir / "test_results_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "ground_truth", "prediction", "correct"])
        for i, (p, g, c) in enumerate(zip(pred_texts, true_texts, correct_flags)):
            w.writerow([i, g, p, int(c)])

    return {
        "n_samples": n,
        "word_accuracy": wa,
        "word_accuracy_pct": round(wa * 100, 4),
        "cer": cer,
        "wer": wer,
        "wilson_95ci": [round(ci_lo * 100, 4), round(ci_hi * 100, 4)],
        "correct_flags": correct_flags,
        "csv_path": str(csv_path),
    }


def save_training_log(history: dict, model_dir: Path, results_dir: Path):
    """Her epoch için train/val metriklerini ve süreyi CSV'e yazar.

    Ablation'da her mod ayri model_dir'e yazar; results_dir'deki kopya her
    kosuda uzerine yazildigi icin asil kopya model_dir'dedir.
    """
    last = None
    for log_path in (model_dir / "training_log.csv",
                     results_dir / "training_log.csv"):
        last = _write_training_log(history, log_path)
    return last


def _write_training_log(history: dict, log_path):
    import csv
    n_epochs = len(history["train_loss"])
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch", "train_loss", "val_loss", "val_cer",
            "val_wa_pct", "lr", "epoch_time_min", "elapsed_total_min",
        ])
        elapsed = 0.0
        for i in range(n_epochs):
            t_min = history["epoch_time_s"][i] / 60.0
            elapsed += t_min
            w.writerow([
                i + 1,
                round(history["train_loss"][i], 6),
                round(history["val_loss"][i], 6),
                round(history["val_cer"][i], 6),
                round(history["val_wa"][i] * 100, 4),
                f"{history['lr'][i]:.6e}",
                round(t_min, 2),
                round(elapsed, 2),
            ])
    print(f"  Training log kaydedildi → {log_path}")
    return log_path


def evaluate_wbs(model, test_loader, iam_words_path: str) -> dict:
    """
    Word Beam Search değerlendirmesi.
    CTC decode sırasında sözlük kısıtlaması uygular — post-hoc trigram'dan çok daha etkili.
    word-beam-search paketi yoksa boş dict döner.
    """
    try:
        from word_beam_search import WordBeamSearch
    except ImportError:
        print("  ⚠️  word-beam-search kurulu değil, WBS atlandı")
        return {}

    # IAM "ok" kelimelerinden sözlük corpus'u oluştur
    ok_words = set()
    if iam_words_path and os.path.exists(iam_words_path):
        with open(iam_words_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 9 and parts[1] == "ok":
                    w = "".join(parts[8:])
                    if w and all(c in CHAR_LIST for c in w):
                        ok_words.add(w)

    if not ok_words:
        print("  ⚠️  WBS: corpus boş (iam_words_path geçerli değil?), atlandı")
        return {}

    corpus = " ".join(sorted(ok_words))
    # Blank token model'de index 84 (len(CHAR_LIST)).
    # word_beam_search son karakteri blank sayar → '|' ekliyoruz.
    chars_str = CHAR_LIST + "|"
    word_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    print(f"\n  WBS kurulumu: {len(ok_words):,} kelime, beam=25 ...")
    try:
        wbs = WordBeamSearch(
            25, "Words", 0.01,
            corpus.encode("utf8"),
            chars_str.encode("utf8"),
            word_chars.encode("utf8"),
        )
    except Exception as e:
        print(f"  ⚠️  WBS init hatası: {e}")
        return {}

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            ctx = torch.amp.autocast("cuda") if torch.cuda.is_available() else torch.no_grad()
            with ctx:
                lp = model(images)                          # (T, B, C+1)
            probs = torch.exp(lp).cpu().numpy()             # (T, B, C+1)
            B = probs.shape[1]
            for b in range(B):
                mat = probs[:, b, :]                        # (T, C+1)
                decoded = wbs.compute(mat)                  # list[int]
                all_preds.append(list(decoded))
            all_targets.extend(labels)

    cer, wa, _ = calculate_metrics(all_preds, all_targets)
    n = len(all_preds)
    ci_lo, ci_hi = wilson_ci(int(wa * n), n)

    print(f"  WBS Test WA : {wa*100:.4f}%")
    print(f"  WBS Test CER: {cer*100:.4f}%")
    return {
        "wa_pct": round(wa * 100, 4),
        "cer_pct": round(cer * 100, 4),
        "wilson_95ci_pct": [round(ci_lo * 100, 4), round(ci_hi * 100, 4)],
        "n_samples": n,
        "corpus_words": len(ok_words),
    }


def compare_with_baseline(result: dict, baseline_csv: str) -> dict:
    """McNemar testi. Baseline CSV'si okunamazsa sessizce atlanir.

    Sutun adi surumler arasinda degisiyor: yeni cikti "correct", eski
    Model_aachen_v3 ciktisi "Is_Correct" kullaniyor. Deger de "1"/"0" ya da
    "True"/"False" olabiliyor. Ikisini de kabul ediyoruz.
    """
    if not baseline_csv or not os.path.exists(baseline_csv):
        return {}
    import csv

    def _flag(v):
        return str(v).strip().lower() in ("1", "true", "yes")

    baseline_flags = []
    try:
        with open(baseline_csv, encoding="utf-8") as f:
            rd = csv.DictReader(f)
            cols = rd.fieldnames or []
            key = next((c for c in ("correct", "Is_Correct", "is_correct")
                        if c in cols), None)
            if key is None:
                print(f"  McNemar atlandi: baseline CSV'de 'correct' sutunu yok "
                      f"({baseline_csv})")
                return {}
            for row in rd:
                baseline_flags.append(_flag(row[key]))
    except Exception as exc:
        print(f"  McNemar atlandi: baseline okunamadi ({exc})")
        return {}

    new_flags = result["correct_flags"]
    if len(baseline_flags) != len(new_flags):
        print(f"  ⚠️  McNemar: sample sayısı eşleşmiyor ({len(baseline_flags)} vs {len(new_flags)})")
        return {}
    chi2, p = mcnemar_test(baseline_flags, new_flags)
    delta = result["word_accuracy"] - (sum(baseline_flags) / len(baseline_flags))
    return {
        "baseline_wa_pct": round(sum(baseline_flags) / len(baseline_flags) * 100, 4),
        "delta_pp": round(delta * 100, 4),
        "mcnemar_chi2": round(chi2, 4),
        "mcnemar_p": float(f"{p:.4e}"),
        "significant_p01": p < 0.01,
    }


# ─────────────────────── Main ────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    global AUG_MODE
    AUG_MODE = args.aug_mode
    _el, _mo, _wi = _aug_flags(AUG_MODE)
    print(f" Aug mode  : {AUG_MODE}  "
          f"(elastic={_el}, morphological={_mo}, wide photometric={_wi})")

    model_dir = REPO_ROOT / args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" V3 Augmented — IAM Aachen (scratch)")
    print(f" Device    : {DEVICE}")
    print(f" Epochs    : {args.epochs}  LR: {args.lr}  Batch: {args.batch}")
    print(f" Patience  : {args.patience}")
    print(f" Model dir : {model_dir}")
    print("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[1/4] Loading IAM Aachen dataset...")
    (train_imgs, train_labs,
     val_imgs,   val_labs,
     test_imgs,  test_labs) = load_iam_aachen(
        REPO_ROOT,
        iam_words_override=args.iam_words,
        iam_root_override=args.iam_root,
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    print("\n[2/4] Creating DataLoaders (IAMDatasetV4 with elastic + morph aug)...")
    # Dataset'i CPU'da tutuyoruz. Onceden device=DEVICE idi: her ornek TEK TEK
    # GPU'ya tasiniyor, elastic/morfoloji icin CPU'ya geri donuyor ve tekrar
    # GPU'ya cikiyordu. Bu, epoch suresini ~11 dakikaya cikariyordu.
    # CPU'da tutunca augmentation DataLoader iscileri arasinda paralellesiyor;
    # batch'i GPU'ya tasima isini egitim/degerlendirme donguleri zaten
    # kendileri yapiyor (model_v3.py: images.to(self.device)).
    _ds_device = "cpu" if args.num_workers > 0 else DEVICE
    train_ds = IAMDatasetV4(train_imgs, train_labs, is_training=True,  device=_ds_device)
    val_ds   = IAMDatasetV4(val_imgs,   val_labs,   is_training=False, device=_ds_device)
    test_ds  = IAMDatasetV4(test_imgs,  test_labs,  is_training=False, device=_ds_device)

    # Augmentation CPU'da yapiliyor (elastic deformation + morfoloji pahali).
    # num_workers=0 ile tek cekirdekte kaliyor ve epoch suresini birkac kat
    # uzatiyor; --num-workers ile ayarlanabilir.
    _nw = args.num_workers
    _pw = _nw > 0          # persistent_workers + pin_memory
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=_nw, pin_memory=_pw,
                              persistent_workers=False, drop_last=True,
                              collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=_nw, pin_memory=_pw,
                              persistent_workers=False, drop_last=False,
                              collate_fn=custom_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=_nw, pin_memory=_pw,
                              persistent_workers=False, drop_last=False,
                              collate_fn=custom_collate_fn)
    print(f"  train:{len(train_loader)} batches | val:{len(val_loader)} | test:{len(test_loader)}")

    # ── Trigram LM ────────────────────────────────────────────────────────────
    trigram_lm = None
    lm_src = _find_path([
        str(REPO_ROOT / "aachen_splits" / "train_words.txt"),
        args.iam_words,   # Kaggle'da zaten mevcut — en geniş vocab
        str(REPO_ROOT / "HTR_Using_CRNN" / "IAM" / "processed" / "archive" /
            "iam_words" / "words.txt"),
    ])
    if lm_src:
        try:
            from trigram_lm import TrigramLanguageModel
            print(f"\n  Trigram LM yükleniyor: {lm_src}")
            trigram_lm = TrigramLanguageModel(lm_src)
        except ImportError:
            print("  ⚠️  trigram_lm.py bulunamadı, trigram'sız devam")

    # ── Model (scratch) ───────────────────────────────────────────────────────
    print("\n[3/4] Building V3 model from scratch...")
    model = CRNNModel(img_height=32, img_width=128, num_classes=len(CHAR_LIST) + 1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params/1e6:.2f}M")

    trainer = CRNNTrainer(
        model,
        lr=args.lr,
        warmup_epochs=5,
        total_epochs=args.epochs,
        patience=args.patience,
        model_dir=str(model_dir),
        device=DEVICE,
        trigram_lm=trigram_lm,
    )

    history = trainer.train(train_loader, val_loader, epochs=args.epochs, cnn_freeze_epochs=0)

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    save_training_log(history, model_dir, results_dir)

    # ── Test evaluation (ONE-SHOT) ────────────────────────────────────────────
    print("\n[4/4] Evaluating on Aachen test set (ONE-SHOT)...")
    print("  KURALLAR: test set'e sadece bir kez bakılır.")

    best_ckpt = model_dir / "best_model_wa.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(str(best_ckpt), map_location=DEVICE))
        print(f"  Best WA checkpoint yüklendi: {best_ckpt}")

    test_result = evaluate_test_set(model, test_loader, trigram_lm, model_dir)

    # McNemar vs V3 baseline
    mcnemar_result = {}
    # Varsayilan baseline: ayni ortamda egitilmis CRNN-L (narrow). Eski
    # Model_aachen_v3 yarim veriyle egitildigi icin (5,338 vs 20,310 ornek)
    # McNemar zaten uzunluk uyusmazligindan atlanir.
    baseline_csv = args.baseline_csv
    if not baseline_csv:
        for cand in (REPO_ROOT / "Model_abl_narrow" / "test_results_analysis.csv",
                     REPO_ROOT / "Model_aachen_v3" / "test_results_analysis.csv"):
            if cand.exists():
                baseline_csv = str(cand)
                break
        baseline_csv = baseline_csv or ""
    if os.path.exists(baseline_csv):
        print(f"  McNemar baseline: {baseline_csv}")
        mcnemar_result = compare_with_baseline(test_result, baseline_csv)

    # WBS evaluation
    print("\n  Word Beam Search değerlendirmesi ...")
    wbs_result = evaluate_wbs(model, test_loader, args.iam_words)

    # Save results
    final = {
        "phase": "v3_augmented",
        "model": "V3_scratch_elastic_morph",
        "greedy_trigram_wa_pct": test_result["word_accuracy_pct"],
        "greedy_trigram_cer_pct": round(test_result["cer"] * 100, 4),
        "greedy_trigram_wilson_95ci_pct": test_result["wilson_95ci"],
        "wbs_wa_pct": wbs_result.get("wa_pct"),
        "wbs_cer_pct": wbs_result.get("cer_pct"),
        "wbs_wilson_95ci_pct": wbs_result.get("wilson_95ci_pct"),
        "n_samples": test_result["n_samples"],
        "mcnemar_vs_v3_base": mcnemar_result,
        "training_best_val_wa_pct": round(max(history["val_wa"]) * 100, 4),
    }
    # Geriye dönük uyumluluk için test_wa_pct en iyi sonucu gösterir
    best_wa = max(filter(None, [final["greedy_trigram_wa_pct"], final["wbs_wa_pct"]]))
    final["test_wa_pct"] = best_wa

    # Asil kopya model_dir'de; results_dir'deki her kosuda uzerine yazilir.
    for out_path in (model_dir / "results.json",
                     results_dir / "v3_augmented_results.json"):
        with open(out_path, "w") as f:
            json.dump(final, f, indent=2)

    print("\n" + "=" * 60)
    print(" SONUÇ")
    print("=" * 60)
    print(f" Greedy+Trigram WA: {final['greedy_trigram_wa_pct']:.2f}%")
    if wbs_result:
        print(f" WBS WA           : {final['wbs_wa_pct']:.2f}%")
    ci = final["greedy_trigram_wilson_95ci_pct"]
    print(f" Wilson 95% CI    : [{ci[0]:.2f}%, {ci[1]:.2f}%]")
    print(f" Best Val WA      : {final['training_best_val_wa_pct']:.2f}%")
    print(f" N samples        : {final['n_samples']:,}")
    if mcnemar_result:
        print(f"\n McNemar vs V3-base")
        print(f" Baseline WA      : {mcnemar_result['baseline_wa_pct']:.2f}%")
        print(f" Delta            : {mcnemar_result['delta_pp']:+.2f}pp")
        print(f" p-value          : {mcnemar_result['mcnemar_p']:.2e}")
    print(f"\n Results: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
