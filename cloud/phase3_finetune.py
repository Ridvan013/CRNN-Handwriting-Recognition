#!/usr/bin/env python3
"""
Phase 3 — IAM Aachen Fine-tuning

Phase 2'nin pretrained checkpoint'inden başlayarak IAM Aachen train set'i
üzerinde fine-tune eder. İlk 5 epoch CNN freeze (catastrophic forgetting önlemi).

Hyperparameters (H2_PLAN.md):
  optimizer     : AdamW(lr=1e-4, weight_decay=1e-5)   ← pretrain'den 10x düşük
  scheduler     : cosine warmup 3 epoch + cosine decay
  epochs        : 40-50 (early stopping patience=15)
  batch_size    : 128
  CNN freeze    : ilk 5 epoch
  AMP           : enabled
  trigram       : V3 trigram (val WA için)

Çalıştırma (repo root'undan):
  python cloud/phase3_finetune.py [--epochs 50] [--batch 128]

Output:
  checkpoints/finetune_best.pth
  results/phase3_results.json     ← Test WA, Wilson CI, McNemar p
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
sys.path.insert(0, str(REPO_ROOT))   # trigram_lm import için

from model_v3 import (
    DEVICE, CRNNModel, CHAR_LIST, BLANK_TOKEN, PAD_TOKEN,
    IAMDataset, CRNNTrainer, CTCLoss, greedy_decode,
    custom_collate_fn, process_image_cpu_minimal, encode_to_labels,
    decode_labels, calculate_metrics, wilson_ci, mcnemar_test,
)
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch",        type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--cnn-freeze",   type=int,   default=5,
                   help="CNN freeze epoch sayısı (default: 5)")
    p.add_argument("--patience",     type=int,   default=15)
    p.add_argument("--ckpt-dir",     type=str,   default="checkpoints")
    p.add_argument("--model-dir",    type=str,   default="Model_aachen_v3_pretrained")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--baseline-csv", type=str,   default="",
                   help="McNemar karşılaştırması için V3-base test_results_analysis.csv")
    # Kaggle / custom path overrides
    p.add_argument("--iam-words",    type=str,   default="",
                   help="words.txt path override (Kaggle: /kaggle/input/<slug>/words.txt)")
    p.add_argument("--iam-root",     type=str,   default="",
                   help="words/ image dir override (Kaggle: /kaggle/input/<slug>/words)")
    return p.parse_args()


# ─────────────────────── IAM data loading ────────────────────────────────────

def _find_path(candidates: List[str]) -> str:
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_iam_aachen(repo_root: Path, iam_words_override: str = "", iam_root_override: str = ""):
    """
    greedy_aachen_v3.py ile aynı data loading mantığı.
    iam_words_override / iam_root_override: Kaggle path'leri için kullan.
    Returns: (train_imgs, train_labs, val_imgs, val_labs, test_imgs, test_labs)
    """
    words_file = iam_words_override or _find_path([
        str(repo_root / "HTR_Using_CRNN/IAM/processed/archive/iam_words/words.txt"),
        str(repo_root / "words.txt"),
    ])
    img_root = iam_root_override or _find_path([
        str(repo_root / "HTR_Using_CRNN/IAM/processed/archive/iam_words/words"),
    ])

    if not words_file:
        raise FileNotFoundError(
            "words.txt bulunamadı.\n"
            "  setup.sh'i çalıştır: GDRIVE_ID=<ID> bash cloud/setup.sh"
        )
    if not img_root:
        raise FileNotFoundError(
            "words/ image dizini bulunamadı.\n"
            "  IAM dataset tam çıkarıldı mı kontrol et."
        )

    # Aachen split
    aachen_dir = repo_root / "aachen_splits" / "splits"
    if not aachen_dir.exists():
        raise FileNotFoundError(
            f"Aachen split dizini bulunamadı: {aachen_dir}\n"
            "  git checkout feature/aachen-v3-extended-trigram"
        )

    def _load_forms(name):
        p = aachen_dir / f"{name}.uttlist"
        with open(p) as f:
            return set(line.strip() for line in f if line.strip())

    aachen_train = _load_forms("train")
    aachen_val   = _load_forms("validation")
    aachen_test  = _load_forms("test")
    print(f"  Aachen forms — train:{len(aachen_train)} val:{len(aachen_val)} test:{len(aachen_test)}")

    with open(words_file, encoding="utf-8") as f:
        lines = [l.strip() for l in f]

    train_imgs, train_labs = [], []
    val_imgs,   val_labs   = [], []
    test_imgs,  test_labs  = [], []
    skipped = 0

    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 9:
            skipped += 1
            continue
        if parts[1] != "ok":
            skipped += 1
            continue

        word_id = parts[0]
        word    = "".join(parts[8:])

        form_id = "-".join(word_id.split("-")[:2])
        if not form_id or "-" not in form_id or form_id.startswith("user"):
            skipped += 1
            continue

        if form_id in aachen_train:
            bucket = "train"
        elif form_id in aachen_val:
            bucket = "val"
        elif form_id in aachen_test:
            bucket = "test"
        else:
            bucket = "train"  # unassigned → train (+%22 ekstra)

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
    """
    Test set üzerinde full evaluation.
    Trigram correction + greedy decode.
    Returns dict with WA, CER, Wilson CI, correct_flags.
    KURALLAR: test setine sadece bir kez bakılır, ONE-SHOT.
    """
    model.eval()
    all_preds, all_targets = [], []
    correct_flags = []
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

    # Metrics
    cer, wa, wer = calculate_metrics(all_preds, all_targets)

    # Per-sample correctness (for McNemar)
    pred_texts = [decode_labels(p) for p in all_preds]
    true_texts = []
    for t in all_targets:
        if isinstance(t, torch.Tensor):
            true_texts.append("".join(CHAR_LIST[i] for i in t.tolist() if i != PAD_TOKEN))
        else:
            true_texts.append(decode_labels(t))
    correct_flags = [p.strip() == g.strip() for p, g in zip(pred_texts, true_texts)]

    n = len(correct_flags)
    successes = sum(correct_flags)
    ci_lo, ci_hi = wilson_ci(successes, n)

    result = {
        "n_samples": n,
        "word_accuracy": wa,
        "word_accuracy_pct": round(wa * 100, 4),
        "cer": cer,
        "wer": wer,
        "wilson_95ci": [round(ci_lo * 100, 4), round(ci_hi * 100, 4)],
        "correct_flags": correct_flags,
    }

    # Save predictions CSV
    csv_path = model_dir / "test_results_analysis.csv"
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "ground_truth", "prediction", "correct"])
        for i, (p, g, c) in enumerate(zip(pred_texts, true_texts, correct_flags)):
            w.writerow([i, g, p, int(c)])
    result["csv_path"] = str(csv_path)
    return result


# ─────────────────────── McNemar vs baseline ─────────────────────────────────

def compare_with_baseline(result: dict, baseline_csv: str) -> dict:
    """V3-base CSV ile paired McNemar test."""
    if not baseline_csv or not os.path.exists(baseline_csv):
        return {}

    import csv
    baseline_flags = []
    with open(baseline_csv, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            baseline_flags.append(bool(int(row["correct"])))

    new_flags = result["correct_flags"]
    if len(baseline_flags) != len(new_flags):
        print(f"  ⚠️  McNemar: sample sayısı eşleşmiyor "
              f"({len(baseline_flags)} vs {len(new_flags)})")
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

    ckpt_dir  = REPO_ROOT / args.ckpt_dir
    model_dir = REPO_ROOT / args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" Phase 3 — IAM Aachen Fine-tuning")
    print(f" Device      : {DEVICE}")
    print(f" Epochs      : {args.epochs}")
    print(f" Batch       : {args.batch}")
    print(f" LR (peak)   : {args.lr}")
    print(f" CNN freeze  : first {args.cnn_freeze} epochs")
    print(f" Patience    : {args.patience}")
    print(f" Model dir   : {model_dir}")
    print("=" * 60)

    # ── Load IAM data ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading IAM Aachen dataset...")
    (train_imgs, train_labs,
     val_imgs,   val_labs,
     test_imgs,  test_labs) = load_iam_aachen(
        REPO_ROOT,
        iam_words_override=args.iam_words,
        iam_root_override=args.iam_root,
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    print("\n[2/5] Creating DataLoaders...")
    train_ds = IAMDataset(train_imgs, train_labs, is_training=True,  device=DEVICE)
    val_ds   = IAMDataset(val_imgs,   val_labs,   is_training=False, device=DEVICE)
    test_ds  = IAMDataset(test_imgs,  test_labs,  is_training=False, device=DEVICE)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True,
                              collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0, pin_memory=False, drop_last=False,
                              collate_fn=custom_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=0, pin_memory=False, drop_last=False,
                              collate_fn=custom_collate_fn)
    print(f"  train:{len(train_loader)} batches | val:{len(val_loader)} | test:{len(test_loader)}")

    # ── Trigram LM ────────────────────────────────────────────────────────────
    trigram_lm = None
    lm_src_candidates = [
        str(REPO_ROOT / "aachen_splits" / "train_words.txt"),
        str(REPO_ROOT / "HTR_Using_CRNN" / "IAM" / "processed" / "archive" /
            "iam_words" / "words.txt"),
    ]
    lm_src = _find_path(lm_src_candidates)
    if lm_src:
        try:
            from trigram_lm import TrigramLanguageModel
            print(f"\n  Trigram LM yükleniyor: {lm_src}")
            trigram_lm = TrigramLanguageModel(lm_src)
        except ImportError:
            print("  ⚠️  trigram_lm.py bulunamadı, validation trigram'sız")

    # ── Model + load pretrain checkpoint ─────────────────────────────────────
    print("\n[3/5] Building model and loading pretrain checkpoint...")
    model = CRNNModel(img_height=32, img_width=128, num_classes=len(CHAR_LIST) + 1)

    pretrain_path = ckpt_dir / "pretrain_best.pth"
    if pretrain_path.exists():
        state = torch.load(str(pretrain_path), map_location=DEVICE)
        model.load_state_dict(state)
        print(f"  Pretrain checkpoint yüklendi: {pretrain_path}")
    else:
        print(f"  ⚠️  pretrain_best.pth bulunamadı: {pretrain_path}")
        print("  Phase 2'yi önce çalıştır. Sıfırdan başlanacak.")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[4/5] Fine-tuning...")
    trainer = CRNNTrainer(
        model,
        lr=args.lr,
        warmup_epochs=3,
        total_epochs=args.epochs,
        patience=args.patience,
        model_dir=str(model_dir),
        device=DEVICE,
        trigram_lm=trigram_lm,
    )

    history = trainer.train(
        train_loader, val_loader,
        epochs=args.epochs,
        cnn_freeze_epochs=args.cnn_freeze,
    )

    # ── Final: load best WA checkpoint and evaluate test set ─────────────────
    print("\n[5/5] Evaluating on Aachen test set (ONE-SHOT)...")
    print("  KURALLAR: test set'e sadece bir kez bakılır.")

    best_ckpt = model_dir / "best_model_wa.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(str(best_ckpt), map_location=DEVICE))
        print(f"  Best WA checkpoint yüklendi: {best_ckpt}")
    else:
        print("  ⚠️  best_model_wa.pth yok, son epoch modeli kullanılıyor")

    test_result = evaluate_test_set(model, test_loader, trigram_lm, model_dir)

    # McNemar vs V3-base
    mcnemar_result = {}
    if args.baseline_csv:
        mcnemar_result = compare_with_baseline(test_result, args.baseline_csv)
    else:
        # Kanonik path'i otomatik dene
        default_baseline = REPO_ROOT / "Model_aachen_v3" / "test_results_analysis.csv"
        if default_baseline.exists():
            print(f"  McNemar baseline: {default_baseline}")
            mcnemar_result = compare_with_baseline(test_result, str(default_baseline))

    # Save results
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    final = {
        "phase": "phase3_finetune",
        "model": "V3_pretrained",
        "test_wa_pct": test_result["word_accuracy_pct"],
        "test_cer_pct": round(test_result["cer"] * 100, 4),
        "wilson_95ci_pct": test_result["wilson_95ci"],
        "n_samples": test_result["n_samples"],
        "mcnemar_vs_v3_base": mcnemar_result,
        "training_best_val_wa_pct": round(max(history["val_wa"]) * 100, 4),
    }

    results_path = results_dir / "phase3_results.json"
    with open(results_path, "w") as f:
        json.dump(final, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" Phase 3 TAMAMLANDI")
    print(f" Test WA             : {test_result['word_accuracy_pct']:.2f}%")
    print(f" Test CER            : {test_result['cer']*100:.2f}%")
    ci = test_result["wilson_95ci"]
    print(f" Wilson 95% CI       : [{ci[0]:.2f}%, {ci[1]:.2f}%]")
    print(f" N samples           : {test_result['n_samples']:,}")
    if mcnemar_result:
        print(f" vs V3-base WA       : {mcnemar_result.get('baseline_wa_pct', '?'):.2f}%")
        print(f" Delta pp            : {mcnemar_result.get('delta_pp', '?'):.2f}pp")
        print(f" McNemar p           : {mcnemar_result.get('mcnemar_p', '?'):.2e}")
        sig = mcnemar_result.get("significant_p01", False)
        print(f" Significant (p<.01) : {'YES ✓' if sig else 'NO'}")
    print(f" Results JSON        : {results_path}")
    print(f"{'='*60}")

    # ── Arkadaşa gönderilecek özet ────────────────────────────────────────────
    print(f"\n--- Arkadaşına gönderilecek sayılar ---")
    print(f"Test WA  : {test_result['word_accuracy_pct']:.2f}%")
    print(f"Wilson CI: [{ci[0]:.2f}%, {ci[1]:.2f}%]")
    if mcnemar_result:
        print(f"McNemar p: {mcnemar_result.get('mcnemar_p', 'N/A'):.2e}")
        print(f"Delta pp : {mcnemar_result.get('delta_pp', 'N/A'):+.2f}pp")
    print(f"Log      : logs/phase3.log")


if __name__ == "__main__":
    main()
