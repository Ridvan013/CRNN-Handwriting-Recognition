#!/usr/bin/env python3
"""
Phase 2 — Synthetic Pretraining

V3 CRNN modelini 300K synthetic image üzerinde 8 epoch eğitir.
Pretrain checkpoint Phase 3 fine-tuning için başlangıç noktası olacak.

Hyperparameters (H2_PLAN.md):
  optimizer  : AdamW(lr=1e-3, weight_decay=1e-5)
  scheduler  : cosine warmup 2 epoch + cosine decay
  epochs     : 8
  batch_size : 128
  AMP        : enabled
  trigram    : OFF (synthetic için gereksiz)

Çalıştırma (repo root'undan):
  python cloud/phase2_pretrain.py [--epochs 8] [--batch 128]

Output:
  checkpoints/pretrain_best.pth   ← Phase 3'ün başlangıç noktası
  checkpoints/pretrain_history.json
"""

import argparse
import os
import sys
import json
import random
from pathlib import Path

# Repo root'u path'e ekle (model_v3 importu için)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "cloud"))

from model_v3 import (
    DEVICE, CRNNModel, CHAR_LIST, SyntheticDataset,
    CRNNTrainer, custom_collate_fn
)
import torch
from torch.utils.data import DataLoader, Subset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int,   default=8)
    p.add_argument("--batch",        type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--synthetic-dir",type=str,   default="synthetic_data")
    p.add_argument("--ckpt-dir",     type=str,   default="checkpoints")
    p.add_argument("--val-size",     type=int,   default=10_000,
                   help="Validation olarak ayırılacak sample sayısı")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    synth_dir  = REPO_ROOT / args.synthetic_dir
    labels_file = synth_dir / "labels.txt"
    img_dir    = synth_dir / "words"
    ckpt_dir   = REPO_ROOT / args.ckpt_dir

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" Phase 2 — Synthetic Pretraining")
    print(f" Device   : {DEVICE}")
    print(f" Epochs   : {args.epochs}")
    print(f" Batch    : {args.batch}")
    print(f" LR (peak): {args.lr}")
    print(f" Synth dir: {synth_dir}")
    print(f" Ckpt dir : {ckpt_dir}")
    print("=" * 60)

    # ── Dataset ───────────────────────────────────────────────────────────────
    if not labels_file.exists():
        print(f"ERROR: labels.txt bulunamadı: {labels_file}")
        print("Phase 1'i önce çalıştır: python cloud/phase1_synthetic_gen.py")
        sys.exit(1)

    print("\n[1/3] Loading SyntheticDataset...")
    full_dataset = SyntheticDataset(
        str(labels_file), str(img_dir),
        is_training=True, device=DEVICE,
    )
    total = len(full_dataset)
    print(f"  Total samples: {total:,}")

    if total < 1000:
        print("ERROR: çok az sample. Phase 1 tamamlandı mı?")
        sys.exit(1)

    # Train / val split
    indices = list(range(total))
    random.shuffle(indices)
    val_n = min(args.val_size, max(1000, total // 10))
    val_idx   = indices[:val_n]
    train_idx = indices[val_n:]
    print(f"  Train: {len(train_idx):,}  |  Val: {len(val_idx):,}")

    # Val için augmentation kapalı
    val_dataset_raw = SyntheticDataset(
        str(labels_file), str(img_dir),
        is_training=False, device=DEVICE,
    )

    train_subset = Subset(full_dataset, train_idx)
    val_subset   = Subset(val_dataset_raw, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=args.batch, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False,
        collate_fn=custom_collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[2/3] Building V3 CRNN model...")
    model = CRNNModel(img_height=32, img_width=128, num_classes=len(CHAR_LIST) + 1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[3/3] Training...")
    trainer = CRNNTrainer(
        model,
        lr=args.lr,
        warmup_epochs=2,
        total_epochs=args.epochs,
        patience=args.epochs,   # pretrain'de early stopping yok, tam 8 epoch koş
        model_dir=str(ckpt_dir),
        device=DEVICE,
        trigram_lm=None,        # pretrain'de trigram kapalı
    )

    history = trainer.train(
        train_loader, val_loader,
        epochs=args.epochs,
        cnn_freeze_epochs=0,    # pretrain'de freeze yok
    )

    # ── Checkpoint yeniden adlandır ────────────────────────────────────────────
    # CRNNTrainer "best_model_wa.pth" kaydeder; pretrain için rename ediyoruz
    best_wa_ckpt = ckpt_dir / "best_model_wa.pth"
    pretrain_ckpt = ckpt_dir / "pretrain_best.pth"
    if best_wa_ckpt.exists():
        import shutil
        shutil.copy(str(best_wa_ckpt), str(pretrain_ckpt))
        print(f"\n  Pretrain checkpoint → {pretrain_ckpt}")
    else:
        print("  ⚠️  best_model_wa.pth bulunamadı")

    # ── JSON history ──────────────────────────────────────────────────────────
    hist_path = ckpt_dir / "pretrain_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    best_wa = max(history["val_wa"]) if history["val_wa"] else 0.0
    best_cer = min(history["val_cer"]) if history["val_cer"] else 1.0
    print(f"\n{'='*60}")
    print(f" Phase 2 TAMAMLANDI")
    print(f" Best synthetic val WA  : {best_wa*100:.2f}%")
    print(f" Best synthetic val CER : {best_cer*100:.2f}%")
    print(f" Checkpoint             : {pretrain_ckpt}")
    print(f"{'='*60}")

    if best_wa < 0.85:
        print(f"\n  ⚠️  Synthetic val WA %{best_wa*100:.1f} < %85 — beklenenin altında.")
        print(f"  Kontrol et: augmentation çok agresif mi? Fontlar var mı?")


if __name__ == "__main__":
    main()
