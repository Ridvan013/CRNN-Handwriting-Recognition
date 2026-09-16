#!/usr/bin/env python3
"""Hash the greedy test hypotheses of one checkpoint.  Run it twice in
separate processes: identical hashes mean the optical pass is reproducible
across processes (cuDNN deterministic algorithms, fp32).

Usage:  python cloud/determinism_probe.py --iam-words ... --iam-root ... [--model Model_abl_full]
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cloud"))
import torch
from torch.utils.data import DataLoader
from model_v3 import DEVICE, CRNNModel, CHAR_LIST, IAMDataset, custom_collate_fn
from v3_augmented_train import load_iam_aachen
from ablation_lexicon_all import hypotheses

p = argparse.ArgumentParser()
p.add_argument("--iam-words", default="")
p.add_argument("--iam-root", default="")
p.add_argument("--model", default="Model_abl_full")
a = p.parse_args()
(_, _, _, _, test_imgs, test_labs) = load_iam_aachen(REPO_ROOT, iam_words_override=a.iam_words, iam_root_override=a.iam_root)
loader = DataLoader(IAMDataset(test_imgs, test_labs, is_training=False), batch_size=128, shuffle=False, collate_fn=custom_collate_fn)
m = CRNNModel(img_height=32, img_width=128, num_classes=len(CHAR_LIST) + 1).to(DEVICE)
m.load_state_dict(torch.load(str(REPO_ROOT / a.model / "best_model_wa.pth"), map_location=DEVICE))
m.eval()
raw, _ = hypotheses(m, loader)
h = hashlib.sha256("\n".join(raw).encode("utf-8")).hexdigest()
print(f"{a.model}: {len(raw)} hypotheses, sha256 {h[:16]}")
