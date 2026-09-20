#!/usr/bin/env python3
"""
Exact McNemar for CRNN-B vs CRNN-LX under the LEFT-TO-RIGHT trigram corrector.

Section 5.1 reports the same pair of optical models under all three
correctors, to show that the sign of the augmentation difference does not
depend on the post-processing.  The unigram and whole-line numbers come from
results/ablation_lexicon5_all.json and results/ablation_viterbi.json; this
script supplies the missing left-to-right pair and writes it to
results/mcnemar_left_pair.json.

Usage:
    python cloud/mcnemar_left_pair.py \
        --iam-words HTR_Using_CRNN/IAM/processed/archive/iam_words/words.txt \
        --iam-root  HTR_Using_CRNN/IAM/processed/archive/iam_words/words
"""
from __future__ import annotations

import argparse
import json
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
from trigram_lm import TrigramLanguageModel
from v3_augmented_train import load_iam_aachen
from ablation_lexicon import score
from ablation_lexicon_all import hypotheses, mcnemar_exact
from ablation_trigram import split_ids
from kn_trigram import KNTrigram, ContextCorrector, iam_line_segments, brown_sentences

p = argparse.ArgumentParser()
p.add_argument("--iam-words", default="")
p.add_argument("--iam-root", default="")
p.add_argument("--alpha", type=float, default=7.0, help="alpha selected on validation for both models")
a = p.parse_args()

(_, _, _, _, test_imgs, test_labs) = load_iam_aachen(
    REPO_ROOT, iam_words_override=a.iam_words, iam_root_override=a.iam_root)
img_root = a.iam_root or str(REPO_ROOT / "HTR_Using_CRNN/IAM/processed/archive/iam_words/words")
ids = split_ids(REPO_ROOT / "aachen_splits" / "test_words.txt", img_root)
loader = DataLoader(IAMDataset(test_imgs, test_labs, is_training=False), batch_size=128,
                    shuffle=False, collate_fn=custom_collate_fn)

train_words = str(REPO_ROOT / "aachen_splits" / "train_words.txt")
lex = TrigramLanguageModel(train_words, use_nltk_extension=True)
kn = KNTrigram(iam_line_segments(train_words) + brown_sentences(), discount=0.75,
               extra_vocab=lex.vocabulary)
cc = ContextCorrector(lex, kn)

out = {"corrector": "KN3 (IAM+Brown), left-to-right", "alpha": a.alpha, "models": {}}
flags = {}
for mode in ("narrow", "full"):
    model = CRNNModel(img_height=32, img_width=128, num_classes=len(CHAR_LIST) + 1).to(DEVICE)
    model.load_state_dict(torch.load(str(REPO_ROOT / f"Model_abl_{mode}" / "best_model_wa.pth"),
                                     map_location=DEVICE))
    model.eval()
    raw, refs = hypotheses(model, loader)
    assert all(r == row[3] for r, row in zip(refs, ids))
    rows = [(lid, idx, h) for (_, lid, idx, _), h in zip(ids, raw)]
    o = cc.correct_lines(rows, a.alpha, True)
    s = score(o, refs)
    out["models"][mode] = s
    flags[mode] = [x == r for x, r in zip(o, refs)]
    print(f" {mode:<8} WA {s['wa_pct']:.4f}  CER {s['cer_pct']:.4f}")
    del model
    torch.cuda.empty_cache()

b, c, pv = mcnemar_exact(flags["narrow"], flags["full"])
d = out["models"]["full"]["wa_pct"] - out["models"]["narrow"]["wa_pct"]
out["mcnemar_full_vs_narrow"] = {"only_narrow_correct": b, "only_full_correct": c,
                                 "delta_wa_pp": round(d, 4), "p_value": pv}
print(f"\n CRNN-LX - CRNN-B = {d:+.4f} pp   (only narrow {b}, only full {c})   p = {pv:.4g}")
dst = REPO_ROOT / "results" / "mcnemar_left_pair.json"
json.dump(out, open(dst, "w"), indent=2)
print(" written:", dst)
