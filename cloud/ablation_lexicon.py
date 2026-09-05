#!/usr/bin/env python3
"""
Lexicon / trigram ablation for the paper (Table: post-processing ablation).

Runs ONE trained AugCRNN model over the Aachen test set once, caches the raw
greedy hypotheses, and then applies four increasingly informed post-processing
configurations to the same hypotheses:

    1. AugCRNN                      raw greedy CTC output, no lexicon
    2. AugCRNN + IAM lexicon        IAM-only lexicon (5.9K), edit distance only
    3. AugCRNN + IAM lex. + trigram IAM-only lexicon, n-gram rescoring
    4. AugCRNN-T (proposed)         IAM+NLTK lexicon (238K), n-gram rescoring

Because all four share the same optical hypotheses, the differences isolate
the contribution of each post-processing stage exactly.

Usage:
    python cloud/ablation_lexicon.py \
        --model  Model_aachen_v3_augmented/best_model_wa.pth \
        --iam-words <words.txt> --iam-root <words/> \
        --out results/ablation_lexicon.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cloud"))

import torch

from model_v3 import (
    DEVICE, CRNNModel, CHAR_LIST, PAD_TOKEN,
    IAMDataset, greedy_decode, custom_collate_fn, decode_labels,
    calculate_metrics, wilson_ci,
)
from torch.utils.data import DataLoader
from trigram_lm import TrigramLanguageModel
from v3_augmented_train import load_iam_aachen


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default="Model_aachen_v3_augmented/best_model_wa.pth")
    p.add_argument("--iam-words", type=str, default="")
    p.add_argument("--iam-root", type=str, default="")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--out", type=str, default="results/ablation_lexicon.json")
    return p.parse_args()


# ─────────────────── post-processing configurations ────────────────────────

def correct_lexicon_only(word: str, lm: TrigramLanguageModel) -> str:
    """Lexicon membership + edit-distance repair, WITHOUT n-gram rescoring.

    Mirrors TrigramLanguageModel.correct_word but replaces the n-gram score
    with a constant, so candidates are ranked purely by edit distance.
    """
    if word in lm.vocabulary or word.lower() in lm.vocabulary_lower:
        return word

    if len(word) <= 4:
        max_dist = 1
    elif len(word) <= 8:
        max_dist = 2
    else:
        max_dist = 2

    best, best_d = word, max_dist + 1
    for cand in lm.vocabulary:
        if abs(len(word) - len(cand)) > max_dist:
            continue
        d = lm._edit_distance(word, cand)
        if d < best_d:
            best, best_d = cand, d
            if d == 1:
                break
    return best


def score(preds, refs):
    n = len(refs)
    k = sum(1 for p, r in zip(preds, refs) if p.strip() == r.strip())
    lo, hi = wilson_ci(k, n)
    # character error rate
    def lev(a, b):
        if len(a) < len(b):
            a, b = b, a
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            cur = [i + 1]
            for j, cb in enumerate(b):
                cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (ca != cb)))
            prev = cur
        return prev[-1]
    err = sum(lev(p, r) for p, r in zip(preds, refs))
    tot = sum(len(r) for r in refs)
    return {
        "correct": k, "n": n,
        "wa_pct": round(k / n * 100, 4),
        "cer_pct": round(err / tot * 100, 4),
        "wilson_95ci_pct": [round(lo * 100, 4), round(hi * 100, 4)],
    }


def main():
    args = parse_args()
    print("=" * 66)
    print(" Lexicon / trigram post-processing ablation")
    print("=" * 66)

    # ── data ────────────────────────────────────────────────────────────────
    (_, _, _, _, test_imgs, test_labs) = load_iam_aachen(
        REPO_ROOT,
        iam_words_override=args.iam_words,
        iam_root_override=args.iam_root,
    )
    test_ds = IAMDataset(test_imgs, test_labs, is_training=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             collate_fn=custom_collate_fn)

    # ── model ───────────────────────────────────────────────────────────────
    model = CRNNModel(img_height=32, img_width=128,
                      num_classes=len(CHAR_LIST) + 1).to(DEVICE)
    ckpt = args.model if os.path.isabs(args.model) else str(REPO_ROOT / args.model)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    print(f" Model     : {ckpt}")

    # ── one forward pass, cache greedy hypotheses ───────────────────────────
    raw, refs = [], []
    cached_T = None
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            B = images.size(0)
            ctx = torch.amp.autocast("cuda") if torch.cuda.is_available() \
                else torch.autocast("cpu", enabled=False)
            with ctx:
                lp = model(images)
                if cached_T is None:
                    cached_T = lp.size(0)
            lengths = torch.full((B,), cached_T, dtype=torch.long, device=DEVICE)
            for seq in greedy_decode(lp, lengths):
                raw.append(decode_labels(seq))
            for t in labels:
                refs.append("".join(CHAR_LIST[i] for i in t.tolist()
                                    if i != PAD_TOKEN)
                            if isinstance(t, torch.Tensor) else decode_labels(t))
    print(f" Test words: {len(refs):,}\n")

    # ── lexicons ────────────────────────────────────────────────────────────
    lm_src = args.iam_words if args.iam_words else None
    aachen_words = str(REPO_ROOT / "aachen_splits" / "train_words.txt")
    print(" Building IAM-only lexicon ...")
    lm_iam = TrigramLanguageModel(aachen_words, use_nltk_extension=False)
    print("\n Building IAM+NLTK lexicon ...")
    lm_full = TrigramLanguageModel(aachen_words, use_nltk_extension=True)

    # ── four configurations over the SAME hypotheses ────────────────────────
    configs = {
        "AugCRNN (no lexicon)":
            raw,
        "AugCRNN + IAM lexicon":
            [correct_lexicon_only(w, lm_iam) for w in raw],
        "AugCRNN + IAM lexicon + trigram":
            [lm_iam.correct_word(w) for w in raw],
        "AugCRNN-T (IAM+NLTK lexicon + trigram)":
            [lm_full.correct_word(w) for w in raw],
    }

    out = {"n_samples": len(refs), "model": args.model,
           "lexicon_sizes": {"iam_only": len(lm_iam.vocabulary),
                             "iam_plus_nltk": len(lm_full.vocabulary)},
           "configurations": []}

    print(f"\n{'Configuration':<42s}{'WA (%)':>9s}{'CER (%)':>9s}  95% CI")
    print("-" * 78)
    for name, preds in configs.items():
        s = score(preds, refs)
        s["name"] = name
        out["configurations"].append(s)
        ci = s["wilson_95ci_pct"]
        print(f"{name:<42s}{s['wa_pct']:>9.2f}{s['cer_pct']:>9.2f}"
              f"  [{ci[0]:.2f}, {ci[1]:.2f}]")
    print("-" * 78)

    dst = Path(args.out if os.path.isabs(args.out) else REPO_ROOT / args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n written: {dst}")


if __name__ == "__main__":
    main()
