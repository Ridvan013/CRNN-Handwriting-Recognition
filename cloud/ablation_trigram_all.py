#!/usr/bin/env python3
"""
Score all six optical models with the trigram-with-context corrector, so that
the augmentation ablation (Table 2) can be reported with the same corrector as
the headline result.

For every model: fp32 hypotheses on validation and test; the paper's unigram
corrector (U-IAM, alpha=5) for reference; KN3-IAM+Brown with alpha selected on
the model's OWN validation partition from {3,5,7,10}; exact McNemar against the
baseline (narrow) model on the test partition; per-word predictions dumped.
A repeat pass on one model proves determinism.

Usage:
    python cloud/ablation_trigram_all.py \
        --iam-words HTR_Using_CRNN/IAM/processed/archive/iam_words/words.txt \
        --iam-root  HTR_Using_CRNN/IAM/processed/archive/iam_words/words
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
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

MODES = ["none", "narrow", "photo", "elastic", "morph", "full"]
LABELS = {"none": "no augmentation", "narrow": "CRNN-B (baseline)",
          "photo": "+ wide photometric", "elastic": "+ elastic",
          "morph": "+ morphological", "full": "CRNN-LX (all)"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iam-words", type=str, default="")
    p.add_argument("--iam-root", type=str, default="")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--modes", type=str, default=",".join(MODES))
    p.add_argument("--alphas", type=str, default="3,5,7,10")
    p.add_argument("--discount", type=float, default=0.75)
    p.add_argument("--baseline", type=str, default="narrow")
    p.add_argument("--repeat-check", type=str, default="full")
    p.add_argument("--out", type=str, default="results/ablation_trigram_all.json")
    p.add_argument("--dump-preds", type=str, default="results/preds_trigram")
    return p.parse_args()


def main():
    args = parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    # --modes accepts "name" (checkpoint in Model_abl_<name>/) or "name=dir"
    # (checkpoint in <dir>/), e.g. full_s123=Model_seed_full_123 for the seed repeats.
    modes, mode_dirs = [], {}
    for tok in args.modes.split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, _, d = tok.partition("=")
        modes.append(name)
        mode_dirs[name] = d or f"Model_abl_{name}"
    t0 = time.time()
    print("=" * 78)
    print(" Augmentation ablation scored with the trigram-with-context corrector")
    print("=" * 78)

    (_, _, val_imgs, val_labs, test_imgs, test_labs) = load_iam_aachen(
        REPO_ROOT, iam_words_override=args.iam_words, iam_root_override=args.iam_root)
    img_root = args.iam_root or str(REPO_ROOT / "HTR_Using_CRNN/IAM/processed/archive/iam_words/words")
    ids = {"val": split_ids(REPO_ROOT / "aachen_splits" / "validation_words.txt", img_root),
           "test": split_ids(REPO_ROOT / "aachen_splits" / "test_words.txt", img_root)}
    loaders = {"val": DataLoader(IAMDataset(val_imgs, val_labs, is_training=False), batch_size=args.batch,
                                 shuffle=False, collate_fn=custom_collate_fn),
               "test": DataLoader(IAMDataset(test_imgs, test_labs, is_training=False), batch_size=args.batch,
                                  shuffle=False, collate_fn=custom_collate_fn)}

    train_words = str(REPO_ROOT / "aachen_splits" / "train_words.txt")
    lex = TrigramLanguageModel(train_words, use_nltk_extension=True)
    kn = KNTrigram(iam_line_segments(train_words) + brown_sentences(),
                   discount=args.discount, extra_vocab=lex.vocabulary)
    cc = ContextCorrector(lex, kn)
    print(" LM:", kn.describe(), f"[{time.time()-t0:.0f}s]")

    out = {"alphas": alphas, "discount": args.discount, "corrector": "KN3-IAM+Brown",
           "n_samples": len(test_labs), "fp32_deterministic": True, "models": {}}
    final_flags, final_preds, first_raw, refs_all = {}, {}, {}, None
    print(f"\n {'model':<8s}{'alpha':>6s}{'greedy':>8s}{'U-IAM':>8s}{'KN3+Br':>8s}{'CER':>7s}   val(U-IAM / KN3+Br)")
    for mode in modes:
        ckpt = REPO_ROOT / mode_dirs[mode] / "best_model_wa.pth"
        if not ckpt.exists():
            print(f" !! missing checkpoint: {ckpt}")
            continue
        model = CRNNModel(img_height=32, img_width=128, num_classes=len(CHAR_LIST) + 1).to(DEVICE)
        model.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))
        model.eval()
        raw, refs = {}, {}
        for k in ("val", "test"):
            raw[k], refs[k] = hypotheses(model, loaders[k])
            assert len(ids[k]) == len(refs[k])
            assert all(r == row[3] for r, row in zip(refs[k], ids[k])), f"{mode}/{k}: misaligned"
        refs_all = refs["test"]
        first_raw[mode] = raw["test"]
        rows = {k: [(lid, idx, h) for (_, lid, idx, _), h in zip(ids[k], raw[k])] for k in ("val", "test")}

        ref_out = {k: [lex.correct_word(w) for w in raw[k]] for k in ("val", "test")}
        grid = {}
        for a in alphas:
            o_val = cc.correct_lines(rows["val"], a, True)
            grid[a] = {"val": score(o_val, refs["val"])}
        best_a = max(alphas, key=lambda a: (grid[a]["val"]["wa_pct"], -a))
        o_test = cc.correct_lines(rows["test"], best_a, True)
        s_greedy = score(raw["test"], refs["test"])
        s_ref = score(ref_out["test"], refs["test"])
        s_new = score(o_test, refs["test"])
        s_val_ref = score(ref_out["val"], refs["val"])
        out["models"][mode] = {
            "label": LABELS.get(mode, mode), "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
            "selected_alpha_on_val": best_a,
            "val": {"U-IAM": s_val_ref, "KN3-IAM+Brown": grid[best_a]["val"],
                    "alpha_grid": {str(a): grid[a]["val"]["wa_pct"] for a in alphas}},
            "test": {"greedy": s_greedy, "U-IAM": s_ref, "KN3-IAM+Brown": s_new}}
        final_flags[mode] = [p == r for p, r in zip(o_test, refs["test"])]
        final_preds[mode] = o_test
        print(f" {mode:<8s}{best_a:>6.0f}{s_greedy['wa_pct']:>8.2f}{s_ref['wa_pct']:>8.2f}"
              f"{s_new['wa_pct']:>8.2f}{s_new['cer_pct']:>7.2f}   "
              f"{s_val_ref['wa_pct']:.2f} / {grid[best_a]['val']['wa_pct']:.2f}   [{time.time()-t0:.0f}s]")
        del model
        torch.cuda.empty_cache()

    base = args.baseline
    if base in final_flags:
        out["mcnemar_vs_" + base] = {}
        print(f"\n exact McNemar against '{base}' (KN3-IAM+Brown outputs, N={len(refs_all):,})")
        for mode, flags in final_flags.items():
            if mode == base:
                continue
            b, c, pv = mcnemar_exact(final_flags[base], flags)
            d = 100.0 * (c - b) / len(flags)
            out["mcnemar_vs_" + base][mode] = {"only_baseline_correct": b, "only_mode_correct": c,
                                               "delta_wa_pp": round(d, 4), "p_value": pv}
            print(f"   {mode:<8s} b={b:<5d} c={c:<5d} dWA={d:+.2f} p={pv:.3g}")

    rc = args.repeat_check
    if rc in first_raw:
        ckpt = REPO_ROOT / mode_dirs[rc] / "best_model_wa.pth"
        model = CRNNModel(img_height=32, img_width=128, num_classes=len(CHAR_LIST) + 1).to(DEVICE)
        model.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))
        model.eval()
        raw2, _ = hypotheses(model, loaders["test"])
        nd = sum(1 for a, b in zip(first_raw[rc], raw2) if a != b)
        out["determinism_check"] = {"mode": rc, "differing_words": nd, "n": len(raw2)}
        print(f"\n determinism check ({rc}): {nd} of {len(raw2)} hypotheses differ between two passes")

    dst = Path(args.out if os.path.isabs(args.out) else REPO_ROOT / args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w"), indent=2)
    ddir = Path(args.dump_preds if os.path.isabs(args.dump_preds) else REPO_ROOT / args.dump_preds)
    ddir.mkdir(parents=True, exist_ok=True)
    for mode, preds in final_preds.items():
        with open(ddir / f"preds_{mode}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "word_id", "ground_truth", "prediction", "correct"])
            for i, (row, pr, rf) in enumerate(zip(ids["test"], preds, refs_all)):
                w.writerow([i, row[0], rf, pr, int(pr == rf)])
    print(f"\n written: {dst}; per-word predictions in {ddir}   [{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
