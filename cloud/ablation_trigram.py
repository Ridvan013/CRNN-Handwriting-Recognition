#!/usr/bin/env python3
"""
Does a real trigram with line context beat the unigram frequency prior?

Runs the CRNN-LX optical model (fp32, deterministic) on the validation and
test partitions, then applies several post-correctors to the SAME hypotheses:

  U-IAM        the paper's corrector: add-one unigram prior on IAM counts, alpha=5
               (reference; must reproduce the paper's 80.74 % on test)
  KN1-IAM      interpolated Kneser-Ney, IAM training lines, unigram order only
  KN3-IAM      the same model, trigram order, context = own outputs on the line
  KN1-IAM+Br   KN unigram, IAM lines + Brown corpus (1.16 M words)
  KN3-IAM+Br   KN trigram + context, IAM lines + Brown corpus

KN1 vs KN3 isolates the effect of CONTEXT; *-IAM vs *-IAM+Br isolates the
effect of the CORPUS.  The edit penalty alpha is selected on the validation
partition for every variant (never on test).  An ORACLE row that feeds the
ground-truth neighbours as context is printed as a diagnostic upper bound and
must not be reported as a result.

Usage:
    python cloud/ablation_trigram.py \
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

from model_v3 import DEVICE, CRNNModel, CHAR_LIST, IAMDataset, custom_collate_fn, encode_to_labels
from trigram_lm import TrigramLanguageModel
from v3_augmented_train import load_iam_aachen
from ablation_lexicon import score
from ablation_lexicon_all import hypotheses, mcnemar_exact
from kn_trigram import KNTrigram, ContextCorrector, iam_line_segments, brown_sentences


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iam-words", type=str, default="")
    p.add_argument("--iam-root", type=str, default="")
    p.add_argument("--model", type=str, default="Model_abl_full/best_model_wa.pth")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--alphas", type=str, default="3,5,7,10")
    p.add_argument("--discount", type=float, default=0.75)
    p.add_argument("--out", type=str, default="results/ablation_trigram.json")
    p.add_argument("--dump-preds", type=str, default="results/preds_trigram")
    return p.parse_args()


def split_ids(split_file: Path, img_root: str):
    """(word_id, line_id, idx, word) for every row the loader keeps, in loader order."""
    rows = []
    with open(split_file, encoding="utf-8") as fh:
        for l in fh:
            l = l.strip()
            if not l or l.startswith("#"):
                continue
            parts = l.split()
            if len(parts) < 9 or parts[1] != "ok":
                continue
            wid = parts[0]
            word = "".join(parts[8:])
            a, b = wid.split("-")[0], wid.split("-")[1]
            if not os.path.exists(os.path.join(img_root, a, f"{a}-{b}", f"{wid}.png")):
                continue
            try:
                encode_to_labels(word)
            except ValueError:
                continue
            rows.append((wid, "-".join(wid.split("-")[:3]), int(wid.split("-")[3]), word))
    return rows


def main():
    args = parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    t0 = time.time()
    print("=" * 78)
    print(" Trigram-with-context post-correction vs the unigram prior (fp32)")
    print("=" * 78)

    # ---- data + hypotheses ------------------------------------------------
    (_, _, val_imgs, val_labs, test_imgs, test_labs) = load_iam_aachen(
        REPO_ROOT, iam_words_override=args.iam_words, iam_root_override=args.iam_root)
    img_root = args.iam_root or str(REPO_ROOT / "HTR_Using_CRNN/IAM/processed/archive/iam_words/words")
    ids = {"val": split_ids(REPO_ROOT / "aachen_splits" / "validation_words.txt", img_root),
           "test": split_ids(REPO_ROOT / "aachen_splits" / "test_words.txt", img_root)}

    model = CRNNModel(img_height=32, img_width=128, num_classes=len(CHAR_LIST) + 1).to(DEVICE)
    ckpt = args.model if os.path.isabs(args.model) else str(REPO_ROOT / args.model)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    raw, refs = {}, {}
    for name, imgs, labs in (("val", val_imgs, val_labs), ("test", test_imgs, test_labs)):
        loader = DataLoader(IAMDataset(imgs, labs, is_training=False), batch_size=args.batch,
                            shuffle=False, collate_fn=custom_collate_fn)
        raw[name], refs[name] = hypotheses(model, loader)
        assert len(ids[name]) == len(refs[name]), (name, len(ids[name]), len(refs[name]))
        bad = sum(1 for r, row in zip(refs[name], ids[name]) if r != row[3])
        assert bad == 0, f"{name}: {bad} rows misaligned between split file and loader"
        print(f" {name:<4s}: {len(refs[name]):,} words, ids aligned  [{time.time()-t0:.0f}s]")

    # ---- lexicon (candidate generation + the paper's reference corrector) --
    train_words = str(REPO_ROOT / "aachen_splits" / "train_words.txt")
    lex = TrigramLanguageModel(train_words, use_nltk_extension=True)

    # ---- language models --------------------------------------------------
    iam_segs = iam_line_segments(train_words)
    brown = brown_sentences()
    kn_iam = KNTrigram(iam_segs, discount=args.discount, extra_vocab=lex.vocabulary)
    kn_br = KNTrigram(iam_segs + brown, discount=args.discount, extra_vocab=lex.vocabulary)
    print(" IAM lines   :", kn_iam.describe())
    print(" IAM + Brown :", kn_br.describe())
    cc_iam = ContextCorrector(lex, kn_iam)
    cc_br = ContextCorrector(lex, kn_br)
    cc_br._cands = cc_iam._cands            # candidates depend only on the lexicon

    rows = {k: [(lid, idx, h) for (_, lid, idx, _), h in zip(ids[k], raw[k])] for k in ("val", "test")}

    # ---- reference --------------------------------------------------------
    results = {"alphas": alphas, "discount": args.discount, "variants": {}}
    ref_out = {k: [lex.correct_word(w) for w in raw[k]] for k in ("val", "test")}
    ref = {k: score(ref_out[k], refs[k]) for k in ("val", "test")}
    results["variants"]["U-IAM (paper, alpha=5)"] = {"val": ref["val"], "test": ref["test"]}
    print(f"\n {'variant':<22s} {'alpha':>5s} {'val WA':>8s} {'test WA':>8s} {'test CER':>9s}")
    print(f" {'U-IAM (paper)':<22s} {5:>5.0f} {ref['val']['wa_pct']:>8.2f} "
          f"{ref['test']['wa_pct']:>8.2f} {ref['test']['cer_pct']:>9.2f}   [{time.time()-t0:.0f}s]")
    raw_s = {k: score(raw[k], refs[k]) for k in ("val", "test")}
    results["variants"]["none (greedy)"] = {"val": raw_s["val"], "test": raw_s["test"]}

    # ---- KN variants, alpha selected on validation -------------------------
    variants = [("KN1-IAM", cc_iam, False), ("KN3-IAM", cc_iam, True),
                ("KN1-IAM+Brown", cc_br, False), ("KN3-IAM+Brown", cc_br, True)]
    outputs = {}
    test_flags_ref = [p == r for p, r in zip(ref_out["test"], refs["test"])]
    for name, cc, ctx in variants:
        grid = {}
        for a in alphas:
            o_val = cc.correct_lines(rows["val"], a, ctx)
            o_test = cc.correct_lines(rows["test"], a, ctx)
            grid[a] = {"val": score(o_val, refs["val"]), "test": score(o_test, refs["test"]),
                       "_test_out": o_test}
            print(f" {name:<22s} {a:>5.0f} {grid[a]['val']['wa_pct']:>8.2f} "
                  f"{grid[a]['test']['wa_pct']:>8.2f} {grid[a]['test']['cer_pct']:>9.2f}   [{time.time()-t0:.0f}s]")
        best_a = max(alphas, key=lambda a: (grid[a]["val"]["wa_pct"], -a))
        o_best = grid[best_a].pop("_test_out")
        for a in alphas:
            grid[a].pop("_test_out", None)
        flags = [p == r for p, r in zip(o_best, refs["test"])]
        b, c, pv = mcnemar_exact(test_flags_ref, flags)
        results["variants"][name] = {
            "selected_alpha_on_val": best_a,
            "val": grid[best_a]["val"], "test": grid[best_a]["test"],
            "mcnemar_vs_U-IAM": {"only_ref_correct": b, "only_variant_correct": c, "p_value": pv},
            "alpha_grid": {str(a): {"val_wa": grid[a]["val"]["wa_pct"], "test_wa": grid[a]["test"]["wa_pct"],
                                    "test_cer": grid[a]["test"]["cer_pct"]} for a in alphas}}
        outputs[name] = o_best
        print(f"   -> selected alpha={best_a:g} on val; test WA {grid[best_a]['test']['wa_pct']:.2f} "
              f"vs U-IAM {ref['test']['wa_pct']:.2f}: dWA={100*(c-b)/len(flags):+.2f} pp, "
              f"McNemar p={pv:.3g} (only-ref {b}, only-variant {c})")

    # ---- oracle diagnostic (ground-truth context; NOT a result) ------------
    best_name = max(["KN3-IAM", "KN3-IAM+Brown"], key=lambda k: results["variants"][k]["val"]["wa_pct"])
    cc = cc_br if "Brown" in best_name else cc_iam
    a = results["variants"][best_name]["selected_alpha_on_val"]
    o_or = cc.correct_lines(rows["test"], a, True, context_source=refs["test"])
    orc = score(o_or, refs["test"])
    results["oracle_diagnostic"] = {"variant": best_name, "alpha": a, "test": orc,
                                    "note": "ground-truth neighbours as context; upper bound only"}
    print(f"\n ORACLE ({best_name}, GT context, diagnostic only): test WA {orc['wa_pct']:.2f}")

    # ---- context availability on test --------------------------------------
    n = len(rows["test"]); with_prev = 0
    seen = {(lid, idx) for lid, idx, _ in rows["test"]}
    for lid, idx, _ in rows["test"]:
        if (lid, idx - 1) in seen:
            with_prev += 1
    results["context_available_pct"] = round(100 * with_prev / n, 1)
    oov_test = sum(1 for w in raw["test"] if not cc_iam.in_lexicon(w))
    results["hypotheses_out_of_lexicon_test"] = oov_test
    print(f" test words with a preceding word available: {100*with_prev/n:.1f}%; "
          f"hypotheses outside the lexicon (the only ones a corrector touches): {oov_test:,}")

    # ---- write --------------------------------------------------------------
    dst = Path(args.out if os.path.isabs(args.out) else REPO_ROOT / args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(dst, "w"), indent=2)
    ddir = Path(args.dump_preds if os.path.isabs(args.dump_preds) else REPO_ROOT / args.dump_preds)
    ddir.mkdir(parents=True, exist_ok=True)
    with open(ddir / "preds_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["word_id", "ground_truth", "greedy", "U-IAM"] + list(outputs))
        for i, (wid, _, _, gt) in enumerate(ids["test"]):
            w.writerow([wid, gt, raw["test"][i], ref_out["test"][i]] + [outputs[k][i] for k in outputs])
    print(f"\n written: {dst} and {ddir/'preds_test.csv'}   [{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
