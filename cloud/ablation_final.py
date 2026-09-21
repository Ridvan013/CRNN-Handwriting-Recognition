#!/usr/bin/env python3
"""
Score optical models with the FINAL corrector of the paper.

Final corrector (selected on the validation partition, see
results/ablation_lexicon_source.json): the lexicon is the vocabulary of the
language-model corpus -- the IAM training lines plus the Brown corpus,
57,382 types, 96.4 % coverage of the test tokens -- and the candidates are
ranked by the interpolated Kneser-Ney trigram of that corpus with the
recognizer's own previous outputs on the line as context (left to right).
The edit penalty alpha is selected per model on its own validation partition.

Also dumps, for the CRNN-LX model, a per-word file with the greedy output and
the correctors the paper compares against, so the error analysis and the
paired tests read from one place.

Usage:
    python cloud/ablation_final.py \
        --modes none,narrow,photo,elastic,morph,full \
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

LABELS = {"none": "no augmentation", "narrow": "CRNN-B (baseline)",
          "photo": "+ wide photometric", "elastic": "+ elastic",
          "morph": "+ morphological", "full": "CRNN-LX (all)"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iam-words", default="")
    p.add_argument("--iam-root", default="")
    p.add_argument("--modes", default="none,narrow,photo,elastic,morph,full",
                   help="'name' (Model_abl_<name>/) or 'name=dir'")
    p.add_argument("--baseline", default="narrow")
    p.add_argument("--variants-model", default="full",
                   help="model whose per-word comparison file is written")
    p.add_argument("--repeat-check", default="full")
    p.add_argument("--alphas", default="3,5,7,10")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--out", default="results/ablation_final.json")
    p.add_argument("--dump-preds", default="results/preds_final")
    return p.parse_args()


def main():
    a = parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]
    modes, dirs = [], {}
    for tok in a.modes.split(","):
        tok = tok.strip()
        if tok:
            name, _, d = tok.partition("=")
            modes.append(name)
            dirs[name] = d or f"Model_abl_{name}"
    t0 = time.time()
    print("=" * 78)
    print(" Final corrector: corpus-vocabulary lexicon + KN trigram, line context")
    print("=" * 78)

    (_, _, val_imgs, val_labs, test_imgs, test_labs) = load_iam_aachen(
        REPO_ROOT, iam_words_override=a.iam_words, iam_root_override=a.iam_root)
    img_root = a.iam_root or str(REPO_ROOT / "HTR_Using_CRNN/IAM/processed/archive/iam_words/words")
    ids = {"val": split_ids(REPO_ROOT / "aachen_splits" / "validation_words.txt", img_root),
           "test": split_ids(REPO_ROOT / "aachen_splits" / "test_words.txt", img_root)}
    loaders = {"val": DataLoader(IAMDataset(val_imgs, val_labs, is_training=False), batch_size=a.batch,
                                 shuffle=False, collate_fn=custom_collate_fn),
               "test": DataLoader(IAMDataset(test_imgs, test_labs, is_training=False), batch_size=a.batch,
                                  shuffle=False, collate_fn=custom_collate_fn)}

    train_words = str(REPO_ROOT / "aachen_splits" / "train_words.txt")
    segs = iam_line_segments(train_words)
    brown = brown_sentences()
    corpus_vocab = set()
    for s in segs + brown:
        corpus_vocab.update(s)
    lex = TrigramLanguageModel(train_words, use_nltk_extension=False)
    lex.vocabulary = set(corpus_vocab)
    lex.vocabulary_lower = {w.lower() for w in lex.vocabulary}
    lex.__dict__.pop("_vocab_by_len", None)
    kn = KNTrigram(segs + brown, discount=0.75, extra_vocab=lex.vocabulary)
    cc = ContextCorrector(lex, kn)
    # the 239 K NLTK-extended lexicon, kept only for the comparison file
    lex_nltk = TrigramLanguageModel(train_words, use_nltk_extension=True)
    kn_nltk = KNTrigram(segs + brown, discount=0.75, extra_vocab=lex_nltk.vocabulary)
    cc_nltk = ContextCorrector(lex_nltk, kn_nltk)
    print(" lexicon:", f"{len(lex.vocabulary):,} types |", kn.describe(), f"[{time.time()-t0:.0f}s]")

    out = {"corrector": "corpus-vocabulary lexicon + KN3 trigram, left-to-right line context",
           "lexicon_types": len(lex.vocabulary), "alphas": alphas,
           "n_samples": len(test_labs), "models": {}}
    flags, preds, first_raw, refs_t = {}, {}, {}, None
    print(f"\n {'model':<12}{'alpha':>6}{'greedy':>8}{'FINAL':>8}{'CER':>7}   val")
    for mode in modes:
        ckpt = REPO_ROOT / dirs[mode] / "best_model_wa.pth"
        if not ckpt.exists():
            print(f" !! missing: {ckpt}")
            continue
        model = CRNNModel(img_height=32, img_width=128, num_classes=len(CHAR_LIST) + 1).to(DEVICE)
        model.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))
        model.eval()
        raw, refs, rows = {}, {}, {}
        for k in ("val", "test"):
            raw[k], refs[k] = hypotheses(model, loaders[k])
            assert all(r == row[3] for r, row in zip(refs[k], ids[k])), f"{mode}/{k} misaligned"
            rows[k] = [(lid, idx, h) for (_, lid, idx, _), h in zip(ids[k], raw[k])]
        refs_t = refs["test"]
        first_raw[mode] = raw["test"]
        del model
        torch.cuda.empty_cache()

        grid = {x: score(cc.correct_lines(rows["val"], x, True), refs["val"])["wa_pct"] for x in alphas}
        best = max(alphas, key=lambda x: (grid[x], -x))
        o_test = cc.correct_lines(rows["test"], best, True)
        s = score(o_test, refs["test"])
        out["models"][mode] = {"label": LABELS.get(mode, mode), "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
                               "selected_alpha_on_val": best, "val_alpha_grid": grid,
                               "val": score(cc.correct_lines(rows["val"], best, True), refs["val"]),
                               "test": {"greedy": score(raw["test"], refs["test"]), "final": s}}
        flags[mode] = [p == r for p, r in zip(o_test, refs["test"])]
        preds[mode] = o_test
        print(f" {mode:<12}{best:>6.0f}{out['models'][mode]['test']['greedy']['wa_pct']:>8.2f}"
              f"{s['wa_pct']:>8.2f}{s['cer_pct']:>7.2f}   {grid[best]:.2f}   [{time.time()-t0:.0f}s]", flush=True)

        if mode == a.variants_model:
            va = {"greedy": raw["test"], "final (corpus lexicon, KN3 L-R)": o_test,
                  "unigram prior (corpus lexicon)": [lex.correct_word(w) for w in raw["test"]],
                  "KN3 L-R (239K NLTK lexicon)": cc_nltk.correct_lines(rows["test"], 7.0, True),
                  "unigram prior (239K NLTK lexicon)": [lex_nltk.correct_word(w) for w in raw["test"]]}
            out["variants_" + mode] = {k: score(v, refs["test"]) for k, v in va.items()}
            ddir = REPO_ROOT / a.dump_preds
            ddir.mkdir(parents=True, exist_ok=True)
            with open(ddir / "preds_test_variants.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                cols = list(va)
                w.writerow(["idx", "word_id", "ground_truth"] + cols)
                for i, (row, r) in enumerate(zip(ids["test"], refs["test"])):
                    w.writerow([i, row[0], r] + [va[c][i] for c in cols])
            print("   variants written:", {k: round(v["wa_pct"], 2) for k, v in out["variants_" + mode].items()})

    base = a.baseline
    if base in flags:
        out["mcnemar_vs_" + base] = {}
        print(f"\n exact McNemar against '{base}' (N={len(refs_t):,})")
        for mode in modes:
            if mode == base or mode not in flags:
                continue
            b, c, pv = mcnemar_exact(flags[base], flags[mode])
            d = 100.0 * (c - b) / len(flags[mode])
            out["mcnemar_vs_" + base][mode] = {"only_baseline_correct": b, "only_mode_correct": c,
                                               "delta_wa_pp": round(d, 4), "p_value": pv}
            print(f"   {mode:<12} b={b:<5d} c={c:<5d} dWA={d:+.2f} p={pv:.3g}")

    rc = a.repeat_check
    if rc in first_raw:
        model = CRNNModel(img_height=32, img_width=128, num_classes=len(CHAR_LIST) + 1).to(DEVICE)
        model.load_state_dict(torch.load(str(REPO_ROOT / dirs[rc] / "best_model_wa.pth"), map_location=DEVICE))
        model.eval()
        raw2, _ = hypotheses(model, loaders["test"])
        nd = sum(1 for x, y in zip(first_raw[rc], raw2) if x != y)
        out["determinism_check"] = {"mode": rc, "differing_words": nd, "n": len(raw2)}
        print(f"\n determinism ({rc}): {nd} of {len(raw2)} hypotheses differ between two passes")

    dst = REPO_ROOT / a.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w"), indent=2)
    ddir = REPO_ROOT / a.dump_preds
    ddir.mkdir(parents=True, exist_ok=True)
    for mode, p in preds.items():
        with open(ddir / f"preds_{mode}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "word_id", "ground_truth", "prediction", "correct"])
            for i, (row, pr, r) in enumerate(zip(ids["test"], p, refs_t)):
                w.writerow([i, row[0], r, pr, int(pr == r)])
    print(f"\n written: {dst}; per-word predictions in {ddir}   [{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
