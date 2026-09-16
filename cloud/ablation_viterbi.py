#!/usr/bin/env python3
"""
Two-sided line context: does joint (Viterbi) decoding of the whole line beat
left-to-right correction?

Same CRNN-LX hypotheses, same lexicon, same KN trigram (IAM + Brown).  Variants:

  KN3-left        left-to-right, own outputs as context (the current corrector)
  VIT             exact Viterbi over the line; only out-of-lexicon words have
                  candidate columns
  VIT+oov         ... and an out-of-lexicon hypothesis may also stay as it is
  VIT+rw          ... and in-lexicon words may be replaced by neighbours within
                  distance 1 (real-word errors)
  VIT+oov+rw      both

For every variant alpha in {3,5,7,10} is selected on validation; the variant
itself is selected on validation too, and only then scored once on test.
Then the selected configuration is applied to all six optical models.

Usage:
    python cloud/ablation_viterbi.py \
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
from kn_trigram import KNTrigram, LineViterbiCorrector, iam_line_segments, brown_sentences

MODES = ["none", "narrow", "photo", "elastic", "morph", "full"]
VARIANTS = {"VIT": dict(real_word=False, keep_oov=False),
            "VIT+oov": dict(real_word=False, keep_oov=True),
            "VIT+rw": dict(real_word=True, keep_oov=False),
            "VIT+oov+rw": dict(real_word=True, keep_oov=True)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iam-words", type=str, default="")
    p.add_argument("--iam-root", type=str, default="")
    p.add_argument("--batch", type=int, default=128)
    # wider than the {3,5,7,10} grid of the left-to-right corrector: with a
    # whole-line objective the useful penalty range shifts, and a selection on
    # the edge of the grid would be uninformative
    p.add_argument("--alphas", type=str, default="1,2,3,4,5,7,10,15,20,30")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--modes", type=str, default=",".join(MODES),
                   help="'name' (checkpoint in Model_abl_<name>/) or 'name=dir', e.g. "
                        "full_s123=Model_seed_full_123 for the seed repeats; 'full' must be included")
    p.add_argument("--baseline", type=str, default="narrow")
    p.add_argument("--out", type=str, default="results/ablation_viterbi.json")
    p.add_argument("--dump-preds", type=str, default="results/preds_viterbi")
    return p.parse_args()


def main():
    a = parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]
    modes, mode_dirs = [], {}
    for tok in a.modes.split(","):
        tok = tok.strip()
        if tok:
            name, _, d = tok.partition("=")
            modes.append(name)
            mode_dirs[name] = d or f"Model_abl_{name}"
    assert "full" in modes, "--modes must include 'full' (stage 1 selects the decoder on it)"
    t0 = time.time()
    print("=" * 78)
    print(" Two-sided line context (exact Viterbi) vs left-to-right correction")
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
    lex = TrigramLanguageModel(train_words, use_nltk_extension=True)
    kn = KNTrigram(iam_line_segments(train_words) + brown_sentences(), discount=0.75, extra_vocab=lex.vocabulary)
    vc = LineViterbiCorrector(lex, kn, topk=a.topk)
    print(" LM:", kn.describe(), f"[{time.time()-t0:.0f}s]")

    def run_model(mode):
        model = CRNNModel(img_height=32, img_width=128, num_classes=len(CHAR_LIST) + 1).to(DEVICE)
        model.load_state_dict(torch.load(str(REPO_ROOT / mode_dirs[mode] / "best_model_wa.pth"), map_location=DEVICE))
        model.eval()
        raw, refs = {}, {}
        for k in ("val", "test"):
            raw[k], refs[k] = hypotheses(model, loaders[k])
            assert len(ids[k]) == len(refs[k]) and all(r == row[3] for r, row in zip(refs[k], ids[k]))
        del model
        torch.cuda.empty_cache()
        rows = {k: [(lid, idx, h) for (_, lid, idx, _), h in zip(ids[k], raw[k])] for k in ("val", "test")}
        return raw, refs, rows

    # ---- stage 1: variant and alpha selection on validation, CRNN-LX -----
    raw, refs, rows = run_model("full")
    res = {"alphas": alphas, "topk": a.topk, "stage1_model": "full", "variants": {}}
    left = {}
    for al in alphas:
        left[al] = {k: vc.correct_lines(rows[k], al, True) for k in ("val", "test")}
    la = max(alphas, key=lambda x: (score(left[x]["val"], refs["val"])["wa_pct"], -x))
    ref_out = left[la]["test"]
    ref_flags = [p == r for p, r in zip(ref_out, refs["test"])]
    res["variants"]["KN3-left"] = {"selected_alpha_on_val": la,
                                   "val": score(left[la]["val"], refs["val"]),
                                   "test": score(ref_out, refs["test"])}
    print(f"\n {'variant':<12s}{'alpha':>6s}{'val WA':>8s}{'test WA':>9s}{'test CER':>9s}")
    print(f" {'KN3-left':<12s}{la:>6.0f}{res['variants']['KN3-left']['val']['wa_pct']:>8.2f}"
          f"{res['variants']['KN3-left']['test']['wa_pct']:>9.2f}{res['variants']['KN3-left']['test']['cer_pct']:>9.2f}   [{time.time()-t0:.0f}s]")
    outputs = {}
    for name, opts in VARIANTS.items():
        grid = {}
        for al in alphas:
            o_val = vc.decode_lines(rows["val"], al, **opts)
            grid[al] = {"val": score(o_val, refs["val"])}
            print(f" {name:<12s}{al:>6.0f}{grid[al]['val']['wa_pct']:>8.2f}{'':>9s}{'':>9s}   [{time.time()-t0:.0f}s]")
        best = max(alphas, key=lambda x: (grid[x]["val"]["wa_pct"], -x))
        o_test = vc.decode_lines(rows["test"], best, **opts)
        flags = [p == r for p, r in zip(o_test, refs["test"])]
        b, c, pv = mcnemar_exact(ref_flags, flags)
        st = score(o_test, refs["test"])
        res["variants"][name] = {"options": opts, "selected_alpha_on_val": best,
                                 "val": grid[best]["val"], "test": st,
                                 "alpha_grid_val": {str(x): grid[x]["val"]["wa_pct"] for x in alphas},
                                 "mcnemar_vs_KN3-left": {"only_left_correct": b, "only_variant_correct": c, "p_value": pv}}
        outputs[name] = o_test
        print(f"   -> {name}: alpha={best:g}; test WA {st['wa_pct']:.2f} CER {st['cer_pct']:.2f}; "
              f"vs left: +{c} / -{b}, p={pv:.3g}")

    # variant selected on validation (never on test)
    sel = max(VARIANTS, key=lambda v: (res["variants"][v]["val"]["wa_pct"], v == "VIT"))
    res["selected_on_val"] = sel
    print(f"\n selected on validation: {sel} (val {res['variants'][sel]['val']['wa_pct']:.2f}) "
          f"-> test {res['variants'][sel]['test']['wa_pct']:.2f}")

    # ---- stage 2: every variant on all six optical models (alpha per model
    # on validation); McNemar vs narrow uses the variant selected in stage 1 --
    res["models"] = {}
    print(f"\n {'model':<8s}{'greedy':>8s}{'KN3-left':>10s}" + "".join(f"{v:>12s}" for v in VARIANTS)
          + "   (test WA; alpha per model on val)")
    flags_all = {}
    preds_all = {}
    for mode in modes:
        if mode == "full":
            rw, rf, rs = raw, refs, rows
        else:
            rw, rf, rs = run_model(mode)
        gl = {al: score(vc.correct_lines(rs["val"], al, True), rf["val"])["wa_pct"] for al in alphas}
        bl = max(alphas, key=lambda x: (gl[x], -x))
        o_left = vc.correct_lines(rs["test"], bl, True)
        fl_left = [p == r for p, r in zip(o_left, rf["test"])]
        entry = {"alpha_left": bl, "test": {"greedy": score(rw["test"], rf["test"]),
                                             "KN3-left": score(o_left, rf["test"])}, "variants": {}}
        line = f" {mode:<8s}{entry['test']['greedy']['wa_pct']:>8.2f}{entry['test']['KN3-left']['wa_pct']:>10.2f}"
        for name, opts in VARIANTS.items():
            grid = {al: score(vc.decode_lines(rs["val"], al, **opts), rf["val"])["wa_pct"] for al in alphas}
            best = max(alphas, key=lambda x: (grid[x], -x))
            o_test = vc.decode_lines(rs["test"], best, **opts)
            fl = [p == r for p, r in zip(o_test, rf["test"])]
            b, c, pv = mcnemar_exact(fl_left, fl)
            st = score(o_test, rf["test"])
            entry["variants"][name] = {"selected_alpha_on_val": best,
                                       "val_alpha_grid": {str(x): grid[x] for x in alphas}, "test": st,
                                       "mcnemar_vs_KN3-left": {"only_left_correct": b, "only_variant_correct": c, "p_value": pv}}
            line += f"{st['wa_pct']:>12.2f}"
            if name == sel:
                flags_all[mode], preds_all[mode] = fl, o_test
        res["models"][mode] = entry
        print(line + f"   [{time.time()-t0:.0f}s]")
    opts = VARIANTS[sel]
    base = a.baseline
    res["mcnemar_vs_" + base] = {}
    print(f"\n exact McNemar against '{base}' (selected corrector)")
    for mode in modes:
        if mode == base or base not in flags_all:
            continue
        b, c, pv = mcnemar_exact(flags_all[base], flags_all[mode])
        res["mcnemar_vs_" + base][mode] = {"only_baseline_correct": b, "only_mode_correct": c,
                                          "delta_wa_pp": round(100 * (c - b) / len(flags_all[mode]), 4), "p_value": pv}
        print(f"   {mode:<8s} b={b:<5d} c={c:<5d} dWA={100*(c-b)/len(flags_all[mode]):+.2f} p={pv:.3g}")

    # determinism: decode the full model's test lines a second time
    again = vc.decode_lines(rows["test"], res["models"]["full"]["variants"][sel]["selected_alpha_on_val"], **opts)
    res["determinism_check"] = {"differing_words": sum(1 for x, y in zip(again, preds_all["full"]) if x != y), "n": len(again)}
    print(f"\n determinism: {res['determinism_check']['differing_words']} of {len(again)} differ on a second decode")

    dst = REPO_ROOT / a.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(dst, "w"), indent=2)
    ddir = REPO_ROOT / a.dump_preds
    ddir.mkdir(parents=True, exist_ok=True)
    for mode, preds in preds_all.items():
        with open(ddir / f"preds_{mode}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "word_id", "ground_truth", "prediction", "correct"])
            for i, (row, pr, rf_) in enumerate(zip(ids["test"], preds, refs["test"])):
                w.writerow([i, row[0], rf_, pr, int(pr == rf_)])
    with open(ddir / "preds_test_variants.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["word_id", "ground_truth", "greedy", "KN3-left"] + list(outputs))
        for i, (wid, _, _, gt) in enumerate(ids["test"]):
            w.writerow([wid, gt, raw["test"][i], ref_out[i]] + [outputs[k][i] for k in outputs])
    print(f" written: {dst}; per-word predictions in {ddir}   [{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
