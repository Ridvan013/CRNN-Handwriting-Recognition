#!/usr/bin/env python3
"""
Split the jump of the whole-line decoder into its two ingredients.

Table 4 of the paper compares four correctors.  The fourth one changes two
things at once with respect to the third: it decodes the whole line jointly
(exact second-order Viterbi instead of left to right) *and* it may leave an
out-of-lexicon hypothesis untouched (keep-OOV).  With the 7 K training lexicon
that column jumps by 3.77 pp over the left-to-right decoder, and Section 6.2
attributes the jump to the freedom not to replace.  This script measures how
much of it actually comes from keep-OOV, by scoring the same whole-line
decoder with the option switched off.

Protocol is the one used everywhere else: alpha is selected on the validation
partition from the same grid, then applied once to the test partition.  The
keep-OOV column is re-scored at its stored alpha as a reproduction check --
it must return exactly the number already in ablation_lexicon_source.json.

Output: results/ablation_keep_oov.json
Usage:  python cloud/ablation_keep_oov.py [--model Model_abl_full]
"""
from __future__ import annotations

import argparse
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
from ablation_lexicon_all import hypotheses
from ablation_trigram import split_ids
from kn_trigram import (KNTrigram, LineViterbiCorrector, ContextCorrector,
                        iam_line_segments, brown_sentences)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iam-words", default="")
    p.add_argument("--iam-root", default="")
    p.add_argument("--model", default="Model_abl_full")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--alphas-line", default="1,2,3,4,5,7,10,15,20,30")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--out", default="results/ablation_keep_oov.json")
    return p.parse_args()


def with_vocabulary(base_path, vocab):
    lm = TrigramLanguageModel(base_path, use_nltk_extension=False)
    lm.vocabulary = set(vocab)
    lm.vocabulary_lower = {w.lower() for w in lm.vocabulary}
    for attr in ("_vocab_by_len", "_vocab_by_len_n"):
        lm.__dict__.pop(attr, None)
    return lm


def main():
    a = parse_args()
    aline = [float(x) for x in a.alphas_line.split(",")]
    t0 = time.time()
    print("=" * 78)
    print(" How much of the whole-line jump is keep-OOV?")
    print("=" * 78)

    (_, _, val_imgs, val_labs, test_imgs, test_labs) = load_iam_aachen(
        REPO_ROOT, iam_words_override=a.iam_words, iam_root_override=a.iam_root)
    img_root = a.iam_root or str(
        REPO_ROOT / "HTR_Using_CRNN/IAM/processed/archive/iam_words/words")
    ids = {"val": split_ids(REPO_ROOT / "aachen_splits" / "validation_words.txt", img_root),
           "test": split_ids(REPO_ROOT / "aachen_splits" / "test_words.txt", img_root)}
    loaders = {k: DataLoader(IAMDataset(i, l, is_training=False), batch_size=a.batch,
                             shuffle=False, collate_fn=custom_collate_fn)
               for k, (i, l) in {"val": (val_imgs, val_labs),
                                 "test": (test_imgs, test_labs)}.items()}
    model = CRNNModel(img_height=32, img_width=128,
                      num_classes=len(CHAR_LIST) + 1).to(DEVICE)
    model.load_state_dict(torch.load(str(REPO_ROOT / a.model / "best_model_wa.pth"),
                                     map_location=DEVICE))
    model.eval()
    raw, refs, rows = {}, {}, {}
    for k in ("val", "test"):
        raw[k], refs[k] = hypotheses(model, loaders[k])
        assert all(r == row[3] for r, row in zip(refs[k], ids[k])), f"{k}: misaligned"
        rows[k] = [(lid, idx, h) for (_, lid, idx, _), h in zip(ids[k], raw[k])]
    del model
    torch.cuda.empty_cache()
    print(f" hypotheses ready  [{time.time()-t0:.0f}s]")

    train_words = str(REPO_ROOT / "aachen_splits" / "train_words.txt")
    segs = iam_line_segments(train_words)
    brown = brown_sentences()
    corpus_vocab = set()
    for s in segs + brown:
        corpus_vocab.update(s)
    lexicons = {
        "training (7K)": with_vocabulary(
            train_words,
            TrigramLanguageModel(train_words, use_nltk_extension=False).vocabulary),
        "corpus vocabulary (57K)": with_vocabulary(train_words, corpus_vocab),
    }

    # the numbers already in the paper, for the reproduction check
    src = json.load(open(REPO_ROOT / "results" / "ablation_lexicon_source.json",
                         encoding="utf-8"))
    res = {"model": a.model, "n_samples": len(refs["test"]), "topk": a.topk,
           "greedy": score(raw["test"], refs["test"]), "lexicons": {}}

    for lname, lex in lexicons.items():
        kn = KNTrigram(segs + brown, discount=0.75, extra_vocab=lex.vocabulary)
        cc = ContextCorrector(lex, kn)
        vc = LineViterbiCorrector(lex, kn, topk=a.topk)
        vc._cands = cc._cands                      # share the candidate memo
        known = src["lexicons"][lname]["correctors"]
        entry = {"types": len(lex.vocabulary),
                 "left_to_right": known["KN3 left-to-right"]["test"],
                 "keep_oov_reported": known["KN3 whole-line, keep-OOV"]["test"]}
        print(f"\n {lname}: {len(lex.vocabulary):,} types")

        # (a) whole-line WITHOUT keep-OOV -- the new measurement
        grid = {x: score(vc.decode_lines(rows["val"], x, keep_oov=False),
                         refs["val"])["wa_pct"] for x in aline}
        best = max(aline, key=lambda x: (grid[x], -x))
        o = vc.decode_lines(rows["test"], best, keep_oov=False)
        entry["whole_line_forced"] = {"alpha": best, "val_grid": grid,
                                      "test": score(o, refs["test"])}
        print(f"   whole-line, forced replacement   test "
              f"{entry['whole_line_forced']['test']['wa_pct']:.2f}"
              f"  (alpha {best:g})   [{time.time()-t0:.0f}s]", flush=True)

        # (b) whole-line WITH keep-OOV at its stored alpha -- reproduction check
        astored = known["KN3 whole-line, keep-OOV"]["alpha"]
        o2 = vc.decode_lines(rows["test"], astored, keep_oov=True)
        s2 = score(o2, refs["test"])
        entry["whole_line_keep_oov"] = {"alpha": astored, "test": s2}
        ok = abs(s2["wa_pct"] - known["KN3 whole-line, keep-OOV"]["test"]["wa_pct"]) < 1e-9
        entry["reproduces_paper"] = bool(ok)
        print(f"   whole-line, keep-OOV             test {s2['wa_pct']:.2f}"
              f"  (alpha {astored:g})  reproduces paper: {ok}")

        # the split
        lr = entry["left_to_right"]["wa_pct"]
        forced = entry["whole_line_forced"]["test"]["wa_pct"]
        koov = s2["wa_pct"]
        entry["jump_total_pp"] = round(koov - lr, 4)
        entry["from_line_decoding_pp"] = round(forced - lr, 4)
        entry["from_keep_oov_pp"] = round(koov - forced, 4)
        print(f"   -> left-to-right {lr:.2f}  ->  whole-line {forced:.2f} "
              f"({forced-lr:+.2f})  ->  +keep-OOV {koov:.2f} ({koov-forced:+.2f})")
        res["lexicons"][lname] = entry

    dst = REPO_ROOT / a.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(dst, "w"), indent=2)
    print(f"\n written: {dst}   [{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
