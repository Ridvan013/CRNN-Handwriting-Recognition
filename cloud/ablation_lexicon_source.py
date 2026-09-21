#!/usr/bin/env python3
"""
Does the lexicon's *composition* matter more than its size?

Word beam search with the vocabulary of the IAM+Brown corpus (57 K types,
95.97 % coverage of the test tokens) beats the same decoder with the
239 K-type NLTK-extended list (93.94 % coverage).  That comparison confounds
the decoder with its dictionary, so this script gives our own post-corrector
the same three lexicons and measures it the same way:

  training       7,173 types   (IAM training transcriptions)
  extended     239,126 types   (IAM + the NLTK English word list)
  corpus        57,382 types   (vocabulary of the IAM training lines + Brown)

For each lexicon: the unigram prior, the left-to-right trigram and the
whole-line decoder with keep-OOV, with alpha selected on the validation
partition, scored once on the test partition.  Output:
results/ablation_lexicon_source.json

Usage:
    python cloud/ablation_lexicon_source.py --iam-words ... --iam-root ...
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
from kn_trigram import (KNTrigram, ContextCorrector, LineViterbiCorrector,
                        iam_line_segments, brown_sentences)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iam-words", default="")
    p.add_argument("--iam-root", default="")
    p.add_argument("--model", default="Model_abl_full")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--alphas-left", default="3,5,7,10")
    p.add_argument("--alphas-line", default="1,2,3,4,5,7,10,15,20,30")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--out", default="results/ablation_lexicon_source.json")
    p.add_argument("--dump-preds", default="results/preds_lexicon_source")
    return p.parse_args()


def with_vocabulary(base_path, vocab):
    """A TrigramLanguageModel whose lexicon is `vocab` (counts stay on IAM)."""
    lm = TrigramLanguageModel(base_path, use_nltk_extension=False)
    lm.vocabulary = set(vocab)
    lm.vocabulary_lower = {w.lower() for w in lm.vocabulary}
    for attr in ("_vocab_by_len", "_vocab_by_len_n"):
        lm.__dict__.pop(attr, None)
    return lm


def main():
    a = parse_args()
    al = [float(x) for x in a.alphas_left.split(",")]
    aline = [float(x) for x in a.alphas_line.split(",")]
    t0 = time.time()
    print("=" * 78)
    print(" Which lexicon? training / extended / corpus vocabulary, same corrector")
    print("=" * 78)

    (_, _, val_imgs, val_labs, test_imgs, test_labs) = load_iam_aachen(
        REPO_ROOT, iam_words_override=a.iam_words, iam_root_override=a.iam_root)
    img_root = a.iam_root or str(REPO_ROOT / "HTR_Using_CRNN/IAM/processed/archive/iam_words/words")
    ids = {"val": split_ids(REPO_ROOT / "aachen_splits" / "validation_words.txt", img_root),
           "test": split_ids(REPO_ROOT / "aachen_splits" / "test_words.txt", img_root)}
    loaders = {k: DataLoader(IAMDataset(i, l, is_training=False), batch_size=a.batch,
                             shuffle=False, collate_fn=custom_collate_fn)
               for k, (i, l) in {"val": (val_imgs, val_labs), "test": (test_imgs, test_labs)}.items()}
    model = CRNNModel(img_height=32, img_width=128, num_classes=len(CHAR_LIST) + 1).to(DEVICE)
    model.load_state_dict(torch.load(str(REPO_ROOT / a.model / "best_model_wa.pth"), map_location=DEVICE))
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
    lex_ext = TrigramLanguageModel(train_words, use_nltk_extension=True)
    lexicons = {
        "training (7K)": with_vocabulary(train_words, TrigramLanguageModel(train_words, use_nltk_extension=False).vocabulary),
        "extended (239K)": lex_ext,
        "corpus vocabulary (57K)": with_vocabulary(train_words, corpus_vocab),
    }
    res = {"model": a.model, "n_samples": len(refs["test"]), "topk": a.topk,
           "greedy": score(raw["test"], refs["test"]), "lexicons": {}}
    print(f" greedy: {res['greedy']['wa_pct']:.2f}")

    preds = {}
    for lname, lex in lexicons.items():
        test_cov = 100 * sum(1 for w in refs["test"]
                             if w in lex.vocabulary or w.lower() in lex.vocabulary_lower) / len(refs["test"])
        kn = KNTrigram(segs + brown, discount=0.75, extra_vocab=lex.vocabulary)
        cc = ContextCorrector(lex, kn)
        vc = LineViterbiCorrector(lex, kn, topk=a.topk)
        vc._cands = cc._cands
        entry = {"types": len(lex.vocabulary), "test_coverage_pct": round(test_cov, 2), "correctors": {}}
        print(f"\n {lname}: {len(lex.vocabulary):,} types, test coverage {test_cov:.2f}%")

        # edit distance only: the deterministic tie-break of ablation_lexicon.py
        from ablation_lexicon import correct_lexicon_only_fast
        memo = {}
        oe = {k: [correct_lexicon_only_fast(w, lex, memo) for w in raw[k]] for k in ("val", "test")}
        entry["correctors"]["edit distance only"] = {"alpha": 0.0,
                                                     "val": score(oe["val"], refs["val"]),
                                                     "test": score(oe["test"], refs["test"])}
        preds[f"{lname} | edit"] = oe["test"]
        print(f"   edit only          test {entry['correctors']['edit distance only']['test']['wa_pct']:.2f}"
              f"  CER {entry['correctors']['edit distance only']['test']['cer_pct']:.2f}   [{time.time()-t0:.0f}s]",
              flush=True)

        # unigram prior (the corrector of the earlier version), alpha = 5
        o = {k: [lex.correct_word(w) for w in raw[k]] for k in ("val", "test")}
        entry["correctors"]["unigram prior"] = {"alpha": 5.0, "val": score(o["val"], refs["val"]),
                                                "test": score(o["test"], refs["test"])}
        preds[f"{lname} | unigram"] = o["test"]
        print(f"   unigram prior      test {entry['correctors']['unigram prior']['test']['wa_pct']:.2f}"
              f"  CER {entry['correctors']['unigram prior']['test']['cer_pct']:.2f}   [{time.time()-t0:.0f}s]")

        # left-to-right trigram
        grid = {x: score(cc.correct_lines(rows["val"], x, True), refs["val"])["wa_pct"] for x in al}
        best = max(al, key=lambda x: (grid[x], -x))
        o_test = cc.correct_lines(rows["test"], best, True)
        entry["correctors"]["KN3 left-to-right"] = {"alpha": best, "val_grid": grid,
                                                    "test": score(o_test, refs["test"])}
        preds[f"{lname} | left"] = o_test
        print(f"   KN3 left-to-right  test {score(o_test, refs['test'])['wa_pct']:.2f}"
              f"  CER {score(o_test, refs['test'])['cer_pct']:.2f}  (alpha {best:g})   [{time.time()-t0:.0f}s]")

        # whole-line decoder with keep-OOV (the configuration of CRNN-LX)
        gridv = {x: score(vc.decode_lines(rows["val"], x, keep_oov=True), refs["val"])["wa_pct"] for x in aline}
        bestv = max(aline, key=lambda x: (gridv[x], -x))
        o_line = vc.decode_lines(rows["test"], bestv, keep_oov=True)
        entry["correctors"]["KN3 whole-line, keep-OOV"] = {"alpha": bestv, "val_grid": gridv,
                                                           "test": score(o_line, refs["test"])}
        preds[f"{lname} | line"] = o_line
        s = score(o_line, refs["test"])
        print(f"   KN3 whole-line     test {s['wa_pct']:.2f}  CER {s['cer_pct']:.2f}  "
              f"(alpha {bestv:g})   [{time.time()-t0:.0f}s]")
        res["lexicons"][lname] = entry

    # paired tests against WBS with the same corpus vocabulary, if available
    wbs_csv = REPO_ROOT / "results" / "preds_wbs_corpusvocab" / "preds_test.csv"
    if wbs_csv.exists():
        wrows = list(csv.DictReader(open(wbs_csv, encoding="utf-8")))
        col = next(c for c in wrows[0] if "corpus vocab" in c)
        wbs = [r[col] for r in wrows]
        assert all(r["ground_truth"] == g for r, g in zip(wrows, refs["test"]))
        fw = [p == g for p, g in zip(wbs, refs["test"])]
        res["mcnemar_vs_wbs_corpus_vocab"] = {}
        print("\n exact McNemar against WBS (Words, corpus vocabulary):")
        for name, out in preds.items():
            fo = [p == g for p, g in zip(out, refs["test"])]
            b, c, pv = mcnemar_exact(fo, fw)
            res["mcnemar_vs_wbs_corpus_vocab"][name] = {"only_ours_correct": b,
                                                        "only_wbs_correct": c, "p_value": pv}
            print(f"   {name:<40} only ours {b:<5d} only WBS {c:<5d} p={pv:.3g}")

    dst = REPO_ROOT / a.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(dst, "w"), indent=2)
    ddir = REPO_ROOT / a.dump_preds
    ddir.mkdir(parents=True, exist_ok=True)
    with open(ddir / "preds_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        names = list(preds)
        w.writerow(["idx", "ground_truth", "greedy"] + names)
        for i, g in enumerate(refs["test"]):
            w.writerow([i, g, raw["test"][i]] + [preds[n][i] for n in names])
    print(f"\n written: {dst}   [{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
