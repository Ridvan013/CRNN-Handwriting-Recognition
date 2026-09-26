#!/usr/bin/env python3
"""
Word beam search (WBS) on the same optical model, as a decoder-side baseline
for the post-corrector of Section 3.5.

WBS constrains CTC beam search to dictionary words while decoding, so it is
the natural alternative to correcting the greedy output afterwards.  We use
the authors' own implementation (githubharald/CTCWordBeamSearch, installed as
the `word_beam_search` package) on the probabilities of the CRNN-LX
checkpoint, with the lexicons of Table 4:

  Words  / 7 K     dictionary = training vocabulary, no language model
  Words  / 239 K   dictionary = extended lexicon, no language model
  NGrams / corpus  dictionary = vocabulary of the IAM+Brown text, plus the
                   word bigram LM that WBS estimates from that text

WBS builds its dictionary by splitting the corpus at non-word characters, and
it emits dictionary words verbatim, so a capitalized token can only be
produced if the dictionary holds that form.  Our post-corrector tests
membership case-insensitively, which would make the comparison unfair, so
`--case-variants` adds the lower-case and capitalized form of every entry to
the WBS dictionary (reported alongside the raw-lexicon run).

Usage:
    python cloud/ablation_wbs.py --iam-words ... --iam-root ... [--limit 256]
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

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_v3 import (DEVICE, CRNNModel, CHAR_LIST, BLANK_TOKEN, IAMDataset,
                      custom_collate_fn, PAD_TOKEN)
from trigram_lm import TrigramLanguageModel
from v3_augmented_train import load_iam_aachen
from ablation_lexicon import score
from ablation_lexicon_all import mcnemar_exact
from kn_trigram import iam_line_segments, brown_sentences

WORD_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iam-words", default="")
    p.add_argument("--iam-root", default="")
    p.add_argument("--model", default="Model_abl_full")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--beam", type=int, default=25)
    p.add_argument("--smoothing", type=float, default=0.01)
    p.add_argument("--limit", type=int, default=0, help="only the first N words (timing runs)")
    p.add_argument("--only", default="", help="substring filter on config names")
    p.add_argument("--merge", action="store_true", help="merge into an existing --out json")
    p.add_argument("--out", default="results/ablation_wbs.json")
    p.add_argument("--dump-preds", default="results/preds_wbs")
    return p.parse_args()


def dictionary_corpus(words, case_variants: bool):
    """WBS reads its dictionary out of a corpus; for the dictionary-only modes
    the corpus is just the word list."""
    out = set()
    for w in words:
        out.add(w)
        if case_variants:
            out.add(w.lower())
            if w[:1].isalpha():
                out.add(w[:1].upper() + w[1:].lower())
    return " ".join(sorted(out)), len(out)


def probabilities(model, loader, limit=0):
    """Softmax probabilities per sample, plus references, fp32 deterministic."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    mats, refs = [], []
    with torch.no_grad():
        for images, labels in loader:
            lp = model(images.to(DEVICE, non_blocking=True))       # (T, B, C) log-softmax
            p = lp.exp().permute(1, 0, 2).cpu().numpy().astype(np.float32)   # (B, T, C)
            mats.extend(list(p))
            for t in labels:
                refs.append("".join(CHAR_LIST[i] for i in t.tolist() if i != PAD_TOKEN)
                            if torch.is_tensor(t) else "")
            if limit and len(refs) >= limit:
                break
    if limit:
        mats, refs = mats[:limit], refs[:limit]
    return mats, refs


def run_wbs(mats, corpus, mode, beam, smoothing, batch=64):
    from word_beam_search import WordBeamSearch
    wbs = WordBeamSearch(beam, mode, smoothing, corpus.encode("utf8"),
                         CHAR_LIST.encode("utf8"), WORD_CHARS.encode("utf8"))
    out = []
    t0 = time.time()
    for i in range(0, len(mats), batch):
        chunk = mats[i:i + batch]
        mat = np.stack(chunk, axis=1)            # (T, B, C), blank last
        for seq in wbs.compute(mat):
            out.append("".join(CHAR_LIST[j] for j in seq))
        if i % (batch * 20) == 0:
            done = min(i + batch, len(mats))
            rate = done / max(time.time() - t0, 1e-6)
            print(f"    {done}/{len(mats)}  {rate:.1f} words/s", flush=True)
    return out, time.time() - t0


def main():
    a = parse_args()
    t0 = time.time()
    print("=" * 78)
    print(" Word beam search as a decoder-side baseline (reference implementation)")
    print("=" * 78)
    (_, _, _, _, test_imgs, test_labs) = load_iam_aachen(
        REPO_ROOT, iam_words_override=a.iam_words, iam_root_override=a.iam_root)
    loader = DataLoader(IAMDataset(test_imgs, test_labs, is_training=False), batch_size=a.batch,
                        shuffle=False, collate_fn=custom_collate_fn)
    model = CRNNModel(img_height=32, img_width=128, num_classes=len(CHAR_LIST) + 1).to(DEVICE)
    model.load_state_dict(torch.load(str(REPO_ROOT / a.model / "best_model_wa.pth"), map_location=DEVICE))
    model.eval()
    assert BLANK_TOKEN == len(CHAR_LIST), "WBS needs the blank as the last class"
    mats, refs = probabilities(model, loader, a.limit)
    print(f" {len(mats)} words, T={mats[0].shape[0]}, C={mats[0].shape[1]}  [{time.time()-t0:.0f}s]")

    train_words = str(REPO_ROOT / "aachen_splits" / "train_words.txt")
    lex_iam = TrigramLanguageModel(train_words, use_nltk_extension=False)
    lex_ext = TrigramLanguageModel(train_words, use_nltk_extension=True)
    iam_text = " ".join(" ".join(s) for s in iam_line_segments(train_words))
    brown_text = " ".join(" ".join(s) for s in brown_sentences())

    corpus_vocab = set()
    for seg in iam_line_segments(train_words):
        corpus_vocab.update(seg)
    for seg in brown_sentences():
        corpus_vocab.update(seg)

    configs = [
        # dictionary = the vocabulary of the LM corpus, no LM: separates the
        # coverage of that vocabulary from what the bigram LM adds
        ("WBS Words, corpus vocab", "Words", dictionary_corpus(corpus_vocab, False)),
        # the corpus vocabulary already carries the case people write, but the
        # other two dictionaries get a case-variant row, so it gets one too
        ("WBS Words, corpus vocab, case variants", "Words",
         dictionary_corpus(corpus_vocab, True)),
        ("WBS Words, 7K", "Words", dictionary_corpus(lex_iam.vocabulary, False)),
        ("WBS Words, 7K, case variants", "Words", dictionary_corpus(lex_iam.vocabulary, True)),
        ("WBS Words, 239K", "Words", dictionary_corpus(lex_ext.vocabulary, False)),
        ("WBS Words, 239K, case variants", "Words", dictionary_corpus(lex_ext.vocabulary, True)),
        ("WBS NGrams, IAM+Brown", "NGrams", (iam_text + " " + brown_text, None)),
    ]
    res = {"model": a.model, "beam": a.beam, "smoothing": a.smoothing,
           "n_samples": len(refs), "blank_index": BLANK_TOKEN,
           "word_chars": WORD_CHARS, "configs": {}}
    if a.only:
        configs = [c for c in configs if a.only in c[0]]
        print(" filter:", [c[0] for c in configs])
    preds = {}
    greedy = None
    for name, mode, (corpus, size) in configs:
        print(f"\n {name}  ({mode}, dictionary corpus {len(corpus)/1e6:.1f} MB"
              + (f", {size:,} entries)" if size else ", text corpus)"), flush=True)
        out, secs = run_wbs(mats, corpus, mode, a.beam, a.smoothing)
        s = score(out, refs)
        res["configs"][name] = {"mode": mode, "dictionary_entries": size,
                                "test": s, "seconds": round(secs, 1),
                                "words_per_second": round(len(out) / secs, 1)}
        preds[name] = out
        print(f"   WA {s['wa_pct']:.4f}  CER {s['cer_pct']:.4f}  "
              f"CI [{s['wilson_95ci_pct'][0]:.2f},{s['wilson_95ci_pct'][1]:.2f}]  "
              f"{secs:.0f}s ({len(out)/secs:.1f} words/s)")

    # our corrector on the same samples, for the paired tests
    ours_csv = REPO_ROOT / "results" / "preds_viterbi" / "preds_full.csv"
    if ours_csv.exists() and not a.limit:
        rows = list(csv.DictReader(open(ours_csv, encoding="utf-8")))
        assert len(rows) == len(refs)
        assert all(r["ground_truth"] == g for r, g in zip(rows, refs)), "misaligned references"
        ours = [r["prediction"] for r in rows]
        greedy_csv = REPO_ROOT / "results" / "preds_trigram" / "preds_test.csv"
        greedy = [r["greedy"] for r in csv.DictReader(open(greedy_csv, encoding="utf-8"))]
        res["reference"] = {"ours (CRNN-LX)": score(ours, refs), "greedy": score(greedy, refs)}
        fo = [p == g for p, g in zip(ours, refs)]
        res["mcnemar_vs_ours"] = {}
        print("\n exact McNemar against our corrector (CRNN-LX, whole-line keep-OOV):")
        for name, out in preds.items():
            fw = [p == g for p, g in zip(out, refs)]
            b, c, pv = mcnemar_exact(fo, fw)
            res["mcnemar_vs_ours"][name] = {"only_ours_correct": b, "only_wbs_correct": c,
                                            "p_value": pv}
            print(f"   {name:<32} only ours {b:<5d} only WBS {c:<5d} p={pv:.3g}")

    dst = REPO_ROOT / a.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    if a.merge and dst.exists():
        prev = json.load(open(dst, encoding="utf-8"))
        prev["configs"].update(res["configs"])
        prev.setdefault("mcnemar_vs_ours", {}).update(res.get("mcnemar_vs_ours", {}))
        res = prev
    json.dump(res, open(dst, "w"), indent=2)
    if not a.limit:
        ddir = REPO_ROOT / a.dump_preds
        ddir.mkdir(parents=True, exist_ok=True)
        csv_path = ddir / "preds_test.csv"
        names = list(preds)
        # --merge must merge the per-word dump too: a run filtered with --only
        # would otherwise drop the columns of every configuration it skipped.
        if a.merge and csv_path.exists():
            with open(csv_path, encoding="utf-8", newline="") as f:
                prev_rows = list(csv.DictReader(f))
            if len(prev_rows) == len(refs) and all(
                    r["ground_truth"] == g for r, g in zip(prev_rows, refs)):
                kept = [c for c in prev_rows[0]
                        if c not in ("idx", "ground_truth") and c not in preds]
                for c in kept:
                    preds[c] = [r[c] for r in prev_rows]
                names = kept + names
            else:
                print("  ! existing preds_test.csv does not line up; not merged")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "ground_truth"] + names)
            for i, g in enumerate(refs):
                w.writerow([i, g] + [preds[n][i] for n in names])
    print(f"\n written: {dst}   [{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
