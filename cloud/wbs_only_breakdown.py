#!/usr/bin/env python3
"""Why does word beam search get right the words that CRNN-LX gets wrong?

Section 5.3 compares WBS on the corpus lexicon with CRNN-LX (corpus lexicon,
left-to-right trigram). For every test word that only WBS recognizes, this
script records why the post-corrector could not have produced it:

  * reference not in lexicon - the reference is absent from our lexicon (WBS
                      builds its dictionary from the corpus text with its own
                      tokenization, so a few words are in it and not in ours);
  * in lexicon      - the greedy string is itself a lexicon word, so the
                      corrector leaves it alone;
  * beyond bound    - the reference is further from the greedy string than
                      the edit bound (1 for words of up to four characters,
                      2 otherwise), so it is never a candidate;
  * ranked lower    - the reference was a candidate, but another lexicon
                      entry scored higher.

No GPU is needed: it reads the per-word predictions in results/.

    python cloud/wbs_only_breakdown.py   ->  results/wbs_only_breakdown.json
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cloud"))
from kn_trigram import iam_line_segments, brown_sentences  # noqa: E402

LX_COL = "corpus vocabulary (57K) | left"
WBS_COL = "WBS Words, corpus vocab"


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def main():
    vocab = set()
    for seg in iam_line_segments(str(ROOT / "aachen_splits" / "train_words.txt")) + brown_sentences():
        vocab.update(seg)
    lower = {w.lower() for w in vocab}
    assert len(vocab) == 57382, len(vocab)

    with open(ROOT / "results/preds_wbs_corpusvocab/preds_test.csv", encoding="utf-8") as f:
        wbs = list(csv.DictReader(f))
    with open(ROOT / "results/preds_lexicon_source/preds_test.csv", encoding="utf-8") as f:
        lex = list(csv.DictReader(f))
    assert len(wbs) == len(lex) == 20310

    counts = {"reference_not_in_lexicon": 0, "in_lexicon": 0,
              "beyond_bound": 0, "ranked_lower": 0}
    examples = {k: [] for k in counts}
    only_lx = 0
    for w, l in zip(wbs, lex):
        assert w["ground_truth"] == l["ground_truth"]
        ref, hyp, out = l["ground_truth"], l["greedy"], l[LX_COL]
        if out == ref and w[WBS_COL] != ref:
            only_lx += 1
        if w[WBS_COL] != ref or out == ref:
            continue
        if ref not in vocab and ref.lower() not in lower:
            key = "reference_not_in_lexicon"
        elif hyp in vocab or hyp.lower() in lower:
            key = "in_lexicon"
        elif lev(hyp.lower(), ref.lower()) > (1 if len(hyp) <= 4 else 2):
            key = "beyond_bound"
        else:
            key = "ranked_lower"
        counts[key] += 1
        examples[key].append([hyp, ref, out])

    # WBS keeps only the runs of letters of its dictionary (word_chars are the
    # letters), so it works with fewer distinct words than the entries passed.
    letter_runs = {r for w in vocab for r in re.findall(r"[A-Za-z]+", w)}
    res = {"corpus_lexicon_entries": len(vocab),
           "wbs_letter_run_words_corpus": len(letter_runs),
           "only_wbs_correct": sum(counts.values()), "only_lx_correct": only_lx,
           "only_wbs_by_cause": counts,
           "examples": {k: v[:40] for k, v in examples.items()}}
    out = ROOT / "results" / "wbs_only_breakdown.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("only_wbs_correct", "only_lx_correct",
                                          "only_wbs_by_cause")}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
