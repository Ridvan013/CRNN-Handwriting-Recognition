#!/usr/bin/env python3
"""
How many hypotheses does each lexicon leave out, and how many of those can
the corrector reach?

Section 3.5 of the paper quotes three counts for the proposed corrector --- the
hypotheses that fall outside the corpus lexicon, their share of the test set,
and how many of them have at least one candidate inside the edit bound.  They
are properties of the lexicon and of the greedy hypotheses, not of a GPU run,
so this script recomputes them from the dumped predictions and writes them to
a result file that cloud/verify_paper_numbers.py can check the paper against.

No model and no GPU: it reads the greedy column of
results/preds_final/preds_test_variants.csv and rebuilds the three lexicons of
Section 3.5 exactly as cloud/ablation_lexicon_source.py does.

Output: results/oov_hypotheses.json
Usage:  python cloud/count_oov_hypotheses.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cloud"))

from trigram_lm import TrigramLanguageModel                      # noqa: E402
from kn_trigram import (KNTrigram, ContextCorrector,              # noqa: E402
                        iam_line_segments, brown_sentences)

PREDS = ROOT / "results" / "preds_final" / "preds_test_variants.csv"
OUT = ROOT / "results" / "oov_hypotheses.json"


def with_vocabulary(train_words: str, vocab):
    """A TrigramLanguageModel whose lexicon is `vocab` (counts stay on IAM)."""
    lm = TrigramLanguageModel(train_words, use_nltk_extension=False)
    lm.vocabulary = set(vocab)
    lm.vocabulary_lower = {w.lower() for w in lm.vocabulary}
    for attr in ("_vocab_by_len", "_vocab_by_len_n"):
        lm.__dict__.pop(attr, None)
    return lm


def main() -> int:
    rows = list(csv.DictReader(open(PREDS, encoding="utf-8", newline="")))
    greedy = [r["greedy"] for r in rows]
    refs = [r["ground_truth"] for r in rows]
    n = len(rows)

    train_words = str(ROOT / "aachen_splits" / "train_words.txt")
    segs = iam_line_segments(train_words)
    brown = brown_sentences()
    corpus_vocab = set()
    for sent in segs + brown:
        corpus_vocab.update(sent)
    base = TrigramLanguageModel(train_words, use_nltk_extension=False)
    lexicons = {
        "training (7K)": with_vocabulary(train_words, base.vocabulary),
        "extended (239K)": TrigramLanguageModel(train_words, use_nltk_extension=True),
        "corpus vocabulary (57K)": with_vocabulary(train_words, corpus_vocab),
    }

    out = {"n_samples": n, "source": str(PREDS.relative_to(ROOT)), "lexicons": {}}
    for name, lex in lexicons.items():
        kn = KNTrigram(segs + brown, discount=0.75, extra_vocab=lex.vocabulary)
        cc = ContextCorrector(lex, kn)
        oov = [w for w in greedy if not cc.in_lexicon(w)]
        reachable = sum(1 for w in oov if cc.candidates(w))
        covered = sum(1 for w in refs
                      if w in lex.vocabulary or w.lower() in lex.vocabulary_lower)
        out["lexicons"][name] = {
            "types": len(lex.vocabulary),
            "test_coverage_pct": round(100 * covered / n, 2),
            "hypotheses_out_of_lexicon": len(oov),
            "hypotheses_out_of_lexicon_pct": round(100 * len(oov) / n, 2),
            "of_those_with_a_candidate": reachable,
        }
        e = out["lexicons"][name]
        print(f"{name:26s} {e['types']:>7,} types  coverage {e['test_coverage_pct']:5.2f}%"
              f"  out of lexicon {e['hypotheses_out_of_lexicon']:>5,}"
              f" ({e['hypotheses_out_of_lexicon_pct']:.2f}%)"
              f"  reachable {e['of_those_with_a_candidate']:>5,}")

    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
