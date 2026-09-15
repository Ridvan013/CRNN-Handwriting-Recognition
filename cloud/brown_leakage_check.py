#!/usr/bin/env python3
"""
Does the Brown corpus contain the IAM test prompts?

IAM prompts were taken from the LOB corpus; Brown is a different (American,
1961) corpus, but both sample the same genres, so verbatim overlap has to be
measured rather than assumed.  For n = 3..8 the script counts the IAM test
line n-grams (contiguous ok-word runs, case-sensitive tokens, punctuation
included) that occur verbatim anywhere in Brown, and prints the most
frequent shared long n-grams so that idioms can be told apart from copied
sentences.  Output: results/brown_leakage.json

Usage:  python cloud/brown_leakage_check.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "cloud"))
from kn_trigram import iam_line_segments, brown_sentences  # noqa: E402


def ngrams(seqs, n):
    out = Counter()
    for s in seqs:
        for i in range(len(s) - n + 1):
            out[tuple(s[i:i + n])] += 1
    return out


def main():
    test = iam_line_segments(str(REPO_ROOT / "aachen_splits" / "test_words.txt"))
    train = iam_line_segments(str(REPO_ROOT / "aachen_splits" / "train_words.txt"))
    brown = brown_sentences()
    res = {"brown_sentences": len(brown), "brown_tokens": sum(map(len, brown)),
           "iam_test_segments": len(test), "iam_test_tokens": sum(map(len, test)),
           "per_order": {}}
    print(f"Brown: {len(brown):,} sentences, {res['brown_tokens']:,} tokens; "
          f"IAM test: {len(test):,} runs, {res['iam_test_tokens']:,} tokens")
    print(f"\n {'n':>2s} {'test n-gram tokens':>18s} {'in Brown':>9s} {'%':>6s}   "
          f"{'in IAM train':>12s} {'%':>6s}   (train column: same check against IAM training lines)")
    for n in range(3, 9):
        t = ngrams(test, n)
        b = set(ngrams(brown, n))
        tr = set(ngrams(train, n))
        tot = sum(t.values())
        hit_b = sum(c for g, c in t.items() if g in b)
        hit_tr = sum(c for g, c in t.items() if g in tr)
        top = [(" ".join(g), c) for g, c in t.most_common() if g in b][:8]
        res["per_order"][n] = {"test_ngram_tokens": tot, "found_in_brown": hit_b,
                               "pct_in_brown": round(100 * hit_b / tot, 3),
                               "found_in_iam_train": hit_tr,
                               "pct_in_iam_train": round(100 * hit_tr / tot, 3),
                               "most_frequent_shared_with_brown": top}
        print(f" {n:>2d} {tot:>18,d} {hit_b:>9,d} {100*hit_b/tot:>6.2f}   {hit_tr:>12,d} {100*hit_tr/tot:>6.2f}")
    for n in (5, 6, 8):
        print(f"\n most frequent test {n}-grams also in Brown:")
        for g, c in res["per_order"][n]["most_frequent_shared_with_brown"]:
            print(f"   {c:>3d}  {g}")
    # longest shared run: extend the per-order check upwards until nothing matches
    longest = max((n for n, r in res["per_order"].items() if r["found_in_brown"] > 0), default=0)
    n = 9
    while n < 60:
        b = set(ngrams(brown, n))
        if not any(g in b for g in ngrams(test, n)):
            break
        longest = n
        n += 1
    res["longest_shared_ngram"] = longest
    print(f"\n longest IAM-test word sequence that occurs verbatim in Brown: {longest} words")
    dst = REPO_ROOT / "results" / "brown_leakage.json"
    json.dump(res, open(dst, "w"), indent=2)
    print(" written:", dst)


if __name__ == "__main__":
    main()
