#!/usr/bin/env python3
"""
Every per-word statistic quoted in the paper, computed from the released
per-word predictions of the trigram-corrected systems (results/preds_trigram/)
and written to results/paper_stats_trigram.json so that each number in the
text has a machine-readable source.

  * pairwise agreement of the augmented models (disagreement counts, b/c)
  * error analysis of CRNN-LX: misrecognized words, case-only errors,
    substitutions / deletions / insertions from a Levenshtein alignment,
    ten most frequent character substitutions
  * what the corrector touched: in-lexicon hypotheses accepted, corrected
    hypotheses, fixed / broken counts against the unigram prior
  * context availability on the test lines

Usage:  python cloud/paper_stats_trigram.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import argparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "makale"))
_ap = argparse.ArgumentParser()
_ap.add_argument("--pred-dir", default="results/preds_viterbi",
                 help="per-model prediction dumps of the final corrector")
_ap.add_argument("--variants-csv", default="preds_test_variants.csv",
                 help="CRNN-LX per-word file with greedy and reference-corrector columns")
_ap.add_argument("--ref-col", default="KN3-left",
                 help="column of the previous corrector to compare the final one against")
_ap.add_argument("--out", default="results/paper_stats_trigram.json")
_args = _ap.parse_args()
PRED_DIR = REPO_ROOT / _args.pred_dir
MODES = ["none", "narrow", "photo", "elastic", "morph", "full"]


def load(mode):
    with open(PRED_DIR / f"preds_{mode}.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def align_ops(pred: str, true: str):
    """(substitutions, deletions, insertions) and substitution pairs, from a
    Levenshtein alignment; ties broken in the order substitution, insertion,
    deletion (same backtrace as makale/generate_figures.py)."""
    m, n = len(true), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if true[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    i, j = m, n
    sub = dele = ins = 0
    pairs = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and true[i - 1] == pred[j - 1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            pairs.append((true[i - 1], pred[j - 1])); sub += 1; i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1; j -= 1
        else:
            dele += 1; i -= 1
    return sub, dele, ins, pairs


def main():
    preds = {m: load(m) for m in MODES}
    n = len(preds["full"])
    for m in MODES:
        assert len(preds[m]) == n and all(a["word_id"] == b["word_id"] for a, b in zip(preds[m], preds["full"]))
    flags = {m: [r["correct"] == "1" for r in preds[m]] for m in MODES}
    out = {"n": n, "source": str(PRED_DIR.relative_to(REPO_ROOT))}

    # ---- pairwise disagreement -------------------------------------------
    aug = ["narrow", "photo", "elastic", "morph", "full"]
    all_same = sum(1 for i in range(n) if len({flags[m][i] for m in aug}) == 1)
    all_right = sum(1 for i in range(n) if all(flags[m][i] for m in aug))
    all_wrong = sum(1 for i in range(n) if not any(flags[m][i] for m in aug))
    pair = {}
    for a in MODES:
        for b in MODES:
            if a < b:
                only_a = sum(1 for i in range(n) if flags[a][i] and not flags[b][i])
                only_b = sum(1 for i in range(n) if flags[b][i] and not flags[a][i])
                pair[f"{a}|{b}"] = {"only_" + a: only_a, "only_" + b: only_b}
    out["augmented_five"] = {"words_where_all_agree": all_same, "all_correct": all_right,
                             "all_wrong": all_wrong, "disagree": n - all_same,
                             "disagree_pct": round(100 * (n - all_same) / n, 1)}
    out["pairwise_only_correct"] = pair
    print(f"five augmented models: disagree on {n-all_same:,} words ({100*(n-all_same)/n:.1f}%); "
          f"all correct {all_right:,}, all wrong {all_wrong:,}")
    for k in ("narrow|none", "full|narrow"):
        a, b = k.split("|")
        print(f"  {a} vs {b}: only {a} {pair[k]['only_'+a]}, only {b} {pair[k]['only_'+b]}")

    # ---- error analysis of CRNN-LX -----------------------------------------
    full = preds["full"]
    wrong = [r for r in full if r["correct"] != "1"]
    case_only = sum(1 for r in wrong if r["prediction"].lower() == r["ground_truth"].lower())
    S = D = I = 0
    pairs = Counter()
    for r in wrong:
        s, d, i, p = align_ops(r["prediction"], r["ground_truth"])
        S += s; D += d; I += i
        pairs.update(p)
    case_sub = sum(c for (t, p), c in pairs.items() if t != p and t.lower() == p.lower())
    sym = Counter()
    for (t, p), c in pairs.items():
        sym[tuple(sorted((t, p)))] += c
    top10 = pairs.most_common(10)
    out["error_analysis_full"] = {
        "misrecognized": len(wrong), "misrecognized_pct": round(100 * len(wrong) / n, 2),
        "case_only": case_only, "case_only_pct_of_errors": round(100 * case_only / len(wrong), 1),
        "substitutions": S, "deletions": D, "insertions": I,
        "case_substitutions": case_sub, "case_substitutions_pct": round(100 * case_sub / S, 1),
        "top10_substitutions_true_to_pred": [[t, p, c] for (t, p), c in top10],
        "top_symmetric_confusions": [[a, b, c] for (a, b), c in sym.most_common(8)]}
    print(f"CRNN-LX: {len(wrong):,} wrong ({100*len(wrong)/n:.2f}%), case-only {case_only} "
          f"({100*case_only/len(wrong):.1f}%), S/D/I = {S}/{D}/{I}, case subs {case_sub} ({100*case_sub/S:.1f}%)")
    print("  top-10 subs:", ", ".join(f"{t}->{p} {c}" for (t, p), c in top10))
    print("  symmetric  :", ", ".join(f"{a}<->{b} {c}" for (a, b), c in sym.most_common(8)))

    # ---- what the corrector did (full model): final vs previous corrector ----
    ref = _args.ref_col
    tri = {}
    with open(PRED_DIR / _args.variants_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tri[r["word_id"]] = r
    assert len(tri) == n
    g_ok = sum(1 for r in full if tri[r["word_id"]]["greedy"] == r["ground_truth"])
    changed = sum(1 for r in full if tri[r["word_id"]]["greedy"] != r["prediction"])
    fixed = sum(1 for r in full if tri[r["word_id"]]["greedy"] != r["ground_truth"] and r["correct"] == "1")
    broken = sum(1 for r in full if tri[r["word_id"]]["greedy"] == r["ground_truth"] and r["correct"] != "1")
    u_ok = sum(1 for r in full if tri[r["word_id"]][ref] == r["ground_truth"])
    only_u = sum(1 for r in full if tri[r["word_id"]][ref] == r["ground_truth"] and r["correct"] != "1")
    only_t = sum(1 for r in full if tri[r["word_id"]][ref] != r["ground_truth"] and r["correct"] == "1")
    diff_ut = sum(1 for r in full if tri[r["word_id"]][ref] != r["prediction"])
    # how the reference corrector itself treated the greedy output
    r_changed = sum(1 for r in full if tri[r["word_id"]]["greedy"] != tri[r["word_id"]][ref])
    r_fixed = sum(1 for r in full if tri[r["word_id"]]["greedy"] != r["ground_truth"] and tri[r["word_id"]][ref] == r["ground_truth"])
    r_broken = sum(1 for r in full if tri[r["word_id"]]["greedy"] == r["ground_truth"] and tri[r["word_id"]][ref] != r["ground_truth"])
    out["corrector_full"] = {"greedy_correct": g_ok, "hypotheses_changed": changed,
                             "changed_pct": round(100 * changed / n, 2),
                             "fixed": fixed, "broken": broken, "net": fixed - broken,
                             "reference_corrector": ref, "reference_correct": u_ok,
                             "reference_changed": r_changed, "reference_fixed": r_fixed, "reference_broken": r_broken,
                             "final_vs_reference": {"outputs_differ": diff_ut, "only_reference_correct": only_u,
                                                    "only_final_correct": only_t}}
    print(f"final corrector: changed {changed:,} hypotheses ({100*changed/n:.2f}%), fixed {fixed}, broke {broken}, "
          f"net {fixed-broken}; {ref}: changed {r_changed}, fixed {r_fixed}, broke {r_broken}; "
          f"final vs {ref}: differ {diff_ut}, only-{ref} {only_u}, only-final {only_t}")

    # ---- context availability -----------------------------------------------
    ids = {(r["word_id"].rsplit("-", 1)[0], int(r["word_id"].rsplit("-", 1)[1])) for r in full}
    with_prev = sum(1 for (l, i) in ids if (l, i - 1) in ids)
    with_two = sum(1 for (l, i) in ids if (l, i - 1) in ids and (l, i - 2) in ids)
    out["context"] = {"with_previous_word_pct": round(100 * with_prev / n, 1),
                      "with_two_previous_words_pct": round(100 * with_two / n, 1)}
    print(f"context: {100*with_prev/n:.1f}% of test words have a preceding word on the line, "
          f"{100*with_two/n:.1f}% have two")

    dst = REPO_ROOT / _args.out
    json.dump(out, open(dst, "w"), indent=2)
    print("written:", dst)


if __name__ == "__main__":
    main()
