"""
Trigram + Extended Vocabulary Test:
  - Vocabulary = IAM Aachen train (5,939) + NLTK English words (235,892)
  - Strategy: V2 tight (d=1/2/2, alpha=5.0)
  - Post-hoc test on V2 CSV
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import csv
import math
from collections import Counter

import nltk
try:
    from nltk.corpus import words as nltk_words_corpus
except LookupError:
    nltk.download('words', quiet=True)
    from nltk.corpus import words as nltk_words_corpus

CSV_PATH = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\Model_aachen_v2\test_results_analysis.csv"
TRAIN_WORDS_FILE = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\aachen_splits\train_words.txt"

# -----------------------------------------------------------
# Load vocabularies
# -----------------------------------------------------------
print("Loading IAM Aachen train vocabulary...")
iam_vocab = set()
unigram = Counter()
with open(TRAIN_WORDS_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 9:
            w = parts[-1]
            iam_vocab.add(w)
            unigram[w] += 1
print(f"  IAM vocab: {len(iam_vocab):,} unique words")

print("Loading NLTK English words corpus...")
nltk_vocab = set(nltk_words_corpus.words())
print(f"  NLTK vocab: {len(nltk_vocab):,} unique words")

extended_vocab = iam_vocab | nltk_vocab
# Also add lowercase versions for case-insensitive matching
extended_vocab_lower = {w.lower() for w in extended_vocab}
extended_vocab_full = extended_vocab | extended_vocab_lower

print(f"\nMerged extended vocab: {len(extended_vocab):,} unique words")
print(f"Coverage uplift: +{len(extended_vocab) - len(iam_vocab):,} words ({(len(extended_vocab)/len(iam_vocab) - 1)*100:.0f}% increase)")

# Length bucketing for fast candidate lookup
vocab_by_len = {}
for w in extended_vocab:
    vocab_by_len.setdefault(len(w), []).append(w)


# -----------------------------------------------------------
# Edit distance + correction
# -----------------------------------------------------------
def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        cur = [i + 1]
        for j, c2 in enumerate(s2):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (c1 != c2)))
        prev = cur
    return prev[-1]


total_iam = sum(unigram.values())
V = len(iam_vocab)


def unigram_log_prob(word):
    cnt = unigram.get(word, 0)
    return math.log((cnt + 1) / (total_iam + V))


def correct_extended(word):
    """V2 strategy with extended vocabulary."""
    # Check both original and lowercase against extended vocab
    if word in extended_vocab or word.lower() in extended_vocab_lower:
        return word  # Valid English word, don't touch

    # Tight edit distance bounds
    if len(word) <= 4:
        d_max = 1
    elif len(word) <= 8:
        d_max = 2
    else:
        d_max = 2

    cands = []
    for length, ws in vocab_by_len.items():
        if abs(length - len(word)) > d_max:
            continue
        for vw in ws:
            d = edit_distance(word, vw)
            if d <= d_max:
                # Score: prefer high-frequency IAM words, else just penalize edits
                if vw in unigram:
                    score = unigram_log_prob(vw) - d * 5.0
                else:
                    # NLTK-only words: assign baseline score
                    score = math.log(0.5 / (total_iam + V)) - d * 5.0
                cands.append((vw, score))

    if cands:
        cands.sort(key=lambda c: c[1], reverse=True)
        return cands[0][0]
    return word


# -----------------------------------------------------------
# Evaluate on V2 CSV
# -----------------------------------------------------------
print(f"\nLoading V2 test results...")
rows = []
with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
print(f"  {len(rows)} samples")

print("\nApplying extended trigram correction...")
helped = 0
hurt = 0
touched = 0
correct_extended_count = 0
correct_raw = 0
correct_old_smart = 0
for r in rows:
    raw = r["Raw_Predicted_Text"]
    old_smart = r["Predicted_Text"]
    true = r["True_Text"]

    new_pred = correct_extended(raw)
    r["ExtendedTrigram_Text"] = new_pred

    if raw == true:
        correct_raw += 1
    if old_smart == true:
        correct_old_smart += 1
    if new_pred == true:
        correct_extended_count += 1
    if new_pred != raw:
        touched += 1
        if raw == true and new_pred != true:
            hurt += 1
        elif raw != true and new_pred == true:
            helped += 1

N = len(rows)
print(f"\n{'='*72}")
print(f"  COMPARISON ON V2 AACHEN TEST SET ({N} samples)")
print(f"{'='*72}")
print(f"  Raw Greedy (no LM)              : {correct_raw:5d}/{N}  =  {correct_raw/N*100:6.2f}%")
print(f"  V2 Smart Trigram (IAM-only)     : {correct_old_smart:5d}/{N}  =  {correct_old_smart/N*100:6.2f}%  ({(correct_old_smart-correct_raw)/N*100:+.2f}pp)")
print(f"  V3 Extended Trigram (IAM+NLTK)  : {correct_extended_count:5d}/{N}  =  {correct_extended_count/N*100:6.2f}%  ({(correct_extended_count-correct_raw)/N*100:+.2f}pp)")
print(f"")
print(f"  Touched: {touched}, Helped: {helped}, Hurt: {hurt}, Net: {helped - hurt:+d}")
print(f"  Helped/Hurt ratio: {helped}/{hurt} = {helped/max(hurt,1):.2f}x")

# McNemar test
b = sum(1 for r in rows if r["Raw_Predicted_Text"] == r["True_Text"] and r["ExtendedTrigram_Text"] != r["True_Text"])
c = sum(1 for r in rows if r["Raw_Predicted_Text"] != r["True_Text"] and r["ExtendedTrigram_Text"] == r["True_Text"])
if (b + c) > 0:
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_chi = math.erfc(math.sqrt(chi2 / 2.0))
    # Exact binomial p
    n_disc = b + c
    k = min(b, c)
    p_exact = 2 * sum(math.comb(n_disc, i) * (0.5 ** n_disc) for i in range(k + 1))
    p_exact = min(p_exact, 1.0)
    print(f"\n  McNemar (Raw vs Extended): chi2={chi2:.2f}, p_asymp={p_chi:.3e}, p_exact={p_exact:.3e}, b={b}, c={c}")

# Save updated CSV
out_path = CSV_PATH
print(f"\nWriting updated CSV with ExtendedTrigram_Text column to {out_path}...")
fields = list(rows[0].keys())
with open(out_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
print("Done.")
