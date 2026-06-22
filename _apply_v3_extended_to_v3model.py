"""
V3 Model CSV'sine V3 Extended Trigram (IAM+NLTK) post-hoc uygula.
Hem greedy hem beam sutununa ayri ayri uygulanir.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import csv
import math
from collections import Counter

import nltk
try:
    from nltk.corpus import words as nltk_words
    _ = nltk_words.words()
except LookupError:
    nltk.download('words', quiet=True)
    from nltk.corpus import words as nltk_words

CSV_PATH = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\Model_aachen_v3\test_results_analysis.csv"
TRAIN_WORDS = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\aachen_splits\train_words.txt"

# Load IAM vocab + unigram
iam_vocab = set()
unigram = Counter()
with open(TRAIN_WORDS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 9:
            w = parts[-1]
            iam_vocab.add(w)
            unigram[w] += 1

nltk_vocab = set(nltk_words.words())
extended_vocab = iam_vocab | nltk_vocab
extended_vocab_lower = {w.lower() for w in extended_vocab}

vocab_by_len = {}
for w in extended_vocab:
    vocab_by_len.setdefault(len(w), []).append(w)

total_iam = sum(unigram.values())
V = len(iam_vocab)

print(f"Extended vocab: {len(extended_vocab):,} (IAM {len(iam_vocab):,} + NLTK {len(nltk_vocab):,})")


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


def correct_extended(word):
    if word in extended_vocab or word.lower() in extended_vocab_lower:
        return word

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
                if vw in unigram:
                    score = math.log((unigram[vw] + 1) / (total_iam + V)) - d * 5.0
                else:
                    score = math.log(0.5 / (total_iam + V)) - d * 5.0
                cands.append((vw, score))
    if cands:
        cands.sort(key=lambda c: c[1], reverse=True)
        return cands[0][0]
    return word


# Load CSV
with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
fields = list(rows[0].keys())
print(f"Loaded {len(rows)} samples")

# Apply Extended Trigram to raw greedy and beam columns
print("Applying V3 Extended Trigram to raw greedy and beam...")
for r in rows:
    r["ExtendedTrigram_Text"] = correct_extended(r["Raw_Predicted_Text"])
    r["ExtendedTrigram_Beam_Text"] = correct_extended(r["Beam_Predicted_Text"])

for new_col in ["ExtendedTrigram_Text", "ExtendedTrigram_Beam_Text"]:
    if new_col not in fields:
        fields.append(new_col)

# Summary
N = len(rows)
raw_g = sum(1 for r in rows if r["Raw_Predicted_Text"] == r["True_Text"])
raw_b = sum(1 for r in rows if r["Beam_Predicted_Text"] == r["True_Text"])
old_t = sum(1 for r in rows if r["Predicted_Text"] == r["True_Text"])
old_bt = sum(1 for r in rows if r["Beam_Trigram_Text"] == r["True_Text"])
ext_g = sum(1 for r in rows if r["ExtendedTrigram_Text"] == r["True_Text"])
ext_b = sum(1 for r in rows if r["ExtendedTrigram_Beam_Text"] == r["True_Text"])

print(f"\n{'='*72}")
print(f"  V3 MODEL ON AACHEN TEST ({N} samples)")
print(f"{'='*72}")
print(f"  Raw Greedy                  : {raw_g:5d}/{N}  =  {raw_g/N*100:6.2f}%")
print(f"  Raw Beam k=10               : {raw_b:5d}/{N}  =  {raw_b/N*100:6.2f}%")
print(f"  Greedy + In-loop Trigram    : {old_t:5d}/{N}  =  {old_t/N*100:6.2f}%  (+{(old_t-raw_g)/N*100:.2f}pp)")
print(f"  Beam + In-loop Trigram      : {old_bt:5d}/{N}  =  {old_bt/N*100:6.2f}%  (+{(old_bt-raw_b)/N*100:.2f}pp)")
print(f"  Greedy + V3 Extended (NLTK) : {ext_g:5d}/{N}  =  {ext_g/N*100:6.2f}%  (+{(ext_g-raw_g)/N*100:.2f}pp)")
print(f"  Beam + V3 Extended (NLTK)   : {ext_b:5d}/{N}  =  {ext_b/N*100:6.2f}%  (+{(ext_b-raw_g)/N*100:.2f}pp vs raw greedy)")

# Wilson CI for best
import math as _m
def wilson(k, n, z=1.96):
    p = k/n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = z * _m.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (center - margin, center + margin)

best_n = max(old_t, old_bt, ext_g, ext_b)
ci = wilson(best_n, N)
print(f"\n  Best config: {best_n}/{N} = {best_n/N*100:.2f}%  Wilson 95% CI [{ci[0]*100:.2f}, {ci[1]*100:.2f}]")

# Write back
with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
print(f"\nCSV updated with Extended Trigram columns")
